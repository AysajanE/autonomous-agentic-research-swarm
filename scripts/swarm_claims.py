"""Claim refs: the atomic claim/lease primitive (plan §4.1).

A claim is a chain of empty-tree commits under ``refs/swarm/claims/<task_id>``,
one commit per lease renewal, each carrying a JSON payload in its message.
Compare-and-swap semantics come from git itself:

- CREATE: a plain push of a new ref is rejected when the ref already exists
  with unrelated history — rejection means the claim was lost, atomically.
- RENEW: a child commit fast-forwards the ref; ``--force-with-lease`` names
  the expected parent so racing renewers cannot both win.
- RELEASE: a guarded deletion; only the holder of the expected tip (or the
  reaper, after expiry proof) may delete.

The task-file claim fields are an eventually-consistent PROJECTION of the
ref (reconciled by sweep); the ref is authoritative for liveness. When no
remote is configured the same protocol runs against the local ref store and
``transport`` is recorded as ``local`` — single-host correctness only.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from pathlib import Path
import subprocess
import uuid


CLAIM_SCHEMA_VERSION = "research_swarm.claim.v1"
CLAIM_REF_PREFIX = "refs/swarm/claims/"
_EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
DEFAULT_LEASE_TTL_SECONDS = 3600


@dataclasses.dataclass(frozen=True)
class ClaimResult:
    ok: bool
    task_id: str
    sha: str | None
    lease_id: int | None
    reason: str | None
    transport: str


@dataclasses.dataclass(frozen=True)
class ClaimState:
    task_id: str
    sha: str
    payload: dict[str, object]
    chain_len: int

    @property
    def lease_id(self) -> int | None:
        value = self.payload.get("lease_id")
        return value if isinstance(value, int) else None

    @property
    def session_id(self) -> str | None:
        value = self.payload.get("claimed_by")
        return value if isinstance(value, str) else None

    def expired(self, *, now: dt.datetime) -> bool:
        heartbeat = self.payload.get("heartbeat_at_utc")
        ttl = self.payload.get("lease_ttl_seconds")
        if not isinstance(heartbeat, str) or not isinstance(ttl, int):
            return True  # an unreadable lease cannot prove liveness
        try:
            beat = dt.datetime.fromisoformat(heartbeat.replace("Z", "+00:00"))
        except ValueError:
            return True
        return (now - beat).total_seconds() > ttl


def _run(repo: Path, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _utc_now_iso(now: dt.datetime | None = None) -> str:
    moment = now or dt.datetime.now(tz=dt.timezone.utc)
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _remote_exists(repo: Path, remote: str) -> bool:
    return _run(repo, ["remote", "get-url", remote], check=False).returncode == 0


def _claim_ref(task_id: str) -> str:
    return f"{CLAIM_REF_PREFIX}{task_id}"


def _commit_claim(repo: Path, payload: dict[str, object], parent: str | None) -> str:
    args = ["commit-tree", _EMPTY_TREE_SHA]
    if parent:
        args.extend(["-p", parent])
    args.extend(["-m", json.dumps(payload, indent=2, sort_keys=True)])
    cp = _run(repo, args)
    return cp.stdout.strip()


def _read_claim_payload(repo: Path, sha: str) -> dict[str, object]:
    cp = _run(repo, ["log", "-1", "--pretty=%B", sha], check=False)
    if cp.returncode != 0:
        return {}
    try:
        payload = json.loads(cp.stdout)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _chain_len(repo: Path, sha: str) -> int:
    cp = _run(repo, ["rev-list", "--count", sha], check=False)
    try:
        return int(cp.stdout.strip())
    except ValueError:
        return 0


def fetch_claims(repo: Path, remote: str) -> bool:
    """Force-sync the local claim namespace from the remote (remote wins:
    every claim mutation pushes synchronously, so the remote is the truth)."""
    if not _remote_exists(repo, remote):
        return False
    _run(
        repo,
        ["fetch", "--prune", remote, f"+{CLAIM_REF_PREFIX}*:{CLAIM_REF_PREFIX}*"],
        check=False,
    )
    return True


def read_claims(repo: Path, remote: str, *, fetch: bool = True) -> dict[str, ClaimState]:
    if fetch:
        fetch_claims(repo, remote)
    cp = _run(
        repo,
        ["for-each-ref", "--format=%(refname) %(objectname)", CLAIM_REF_PREFIX],
        check=False,
    )
    claims: dict[str, ClaimState] = {}
    for line in (cp.stdout or "").splitlines():
        parts = line.strip().split(" ")
        if len(parts) != 2:
            continue
        refname, sha = parts
        task_id = refname.removeprefix(CLAIM_REF_PREFIX)
        claims[task_id] = ClaimState(
            task_id=task_id,
            sha=sha,
            payload=_read_claim_payload(repo, sha),
            chain_len=_chain_len(repo, sha),
        )
    return claims


def _push_claim(
    repo: Path,
    remote: str,
    task_id: str,
    *,
    new_sha: str | None,
    expected_sha: str | None,
) -> tuple[bool, str]:
    """Push a claim mutation. ``new_sha=None`` deletes. ``expected_sha``
    engages force-with-lease so a raced remote update is never clobbered."""
    ref = _claim_ref(task_id)
    refspec = f"{new_sha}:{ref}" if new_sha else f":{ref}"
    args = ["push", remote, refspec]
    if expected_sha is not None:
        args.append(f"--force-with-lease={ref}:{expected_sha}")
    cp = _run(repo, args, check=False)
    return cp.returncode == 0, (cp.stderr or cp.stdout or "").strip()


def claim_task(
    repo: Path,
    remote: str,
    task_id: str,
    *,
    session_id: str,
    branch: str,
    ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
    now: dt.datetime | None = None,
    journal=None,
) -> ClaimResult:
    ref = _claim_ref(task_id)
    has_remote = fetch_claims(repo, remote)
    transport = "remote" if has_remote else "local"

    if _run(repo, ["rev-parse", "--verify", "--quiet", ref], check=False).returncode == 0:
        return ClaimResult(False, task_id, None, None, "claim_exists", transport)

    stamp = _utc_now_iso(now)
    payload: dict[str, object] = {
        "schema_version": CLAIM_SCHEMA_VERSION,
        "task_id": task_id,
        # epoch nonce: two claims created in the same second would otherwise
        # hash to the SAME root commit (identical tree/message/timestamps),
        # making a reap+reclaim indistinguishable from the original chain and
        # defeating ancestry fencing at merge time.
        "epoch": uuid.uuid4().hex,
        "lease_id": 1,
        "claimed_by": session_id,
        "claimed_at_utc": stamp,
        "lease_ttl_seconds": int(ttl_seconds),
        "heartbeat_at_utc": stamp,
        "branch": branch,
        "run_id": None,
    }
    sha = _commit_claim(repo, payload, parent=None)

    # local CAS create: fails if the ref appeared since the check above
    created = _run(repo, ["update-ref", ref, sha, ""], check=False).returncode == 0
    if not created:
        return ClaimResult(False, task_id, None, None, "claim_exists", transport)

    if has_remote:
        pushed, detail = _push_claim(repo, remote, task_id, new_sha=sha, expected_sha=None)
        if not pushed:
            # lost the remote race: roll the local ref back and report loss
            _run(repo, ["update-ref", "-d", ref, sha], check=False)
            return ClaimResult(False, task_id, None, None, f"claim_lost_race:{detail[:200]}", transport)

    if journal is not None:
        journal(
            {
                "event": "claim_created",
                "task_id": task_id,
                "sha": sha,
                "lease_id": 1,
                "session_id": session_id,
                "transport": transport,
            }
        )
    return ClaimResult(True, task_id, sha, 1, None, transport)


def renew_lease(
    repo: Path,
    remote: str,
    task_id: str,
    *,
    expected_sha: str,
    session_id: str,
    now: dt.datetime | None = None,
    journal=None,
) -> ClaimResult:
    ref = _claim_ref(task_id)
    has_remote = _remote_exists(repo, remote)
    transport = "remote" if has_remote else "local"

    current = _run(repo, ["rev-parse", "--verify", "--quiet", ref], check=False)
    if current.returncode != 0 or current.stdout.strip() != expected_sha:
        return ClaimResult(False, task_id, None, None, "lease_superseded", transport)

    payload = _read_claim_payload(repo, expected_sha)
    lease_id = payload.get("lease_id")
    if not isinstance(lease_id, int):
        return ClaimResult(False, task_id, None, None, "claim_unreadable", transport)

    renewed = dict(payload)
    renewed["lease_id"] = lease_id + 1
    renewed["heartbeat_at_utc"] = _utc_now_iso(now)
    renewed["claimed_by"] = session_id
    sha = _commit_claim(repo, renewed, parent=expected_sha)

    if _run(repo, ["update-ref", ref, sha, expected_sha], check=False).returncode != 0:
        return ClaimResult(False, task_id, None, None, "lease_superseded", transport)

    if has_remote:
        pushed, detail = _push_claim(repo, remote, task_id, new_sha=sha, expected_sha=expected_sha)
        if not pushed:
            _run(repo, ["update-ref", ref, expected_sha, sha], check=False)
            return ClaimResult(False, task_id, None, None, f"lease_superseded_remote:{detail[:200]}", transport)

    if journal is not None:
        journal(
            {
                "event": "lease_renewed",
                "task_id": task_id,
                "sha": sha,
                "lease_id": lease_id + 1,
                "session_id": session_id,
                "transport": transport,
            }
        )
    return ClaimResult(True, task_id, sha, lease_id + 1, None, transport)


def release_claim(
    repo: Path,
    remote: str,
    task_id: str,
    *,
    expected_sha: str,
    reason: str,
    journal=None,
) -> ClaimResult:
    ref = _claim_ref(task_id)
    has_remote = _remote_exists(repo, remote)
    transport = "remote" if has_remote else "local"

    if has_remote:
        pushed, detail = _push_claim(repo, remote, task_id, new_sha=None, expected_sha=expected_sha)
        if not pushed:
            return ClaimResult(False, task_id, None, None, f"release_refused:{detail[:200]}", transport)

    deleted = _run(repo, ["update-ref", "-d", ref, expected_sha], check=False).returncode == 0
    if not deleted and not has_remote:
        return ClaimResult(False, task_id, None, None, "release_refused_local", transport)

    if journal is not None:
        journal(
            {
                "event": "claim_released",
                "task_id": task_id,
                "sha": expected_sha,
                "reason": reason,
                "transport": transport,
            }
        )
    return ClaimResult(True, task_id, expected_sha, None, None, transport)


@dataclasses.dataclass(frozen=True)
class ReapAction:
    task_id: str
    sha: str
    lease_id: int | None
    reason: str


def reap_expired(
    repo: Path,
    remote: str,
    *,
    now: dt.datetime | None = None,
    fetch: bool = True,
) -> list[ReapAction]:
    """Pure planner: enumerate expired claims. The supervisor applies the
    actions (release + reopen-or-relaunch) and journals them — an expired
    lease is ORPHANED work, never silently 'blocked' (plan §4.1)."""
    moment = now or dt.datetime.now(tz=dt.timezone.utc)
    actions: list[ReapAction] = []
    for task_id, state in sorted(read_claims(repo, remote, fetch=fetch).items()):
        if state.expired(now=moment):
            actions.append(
                ReapAction(
                    task_id=task_id,
                    sha=state.sha,
                    lease_id=state.lease_id,
                    reason="lease_expired",
                )
            )
    return actions

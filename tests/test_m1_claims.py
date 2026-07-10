from __future__ import annotations

import datetime as dt
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from runtime_test_utils import init_git_fixture_repo, scaffold_runtime_repo

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import swarm_claims


NOW = dt.datetime(2026, 7, 9, 12, 0, 0, tzinfo=dt.timezone.utc)


def _clone(origin: Path, dest: Path) -> Path:
    subprocess.run(
        ["git", "clone", "--quiet", str(origin), str(dest)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "-C", str(dest), "config", "user.name", "swarm-bot"], check=True)
    subprocess.run(
        ["git", "-C", str(dest), "config", "user.email", "swarm-bot@example.invalid"],
        check=True,
    )
    return dest


class ClaimProtocolTest(unittest.TestCase):
    def _fixture(self, tmp: str) -> tuple[Path, Path]:
        root = Path(tmp) / "repo"
        scaffold_runtime_repo(root)
        init_git_fixture_repo(root)
        return root, Path(f"{root}.origin.git")

    def test_claim_create_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, origin = self._fixture(tmp)
            events: list[dict[str, object]] = []

            result = swarm_claims.claim_task(
                root,
                "origin",
                "T801",
                session_id="worker-a",
                branch="T801_slug",
                ttl_seconds=3600,
                now=NOW,
                journal=events.append,
            )

            self.assertTrue(result.ok, result)
            self.assertEqual(result.lease_id, 1)
            self.assertEqual(result.transport, "remote")

            claims = swarm_claims.read_claims(root, "origin")
            self.assertIn("T801", claims)
            state = claims["T801"]
            self.assertEqual(state.payload["claimed_by"], "worker-a")
            self.assertEqual(state.payload["branch"], "T801_slug")
            self.assertEqual(state.chain_len, 1)
            self.assertFalse(state.expired(now=NOW))

            # the ref is on the remote too — the claim is cluster-visible
            remote_refs = subprocess.run(
                ["git", "-C", str(origin), "for-each-ref", "refs/swarm/claims/"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertIn("refs/swarm/claims/T801", remote_refs)
            self.assertEqual(events[0]["event"], "claim_created")

    def test_second_claim_loses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self._fixture(tmp)
            first = swarm_claims.claim_task(
                root, "origin", "T802", session_id="a", branch="T802_x", now=NOW
            )
            self.assertTrue(first.ok)
            second = swarm_claims.claim_task(
                root, "origin", "T802", session_id="b", branch="T802_x", now=NOW
            )
            self.assertFalse(second.ok)
            self.assertEqual(second.reason, "claim_exists")

    def test_cross_clone_race_exactly_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, origin = self._fixture(tmp)
            competitor = _clone(origin, Path(tmp) / "competitor")

            # Simulate a true race: the competitor claims while the first
            # repo's fetch already happened (fetch=False path exercised via
            # direct local create + push race).
            win_a = swarm_claims.claim_task(
                root, "origin", "T803", session_id="a", branch="T803_x", now=NOW
            )
            win_b = swarm_claims.claim_task(
                competitor, "origin", "T803", session_id="b", branch="T803_x", now=NOW
            )

            self.assertTrue(win_a.ok)
            self.assertFalse(win_b.ok)
            self.assertIn(win_b.reason, ("claim_exists",))

            # harder race: competitor skips the fetch (stale view) — the
            # remote push must reject and the local ref must roll back
            competitor2 = _clone(origin, Path(tmp) / "competitor2")
            swarm_claims.claim_task(
                root, "origin", "T804", session_id="a", branch="T804_x", now=NOW
            )
            # blind the competitor: drop its fetched view of the claim so
            # its local CAS succeeds and only the remote push can reject
            subprocess.run(
                ["git", "-C", str(competitor2), "update-ref", "-d", "refs/swarm/claims/T804"],
                check=False,
                capture_output=True,
            )
            with mock.patch.object(swarm_claims, "fetch_claims", return_value=True):
                blind = swarm_claims.claim_task(
                    competitor2, "origin", "T804", session_id="b", branch="T804_x", now=NOW
                )
            self.assertFalse(blind.ok)
            self.assertTrue(str(blind.reason).startswith("claim_lost_race:"), blind)
            # rollback happened: no local ref remains in the loser
            leftover = subprocess.run(
                ["git", "-C", str(competitor2), "rev-parse", "--verify", "--quiet", "refs/swarm/claims/T804"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(leftover.returncode, 0)

    def test_renew_increments_monotonic_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self._fixture(tmp)
            claim = swarm_claims.claim_task(
                root, "origin", "T805", session_id="a", branch="T805_x", now=NOW
            )
            renewed = swarm_claims.renew_lease(
                root,
                "origin",
                "T805",
                expected_sha=claim.sha,
                session_id="a",
                now=NOW + dt.timedelta(seconds=60),
            )
            self.assertTrue(renewed.ok, renewed)
            self.assertEqual(renewed.lease_id, 2)

            state = swarm_claims.read_claims(root, "origin")["T805"]
            self.assertEqual(state.chain_len, 2)
            self.assertEqual(state.lease_id, 2)

            # renewal against a superseded sha is fenced off
            stale = swarm_claims.renew_lease(
                root, "origin", "T805", expected_sha=claim.sha, session_id="a", now=NOW
            )
            self.assertFalse(stale.ok)
            self.assertEqual(stale.reason, "lease_superseded")

    def test_release_requires_current_tip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, origin = self._fixture(tmp)
            claim = swarm_claims.claim_task(
                root, "origin", "T806", session_id="a", branch="T806_x", now=NOW
            )
            renewed = swarm_claims.renew_lease(
                root, "origin", "T806", expected_sha=claim.sha, session_id="a", now=NOW
            )

            refused = swarm_claims.release_claim(
                root, "origin", "T806", expected_sha=claim.sha, reason="done"
            )
            self.assertFalse(refused.ok)

            released = swarm_claims.release_claim(
                root, "origin", "T806", expected_sha=renewed.sha, reason="done"
            )
            self.assertTrue(released.ok, released)
            self.assertNotIn("T806", swarm_claims.read_claims(root, "origin"))
            remote_refs = subprocess.run(
                ["git", "-C", str(origin), "for-each-ref", "refs/swarm/claims/"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertNotIn("T806", remote_refs)

    def test_expiry_and_reap_planning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self._fixture(tmp)
            swarm_claims.claim_task(
                root, "origin", "T807", session_id="a", branch="T807_x", ttl_seconds=60, now=NOW
            )
            swarm_claims.claim_task(
                root, "origin", "T808", session_id="a", branch="T808_x", ttl_seconds=3600, now=NOW
            )

            later = NOW + dt.timedelta(seconds=120)
            actions = swarm_claims.reap_expired(root, "origin", now=later)
            self.assertEqual([a.task_id for a in actions], ["T807"])
            self.assertEqual(actions[0].reason, "lease_expired")

    def test_local_only_transport(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "solo"
            scaffold_runtime_repo(root)
            subprocess.run(["git", "-C", str(root), "init", "-q", "-b", "main"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "swarm-bot"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "swarm-bot@example.invalid"],
                check=True,
            )
            subprocess.run(["git", "-C", str(root), "add", "-A"], check=True, capture_output=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-q", "-m", "init"],
                check=True,
                capture_output=True,
            )

            result = swarm_claims.claim_task(
                root, "origin", "T809", session_id="a", branch="T809_x", now=NOW
            )
            self.assertTrue(result.ok, result)
            self.assertEqual(result.transport, "local")
            self.assertIn("T809", swarm_claims.read_claims(root, "origin", fetch=False))


if __name__ == "__main__":
    unittest.main()

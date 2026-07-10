# First-Class Swarm — Build Execution Log

Frozen spec: `docs/first_class_swarm_production_plan_2026-07-09.md` (v1.3 FINAL).
This log is the durable record of the M0–M5 + BT2-A build: per-milestone branch, batch decomposition, design decisions taken during implementation, test/review outcomes, and any deviation from the plan (deviations reopen the plan per §10).

Baseline (2026-07-09, commit `f1d1638`): 35/35 tests green, all 14 quality gates green, plan committed to `main`.

Milestone sequencing: M0 → M1 → M2 → {M3a ∥ M3b ∥ M3c} → M4 → M5 → BT2 Stage A.
Delivery: one branch per milestone (`milestone/<id>-<slug>`), red/green-tested before merge to `main`; high-stakes surfaces (gate sandboxing, git-destructive-op guards, historical-record remediation, security preflight) implemented and reviewed inline by the owner model; bounded implementation batches delegated to the Codex workhorse with full diff review.

---

## M0 — Stop the bleeding (branch: `milestone/m0-stop-the-bleeding`) — IN PROGRESS

Scope: plan §4.0 items 1–18, historical-record remediation, golden-suite harness seeded with 10 tasks.

Batch decomposition:

| Batch | Plan items | Surface | Lane |
|---|---|---|---|
| A | 1, 2, 4, 5 | claim regex + loop resilience + task quarantine + shared task-file parser | delegate |
| B | 3, 6, 7, 13, 14 | run-task deps enforcement, provenance_class, durable logs, frontmatter immutability, ownership-scoped commits | delegate |
| C | 9, 10, 17 | guarded base sync, judge merge-base/head-SHA recheck, actor separation | inline (git-destructive + judge-integrity surface) |
| D | 8, 11, 15, 16 | hash recompute gates, transform provenance, content-bound validation, projection-drift gate | delegate |
| E | 12, 18 | pinned constrained gate execution (shell=False, allowlist, env-strip, network-off-where-supported) + rule-level gate output capture | inline (sandbox/security surface — tripwire) |
| F | — | *(dissolved: #8 folded into Batch D, #18 folded into Batch E — shared surfaces)* | — |
| G | remediation | schema v2 bumps, provenance_class backfill annotations, release amendment, gate-scoping rule | inline (historical-record honesty surface) |
| H | golden harness | `tests/golden/` harness + 10 seeded tasks + M0 acceptance tests | delegate |

Design decisions (recorded as taken):

- **D-M0-1 (schema versions).** `SWARM_RUN_MANIFEST_SCHEMA_VERSION` → `research_swarm.runtime_run_manifest.v2`; `JUDGE_REVIEW_LOG_SCHEMA_VERSION` → `research_swarm.judge_review_log.v2`. Strict checks (provenance_class, actor separation, log hash) apply to v2+ artifacts only; the 32 historical run manifests and 23 review logs go on a checked-in, gate-verified exemption list (`contracts/historical_exemptions.json`, per-file sha256 + reason) per the plan's gate-scoping rule.
- **D-M0-2 (gate pinning semantics in M0).** The plan pins gate commands "at claim time"; real claim primitives (refs) arrive in M1. M0 operationalization: run-task captures the task frontmatter **before the executor is invoked** (the pre-run snapshot), records the pinned gates + protected frontmatter hash in the run manifest, executes **only** the pinned copy, and blocks on any executor change to `gates`/`allowed_paths`/`outputs`/`dependencies` (#13). This closes the executor-weakens-gates channel; M1 re-anchors the pin to the claim ref. Recorded as an intent-preserving refinement, not a deviation.
- **D-M0-3 (constrained gate execution, #12).** `shlex.split` (no shell), interpreter allowlist on argv[0] basename (`python`, `python3`, `make`; contract-configurable), per-gate timeout (default 600s, contract-configurable), env allowlist (`PATH`, `HOME`, `LANG`, `LC_ALL`, `TMPDIR`, `TERM` + `GIT_TERMINAL_PROMPT=0`) — credentials/proxy/cloud vars are excluded by construction. Network disabling: `sandbox-exec` deny-network profile on darwin, `unshare -n` where unprivileged user namespaces allow it on Linux; capability probed at runtime and the **effective** network state recorded per gate in the run manifest — never claimed when not enforced. Full executor-parity sandboxing deferred per plan (§4.0 #12).
- **D-M0-4 (actor separation, #17).** Per-process actor session id (uuid4, overridable via `SWARM_ACTOR_SESSION` for tests); run manifests and review logs record `actor: {session_id, recorded_at}`. A v2 review log is invalid if its session id equals the run manifest's, or if the wall-clock separation is below `review.min_separation_seconds` (contract key, default 60).
- **D-M0-5 (guarded base sync, #9).** `_supervisor_sync_to_remote_base` refuses (with an escalation-shaped error) when the local base branch has commits not on the remote (`git rev-list remote/base..base`), instead of `checkout -B`. Control-plane commits/pushes become strict in all modes (failure raises; no `[warn]`-and-continue).

Batch record:

- **Batch C** (items 9, 10, 17) — LANDED (owner-implemented inline; tripwire surface). Guarded base sync: `checkout -B` replaced with fetch → refuse-if-local-ahead (`base_sync_refused_local_ahead`) → ff-merge/branch-update; judge re-runs the run-task ownership/diff check against the merge-base and verifies tip integrity against the selected manifest (branch match, sha ancestry, ≤1 post-manifest commit with the runtime's message and declared paths only); actor separation: per-process session ids in run manifests (`actor`) and v2 review logs (`reviewer.session_id`), same-session and sub-window reviews invalid (contract key `review_bundle.min_separation_seconds`, default 60); judge commits only its own control-plane artifacts (no more `git add -A` sweeping up leftover violations); review-log schema → v2 with distinct `operator_attestation` field; durable executor logs un-gitignored. Known wart logged: frontmatter list parser strips a trailing quote from values — M2 task-lint must reject gate commands ending in a quote. Verified: 66/66 tests, 14/14 gates, sweep clean.
- **Batch A** (items 1, 2, 4, 5) — LANDED. New `scripts/swarm_taskfile.py` shared parser (branch-id regex `^(T\d{3})(?=[_-]|$)`, `## Status`-scoped state parsing, in-section note appends, `WorktreeCollisionError`); quarantine (`load_tasks_quarantined`) wired into plan/tick; loop survives `BaseException` with capped exponential backoff and no unattended death; all three scripts consume the shared parser. Implemented by Codex workhorse (gpt-5.6-sol xhigh), full diff reviewed by owner. Verified independently: 43/43 tests, 14/14 gates, sweep clean.

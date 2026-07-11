# Swarm operator runbook

This is the M5 operator surface for supervising the local swarm. Use
`docs/runbook_swarm.md` for task-level commands and
`docs/runbook_swarm_automation.md` for the automation interface. The Operator
owns environment preflight, supervision, sweep hygiene, repair handling, run
and release logs, catalog refresh, and release assembly. The Operator does not
change scientific contracts or mark scientific work `done`.

## Supervise lifecycle

1. Preflight from a clean, sandboxed checkout. Confirm the containment marker,
   credential scan, vendor-policy acknowledgement, pack compatibility, active
   preregistration locks, task lint, and offline gate policy. Stop on a failed
   preflight; never downgrade a gate or claim a sandbox capability that the
   host did not enforce.
2. Inspect `python3.11 scripts/swarm.py status --json`, the append-only event
   journal, claims, worktrees, and review queue. Reconcile before dispatch if
   journal, git, lease, or state projections disagree.
3. Start `python3.11 scripts/swarm.py supervise` with the reviewed worker cap,
   budget, executor backend, and base branch. Keep one task per branch and
   worktree. Treat `ready_for_review` backpressure as a dispatch stop.
4. During supervision, watch heartbeats, budgets, gate results, journal write
   failures, orphaned processes, main-worktree mutations, and `@human` holds.
   Use bounded REPAIR dispatch only for a diagnosed, in-scope operational
   failure. Scientific conflicts go to Planner plus the human owner.
5. On shutdown, stop new dispatch, let active commands terminate, reconcile,
   reap only verified orphans, and record the run outcome. Preserve failed
   evidence. Do not erase a red run or rewrite an append-only snapshot.
6. Before review or release assembly, require the task, executor-backed run
   manifest, declared outputs, gates, processed manifests, referee evidence,
   integrity audit, and durable swarm-run manifest. Only the Judge may approve
   scientific completion; only a named human may perform L3 attestation.

## Escalation playbooks

Escalations form two tiers. A plain `escalation=True` event is an *operational*
notable event (auto-recovery, refusal, retry) that needs no human playbook. An
event that additionally carries an `escalation_class` is a *human* escalation on
§5.4's standing channel — every such class MUST be one of the registered classes
below, which `runbook_staleness` then forces to carry a playbook. A new
human-escalation class therefore cannot ship without being registered in
`swarm_events.ESCALATION_CLASSES` and documented here: `escalate()` rejects an
unregistered class at runtime, and the golden test
`test_no_emitter_uses_an_unregistered_escalation_class` fails `make test` if any
runtime emitter tags an unregistered class (real CI teeth, independent of the
best-effort journal path).

Of the five classes, three are **auto-emitted** by the runtime with their class
tag — `blocked_with_human` (`plan_awaiting_human_approval` / `human_question` /
`referee_owner_waiver`), `hypothesis_task_retirement`
(`hypothesis_retirement_escalated`), and `budget_breach` (`budget_exceeded`).
The remaining two — `judge_disagreement` and `unsatisfiable_constraints` — are
**owner-raised** channels: the v1 runtime does not auto-detect conflicting
reviews or an infeasible constraint set and emit the class (that automation is
scheduled M1+ supervisor work); the operator invokes these playbooks when they
observe the condition. Both remain registered + documented so a future automated
emitter can adopt them without re-opening this contract.

### Judge disagreement

- escalation_class: judge_disagreement

Freeze promotion, preserve each unaggregated review and artifact version, and
route the disagreement to the human owner. Do not average contradictory
scientific verdicts. The owner reads the raw findings and records a disposition
or requests a bounded revision task.

### Budget breach

- escalation_class: budget_breach

Stop new dispatch and the over-budget executor, preserve usage records, and
report actual tokens, wall time, metered usage, and unfinished outputs. The
owner may authorize a new bounded budget or terminate the program; the Operator
may not silently raise a ceiling.

### Blocked with human

- escalation_class: blocked_with_human

Verify that the task contains the smallest concrete `@human` question and that
no safe in-scope check remains. Preserve its claim and worktree while the hold
is active. Apply the recorded answer through normal Planner/task-contract flow.

### Unsatisfiable constraints

- escalation_class: unsatisfiable_constraints

Stop the affected path, capture the conflicting sources in precedence order,
and ask the owner for the exact contract decision. Never choose a scientific,
security, legal, or release interpretation on the owner's behalf.

### Hypothesis-task retirement

- escalation_class: hypothesis_task_retirement

Block retirement and require L3 human review. Preserve the registered
hypothesis, all runs, and the reason work cannot continue. A null or difficult
result must remain a terminal reported outcome, not be relabeled as an
operationally failed task.

## L3 human attestation

The four release-blocking attestations are: research-program plan approval;
each preregistration lock and amendment; integrity-audit sign-off; and final
release/submission attestation by the named owner. Record who attested, when,
the exact artifact/git identifiers, disposition, defects found, and time spent.

The owner must bypass the swarm-authored bundle for all three checks:

1. Open the kernel-sampled registered claims directly in manuscript and primary
   artifacts, tracing each byte/hash without relying on the bundle's links.
2. On the owner's machine, independently recompute one headline number from
   the replication package and compare it with the manuscript and ledger.
3. Read every raw referee disagreement in full, not only the aggregate verdict.

A mismatch blocks attestation and becomes a journaled escalation. Sustained
zero defects across human gates triggers a process review rather than a claim
that fabrication is absent.

## Seeded-defect drills and evaluation tiers

Run the deterministic rotation with:

```sh
python3.11 scripts/seeded_drill.py --all
```

Each drill creates a disposable rehearsal fixture using the mock executor,
injects exactly one named defect, invokes the real Judge or science gate, and
appends a `seeded_drill` event to the ephemeral drill ledger
(`reports/status/drills/drill_events.jsonl`, gitignored) — **never** the
compliance journal the disclosure reads, so a rehearsal can never pollute a
released artifact's provenance and `make drill` leaves the tree clean. Any
injected-but-not-caught case exits red. The summary event reports
`caught / injected`; zero detected defects is never interpreted as zero
fabrication.

Run the held-out Goodhart-control tier only at milestone gates:

```sh
make eval-heldout
```

It is deliberately excluded from `make test` and must be refreshed
adversarially at every milestone with fresh cases aimed at current mechanisms.
Prompt/contract PRs run only the visible `tests/golden/` regression tier in CI.
Tier-c live-LLM calibration remains on-demand/BT2 and is not a per-PR job.

## BT2 Stage A rehearsal and the pass/fail bar

Re-run the STR pack as a frozen-reference regression plus seeded drills with:

```sh
make bt2a-rehearsal
```

It asserts the KNOWN ANSWERS reproduce (`reproduce-analysis` byte/content identity
+ computed-paper key resolution), executes the seeded-defect rotation (empirical
fabrication classes + a proof defect + a bridge-layer tampered-instance defect —
every case must be CAUGHT), exercises the release perimeter (the all-failing STR
registry correctly BLOCKS a release — recorded as the EXPECTED known outcome), and
writes the deterministic `reports/status/bt2a/rehearsal.json`. A drift in any
frozen-reference value, a missed injection, or a release that fails to block is a
RED regression. The report is committed evidence; the drill KPI events go to the
ephemeral drill journal, never the compliance journal.

The pre-committed BT2 pass/fail bar lives in `contracts/bt2_bar.json` (§10):
`seeded_defect_catch_rate_min`, `human_review_hours_per_artifact_max`,
`token_dollar_ceiling_usd`, `unresolved_fabrication_findings_max`,
`registered_vs_reported_hypothesis_ratio`, plus the Stage B abort clause (an
aborted Stage B is a measured outcome — event-journal closure + a written
negative-result report through the manuscript machinery, not a silent stall). The
`bt2_bar` gate validates the bar is present + well-formed (domain-checked: catch
threshold in (0,1], hypothesis-ratio invariant = 1, zero fabrication ceiling,
positive review-hours/$ ceiling, four unique abort triggers, per-number
provenance) and holds the rehearsal report to it — reconciling the catch rate
against the drill list (not the reported number), requiring the release block to
be attributable to the failing registry, and requiring the frozen reference to
have reproduced. The gate validates the COMMITTED reference; report FRESHNESS is
CI-enforced — CI runs `make bt2a-rehearsal` (regenerating the report) BEFORE
`make gate` and asserts the report is byte-stable, so a stale report cannot pass
in CI. The `human_review_hours` and `$` numbers are committed PROVISIONALLY
(see `contracts/bt2_bar.json` provenance), to be confirmed by Stage B's
spend-ledger actuals under its own project contract.

## Effective scratch confinement

The integrity audit always uses a disposable detached worktree (or a hermetic
temporary copy), an allowlisted command surface, and detection/rejection of
main-repo mutation. On macOS this does **not** provide an OS network namespace or
a kernel-enforced filesystem boundary, so `os_enforced` is always `false` — an
honest residual, not OS confinement.

The confinement report describes what is **actually applied to the launched
subprocesses**, per backend — it never claims a scrub or proxy it does not
perform:

- **mock backend** (CI default): the auditor transcript is read from a local
  file (no auditor egress); EVERY audit subprocess — rebuilds/recomputes AND the
  git worktree/inspection helpers — is launched from a benign-only environment
  ALLOWLIST (credentials are dropped by construction, not by a leaky name
  denylist) with every proxy pointed at a dead local port. Reported as
  `credential_isolation: environment_scrub_only` and
  `effective_network: proxy_environment_only` — both true.
- **live backend**: the auditor subprocess must reach the vendor API, so it
  retains the vendor credential (re-added on top of the same benign allowlist —
  every *other* credential stays dropped) and, because no namespace/proxy/
  destination control is enforced at the process level, UNRESTRICTED process
  egress. Reported honestly as `credential_isolation: vendor_credential_retained`
  and `effective_network: unrestricted_process_egress` — never a full-scrub or
  proxy-only claim the live call cannot honour. (The live `claude` auditor is
  additionally restricted to `Read/Glob/Grep` with an empty MCP config, a
  separate tool-level control, but the network label describes process egress.)

The `integrity_audit` gate binds these labels to the report's backend and
requires `os_enforced: false`, so a hash-bound but dishonest report (a live
backend claiming proxy-only/full-scrub, or any report claiming OS enforcement)
fails the gate.

The executor config requests `network: off`; because no OS namespace enforces it
here, the report records the honest `network: requested_off` (request made, not
kernel-enforced), not a bare `off` that would overstate enforcement. A future
Linux deployment may report `os_enforced: true` / `namespace_enforced_off` only
after its namespace/credential-isolation capability probe succeeds and every
audit subprocess is actually launched inside it.

## Gate registry coverage

The `runbook_staleness` gate parses the following machine-readable entries.
Adding a kernel gate without adding its operator entry fails CI.

- gate: pack_compat
- gate: scaffold_safety
- gate: framework_contract
- gate: repo_structure
- gate: project_contract
- gate: protocol_complete
- gate: workstreams_complete
- gate: task_hygiene
- gate: task_dependencies
- gate: integration_ready_policy
- gate: operator_surface_ownership
- gate: raw_manifest_validity
- gate: processed_manifest_validity
- gate: swarm_run_manifest_validity
- gate: judge_review_log_validity
- gate: referee_rubrics
- gate: referee_report_validity
- gate: referee_calibration
- gate: review_bundle_integrity
- gate: processed_manifest_hashes
- gate: raw_manifest_hashes
- gate: validation_report_content_binding
- gate: projection_drift
- gate: historical_exemptions
- gate: network_strings
- gate: task_lint
- gate: runbook_staleness
- gate: prereg_lock_coverage
- gate: raw_retention
- gate: program_conformance
- gate: replication_package_audit
- gate: venue_compliance
- gate: exhibits_manifest
- gate: paper_registry
- gate: prereg_conformance
- gate: claim_evidence_ledger
- gate: manuscript_computed_paper
- gate: citation_integrity
- gate: literature_corpus
- gate: recall_audit
- gate: integrity_audit
- gate: prompt_surface
- gate: etl_decision_log
- gate: statistical_reporting
- gate: rigor_sections
- gate: instance_manifest_conformance
- gate: seed_budget_lock
- gate: gap_convergence
- gate: theoretical_falsification
- gate: sweep_artifact
- gate: hybrid_interface_conformance
- gate: amendment_exploratory_tagging
- gate: headline_confirmatory
- gate: render_qa
- gate: text_overlap
- gate: checklist_derivation
- gate: bt2_bar

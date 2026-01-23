<review>

### Executive Summary

This template is *very close* to a usable, research-grade multi-agent workflow. The strongest parts are: (i) explicit role contracts (`AGENTS.md` + nested `AGENTS.md`), (ii) protocol-first / contracts-first structure that directly targets “metric shopping,” (iii) immutable raw snapshot + tracked manifest discipline, and (iv) deterministic quality gates that make the repo hard to silently rot.

The biggest problem is a **role boundary contradiction**: the repo’s governance docs say “only Planner moves tasks across lifecycle folders,” but `scripts/swarm.py` *does the moving itself* (and also claims tasks). That’s not a minor nit — it will cause confusion and eventually “who is allowed to do what?” drift. The second biggest problem is **parallel task collision risk**: the scheduler can start multiple tasks whose `allowed_paths` overlap (especially `src/analysis/`), which is the exact kind of friction that makes swarms feel chaotic even with worktrees.

If you fix those two things, plus tighten contracts around **units/edge cases** and add one or two missing “research integrity” gates (dependency validity, contract-change discipline), you’ll have a robust template that can run your STR vertical slice cleanly and scale to 4–8 concurrent workers without devolving into merge-conflict babysitting.

---

### Strengths

* **Protocol lock is correctly treated as a first-class contract.**
  `docs/protocol.md` is canonical, and tasks are instructed to block on definition ambiguity. This is the right defense against “definition drift” in empirical pipelines.

* **Clear workstream ownership boundaries (mostly).**
  `.orchestrator/workstreams.md` and per-directory `AGENTS.md` create a practical “ownership map” that reduces accidental clobbering.

* **Task specs are structured and enforceable.**
  YAML frontmatter (dependencies, allowed/disallowed paths, outputs, gates, stop conditions) is a strong pattern for preventing workers from silently broadening scope.

* **Deterministic quality gates exist and are fast.**
  `scripts/quality_gates.py` checks repo invariants, protocol completeness, workstreams completeness, task hygiene, raw manifest validity. This is exactly how you avoid “dashboard science” infrastructure drift.

* **Reproducibility discipline is built-in.**
  Append-only raw snapshots + tracked provenance manifests are the right core; it makes downstream debates about data much more concrete.

* **You’re aiming for a “vertical slice” early.**
  The planned flow T020→T030→T040→T050→T060 is the correct way to test the orchestration machinery *before* expanding to full historical pipelines and on-chain decomposition.

---

### Critical Issues

#### Severity: Blocker (will cause real operational confusion)

1. **Role boundary contradiction: “Only Planner moves tasks” vs `swarm.py` moving tasks.**

   * Docs (`AGENTS.md`, `.orchestrator/AGENTS.md`) say Planner-only lifecycle moves.
   * `scripts/swarm.py cmd_run_task()` does `git mv` between lifecycle folders and also moves backlog→active on claim.
     This will eventually cause human/agent behavior mismatches (“am I allowed to move tasks or not?”). Decide one rule and enforce it.

2. **Parallel collision risk: scheduler doesn’t prevent overlapping task scopes.**
   `choose_tasks_*` selects tasks purely by readiness/priority. It does **not** check whether concurrently-started tasks have overlapping `allowed_paths` / outputs.
   Result: two workers can legally edit `src/analysis/` simultaneously (even with disallowed file exceptions). Worktrees reduce local clobbering, but merge conflicts and semantic collisions still spike.

#### Severity: High (will degrade research integrity or reproducibility)

3. **`make_raw_manifest.py` naming can desync from snapshot run-date.**
   The README recommends `data/raw_manifest/<source>_<YYYY-MM-DD>.json` keyed to snapshot date.
   Your script writes manifest filename using “today” (`now.date()`), not the snapshot folder date or the `--run-date`. If you backfill historical data, manifests won’t line up.

4. **Protocol tolerances are partly ambiguous (“0.5–1.0% unit-dependent”).**
   Agents will interpret differently. If tolerances are allowed to be “chosen later,” you need a rule for selecting them (or require a W0 decision entry before any validation task relies on them).

5. **Contracts don’t yet cover the decomposition objectives.**
   The repo is currently a STR-only vertical slice. That’s fine — but the template (as “production-ready reusable workflow”) should include contract placeholders for:

   * burn vs tips decomposition,
   * blob vs calldata components,
   * regime classification,
   * counterfactual mechanism inputs (EIP‑7918).
     Right now there’s no canonical “decomposition panel schema” contract, so agents will invent fields ad hoc.

#### Severity: Medium (will cause friction / inefficiency)

6. **Dependency hygiene gate is incomplete.**
   `gate_task_hygiene()` verifies `dependencies` is a list, but doesn’t validate:

   * dependency IDs exist,
   * no cycles,
   * no duplicate task IDs across folders.
     This is the kind of small omission that silently breaks automation later.

7. **Workstreams template has at least one misleading example.**
   `.orchestrator/templates/workstreams_template.md` says W3 owns `data/schemas/` (wrong for your actual repo). That’s a “template footgun.”

---

### Gaps and Missing Elements

* **No explicit “contract-change discipline” gate.**
  You have `contracts/CHANGELOG.md` and `contracts/decisions.md`, but no gate that enforces “if contracts change, decision log and changelog must be updated.” This is *the* common failure mode: agents edit schema/protocol without recording rationale.

* **No “processed artifact manifest” convention.**
  Raw provenance is great, but once you create `data/processed/*`, you need at least lightweight manifesting for:

  * input snapshot(s),
  * transform code version (git sha),
  * output hash,
  * command used.
    Without this, “why did results change?” becomes painful.

* **No registry stub for rollup IDs / evidence structure.**
  You have `registry/AGENTS.md` and `registry/CHANGELOG.md` but no versioned registry file in place. That’s going to become a hotspot later.

* **Automation doesn’t implement “fresh start” policies.**
  Docs mention restarting long-running sessions; `swarm.py` doesn’t encode a max-turn / max-runtime / restart rule. For unattended runs, drift happens.

* **No top-level “Quickstart: run the vertical slice.”**
  There’s a runbook, but it’s orchestration-oriented. You need a minimal “do this to generate the first STR figure from sample” path once the sample exists.

---

### Recommended Improvements

#### Architecture & Coordination

1. **Resolve the Planner vs automation contradiction (pick one, then enforce).**
   You have two clean options:

   **Option A (clean separation; recommended):**

   * `swarm.py run-task` **only updates `State:`** in task file, never moves files between lifecycle folders.
   * Add `scripts/sweep_tasks.py` (Planner tool) that moves task files based on state.
   * Update docs: “Automation is Worker+Judge; Planner sweep is separate.”

   **Option B (explicitly grant automation Planner powers):**

   * Update `.orchestrator/AGENTS.md` to say:
     “Planner-only moves tasks — except `scripts/swarm.py` when run as supervisor, which is authorized to move tasks automatically.”
   * This is workable, but you’re weakening the conceptual clarity of roles.

2. **Add concurrency control to task scheduling (avoid overlapping scopes).**
   Minimum viable rule that works in practice:

   * start **at most one task per workstream** concurrently, unless explicitly marked `parallel_ok: true`.
     This is crude but effective and avoids needing fine-grained path locks (which tend to become the lock anti-pattern).
     If you want better control, add a frontmatter field like:
   * `mutex: ["W4", "src/analysis/"]`
     and ensure the scheduler respects it.

3. **Enforce least-privilege `allowed_paths`.**
   Don’t allow broad prefixes when outputs are specific files.
   Example: for T040, allow `src/analysis/metrics_str.py` and `tests/test_metrics_str.py`, not `src/analysis/` and `tests/`.
   This reduces collision probability *dramatically*.

4. **Strengthen dependency validity.**
   Add a quality gate that ensures:

   * every dependency ID exists in some lifecycle folder,
   * no cycles (simple DFS),
   * no duplicate `task_id` frontmatter across task files.

#### File Structure & Artifacts

5. **Create a `registry/rollup_registry_v1.csv` stub now (even before you fill it).**
   Make the interface explicit before anyone writes ETL that assumes IDs. Include required columns like:

   * `rollup_id` (stable slug)
   * `display_name`
   * `type` (optimistic/zk)
   * `da_posting_method` (calldata/blobs/both)
   * `batcher_addresses` (maybe JSON string)
   * `evidence_url`
   * `verified_utc`
   * `status` (active/deprecated)

6. **Split empirical schema contracts into “minimum STR panel” vs “full decomposition panel.”**

   * `panel_schema_str_v1.yaml`
   * `panel_schema_decomp_v1.yaml`
     This makes your vertical slice clean while keeping the long-run objectives contract-driven.

7. **Add processed-output manifest convention.**
   Create `data/processed_manifest/` parallel to raw manifests (tracked, small).
   Each processed dataset gets a JSON with:

   * input raw manifest reference(s)
   * transform script path + git sha
   * output file list + hashes
   * command line

#### Quality Gates & Testing

8. **Add a gate: “contract changes require decision log entry.”**
   Deterministic implementation idea:
   If `git diff --name-only origin/main...` includes anything under `contracts/` or `docs/protocol.md`, then require that `contracts/decisions.md` has a new entry dated today or contains the task ID.
   (Not perfect, but it catches 80% of silent drift.)

9. **Add sample-based “data integrity” tests early.**
   Once T030 creates `data/samples/...`, gates should validate:

   * schema matches contract,
   * date parsing and uniqueness,
   * no negative fees/rent,
   * `profit ≈ fees − rent` within chosen tolerance.

10. **Make CI meaningful beyond “smoke test.”**
    Right now `make test` is a placebo. Once T040/T050 exist, CI starts paying rent.

#### Tooling & Infrastructure

11. **Fix `make_raw_manifest.py` output naming to accept a date.**
    Add `--as-of YYYY-MM-DD` or infer from snapshot_dir name.
    This matters the moment you backfill or re-run.

12. **Add safety interlocks for unattended mode.**
    Example: require `SWARM_UNATTENDED_I_UNDERSTAND=1` env var before allowing `--unattended`.
    This prevents accidental “bypass permissions” runs on your laptop.

13. **Implement “fresh start” drift control in automation.**
    You don’t need fancy memory. Just:

* limit worker run to N turns or M minutes, and if incomplete, leave task `active` with logs.
* re-run later.
  This prevents runaway sessions.

#### Documentation & Instructions

14. **Add a “Vertical Slice Quickstart” section.**
    Once the sample exists, you want one page that says:

* run ETL snapshot (T030)
* run metrics tests (T040)
* run validation (T050)
* generate figure (T060)
* expected output paths

15. **Clarify protocol edge cases explicitly.**
    Put the following in `docs/protocol.md` as explicit rules:

* If `Σ L2Fees_t == 0`: STR_t is `NaN` (or 0) — pick one.
* How to treat missing `rent_paid` vs missing `fees` days (drop? impute? zero?).
* Whether rollup universe is time-varying (enter/exit).

#### Research-Specific Elements

16. **Add a decomposition contract now, even as a stub.**
    Objective 2 requires fields like:

* `l1_base_fee_burn_eth`
* `l1_blob_fee_burn_eth`
* `l1_priority_fee_eth`
* `l1_total_rent_eth = burn + tips`
  If you don’t lock names/units now, you’ll pay later in reconciliation pain.

17. **Add a “regime definition” contract section.**
    Objective 3 (“blob minimum price vs congestion spikes”) needs a measurable rule:

* e.g., “blob_base_fee <= 1.05 × min for ≥X days”
  Put the rule in protocol/contracts, not in analysis scripts.

---

### Simplification Opportunities

* **Remove or quarantine modeling artifacts for this empirical-only project run.**
  You can keep `contracts/model_spec.md` etc, but move modeling/hybrid scaffolding into `docs/optional/` or a “mode=hybrid” branch. The current template is fine, but it adds cognitive surface area for the first spinoff.

* **Avoid manual SVG generation for figures.**
  Writing raw SVG by hand is a false economy. Add a minimal plotting dependency when you reach T060 (matplotlib), or generate CSV + let your paper tooling plot. The “no deps” bootstrap goal is fine, but don’t contort analysis.

* **Deduplicate frontmatter parsers.**
  `scripts/swarm.py` and `scripts/quality_gates.py` have similar YAML frontmatter parsing. That will drift. Move to a shared stdlib helper module (still no PyYAML).

---

### Implementation Priorities

1. **Fix governance/role contradictions (Planner vs automation).**
   Decide Option A or B and implement it. This is priority #1 because it prevents “process bugs.”

2. **Add concurrency collision control.**
   At minimum: one task per workstream at a time, plus narrower `allowed_paths`.

3. **Lock empirical contracts needed for the vertical slice (T020).**
   Panel schema + data dictionary + rollup_id convention + denominator-zero rule.

4. **Fix raw manifest naming so backfills are reproducible.**

5. **Add dependency validity gate and contract-change discipline gate.**

6. **Only then scale tasks/workers.**
   If you scale before (1)–(3), you’ll experience “swarm chaos” even with good intentions.

---

### Specific Template Additions

#### 1) `scripts/sweep_tasks.py` (Planner sweep tool)

Purpose: enforce “Planner moves tasks; others only set State.”

Minimal behavior:

* read all task files in lifecycle dirs
* parse `State:` line
* if state doesn’t match folder, move it (git mv)

**Snippet (conceptual):**

```py
# scripts/sweep_tasks.py
# - only moves tasks based on State line
# - no network calls, deterministic
```

Then add:

* `make sweep` target

#### 2) Add dependency validity gate

Extend `gate_task_hygiene()` or add `gate_task_dependencies()` to check:

* all `dependencies` refer to existing task IDs
* no cycles
* unique task_ids

#### 3) Add `mutex` or “one per workstream” scheduling rule

In `swarm.py choose_tasks_*`, after selecting candidates:

* filter out tasks whose `workstream` already selected

This single change eliminates a large class of merge conflicts.

#### 4) Fix manifest naming

Update `scripts/make_raw_manifest.py` to accept `--as-of`:

```bash
python scripts/make_raw_manifest.py growthepie data/raw/growthepie/2024-01-01 \
  --as-of 2024-01-01 \
  "python src/etl/growthepie_fetch.py --run-date 2024-01-01"
```

#### 5) Add `registry/rollup_registry_v1.csv` header now

Example:

```csv
rollup_id,display_name,type,da_posting_method,batcher_addresses_json,evidence_url,verified_utc,status,notes
```

#### 6) Add STR panel schema fields (in contracts)

Your current `panel_schema.yaml` is a stub with `unit_id`. For STR it should become rollup-specific:

```yaml
version: 1
table: daily_rollup_panel
grain: [date_utc, rollup_id]
fields:
  - name: date_utc
    type: date
  - name: rollup_id
    type: string
  - name: l2_fees_eth
    type: number
  - name: rent_paid_eth
    type: number
  - name: profit_eth
    type: number
    nullable: true
  - name: txcount
    type: integer
    nullable: true
```

Then explicitly state units: ETH (not wei), or wei (integer). Pick one and enforce.

---

### Final Recommendations

1. **Make roles real, not aspirational.** Align docs and automation: either automation is authorized Planner or it isn’t. Right now it’s neither, and that will break coordination.

2. **Prevent collisions structurally.** Add simple concurrency rules (one task/workstream) and narrow `allowed_paths`. This is the difference between “parallel” and “parallel but miserable.”

3. **Lock contracts for units and edge cases before ETL.** STR is a ratio; denominator-zero and missing-data rules must be contract-level, not analyst-level.

4. **Treat provenance as end-to-end, not just raw.** Raw manifests are great; add processed/figure manifests so results are auditable.

5. **Keep the vertical slice minimal, but add stubs for what’s coming.** Add registry and decomposition schema stubs now so later work doesn’t invent interfaces ad hoc.

</review>

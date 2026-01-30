# Autonomous Agentic Research Swarm (Repo Template)

A lightweight, **file-based** workflow for running multiple AI coding/research agents in parallel (e.g., Codex CLI, Claude Code) while keeping coordination sane, reproducible, and reviewable.

This repo is intentionally **not** an “agents talk to each other” framework. The repo (files + git history) is the shared memory.

## The Mental Model

This template enforces three roles:

- **Planner**: creates/scopes tasks and owns the control plane (`.orchestrator/`).
- **Worker**: executes exactly one task in an isolated branch/worktree; writes only within task `allowed_paths`.
- **Judge**: runs deterministic gates/tests and approves/blocks merges based on success criteria.

Why this works:
- A shared mutable “kanban brain” (single file everyone edits) breaks under parallelism.
- Git worktrees/branches isolate work; task files + contracts prevent definition drift; gates prevent low-quality merges.

## What You Get (Core Primitives)

1. **Protocol / contract lock (definitions-first)**
   - Empirical/hybrid: `docs/protocol.md` + versioned schemas under `contracts/schemas/`
   - Modeling/hybrid: `contracts/model_spec.*` + benchmark instances under `contracts/instances/`
2. **Workstreams (ownership boundaries)**: `.orchestrator/workstreams.md`
3. **Task files (single-task execution units)**: `.orchestrator/{backlog,active,ready_for_review,blocked,done}/`
4. **Deterministic gates (merge firewall)**: `make gate` and `make test`
5. **Optional automation**: `python scripts/swarm.py ...` (supervisor loop that starts workers, runs gates, updates task state)

## Repo Layout

- `.orchestrator/` — control plane: task queue + state + templates
- `contracts/` — canonical specs (project mode, schemas, assumptions, decisions, model spec)
- `docs/` — runbooks + protocol lock (empirical) + reusable workflow notes
- `scripts/` — deterministic quality gates + swarm supervisor + sweep tool
- `src/` — project code split by responsibility:
  - `src/etl/` (network allowed) — data acquisition + snapshotting + manifests
  - `src/validation/` (no network) — reconciliation + sanity checks
  - `src/analysis/` (no network) — analysis scripts + figures/tables
  - `src/model/` (no assumptions invented) — modeling/simulation (for modeling/hybrid)
- `data/` — data policy:
  - `data/raw/` and `data/processed/` are not committed (append-only snapshots + rebuildable transforms)
  - `data/raw_manifest/` and `data/processed_manifest/` are tracked provenance
  - `data/samples/` are tiny, tracked golden samples used in gates/tests
- `reports/` — generated outputs (figures/tables/validation) + `reports/catalog.yaml` index
- `tests/` — fast deterministic tests (sample-only)

## Quickstart (Best Practice for a New Project)

1. **Decide research mode**: set `mode: empirical | modeling | hybrid` in `contracts/project.yaml`.
2. **Lock the spec (Phase 0)**:
   - Empirical/hybrid: complete `docs/protocol.md` and add versioned schema(s) in `contracts/schemas/`.
   - Modeling/hybrid: complete `contracts/model_spec.*` and add `contracts/instances/benchmark_small/`.
3. **Define ownership boundaries**: edit `.orchestrator/workstreams.md` (Planner-only).
4. **Create 3–5 small tasks** (30–180 minutes each) in `.orchestrator/backlog/` using:
   - `.orchestrator/templates/task_template.md` (generic)
   - `.orchestrator/templates/task_template_w1_w2_etl.md` (ETL)
5. **Run gates locally** (and keep them green):
   - `make gate`
   - `make test`

## Manual Swarm (Recommended First)

Follow `docs/runbook_swarm.md` for the full “golden path”. At a high level:

- Planner creates tasks under `.orchestrator/backlog/`.
- Worker creates a dedicated worktree per task, runs an agent session, and updates only task `## Status` + `## Notes / Decisions`.
- Judge runs `make gate`/`make test`, reviews outputs vs success criteria, and marks task `ready_for_review` or `done`.
- Planner optionally runs `make sweep` to align task folders with each task’s `State:`.

## Automated Swarm (`scripts/swarm.py`)

See `docs/runbook_swarm_automation.md` for a press-go guide.

Key commands:
- `python scripts/swarm.py plan` — show done/claimed/ready tasks
- `python scripts/swarm.py tick --runner local --max-workers 1 --dry-run` — verify selection logic
- `python scripts/swarm.py tmux-start ...` — start a tmux supervisor loop

Safety interlock:
- Unattended mode requires `SWARM_UNATTENDED_I_UNDERSTAND=1` and should only be run in a sandboxed environment (VM/devcontainer/Codespaces) that contains only this repo and no secrets.

## Example “Vertical Slice” (Empirical STR)

This repo currently includes an example empirical protocol (`docs/protocol.md`) and a set of backlog tasks (`T030`–`T060`) intended to produce the first end-to-end artifact:
- ETL snapshot + golden sample → metric module + tests → validation report → figure output.

Once those tasks are implemented and the golden sample exists at `data/samples/growthepie/vendor_daily_rollup_panel_sample.csv`, you should be able to run:

```bash
make gate
make test
python src/validation/validate_vendor_panel.py --sample
python src/analysis/plot_str_timeseries_sample.py
```

Expected outputs:
- `reports/validation/vendor_panel_validation.md`
- `reports/validation/vendor_panel_validation.json`
- `reports/figures/str_timeseries_sample.svg`

## Optional GitHub Automation

The only workflow that should be enabled by default is CI (`.github/workflows/ci.yml`).

Optional agent-triggered workflows (e.g., Claude Code review) live under `docs/optional/github_workflows/`. If you enable them:
- read triggers carefully (comment-based invocation can be abused on public repos),
- configure required secrets,
- prefer restrictive allowlists.

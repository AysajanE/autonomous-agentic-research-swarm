# Autonomous Agentic Research Workflow (Reusable Architecture + Roadmap)

This document defines a reusable, file-based workflow for running multiple AI coding/research agents in parallel on empirical research projects with minimal conflict and minimal human babysitting.

It is designed to work with **subscription CLIs/IDEs** (e.g., Claude Code, Codex CLI) and does **not** assume API-based agent frameworks.

---

## 1) Design goals

**Goal:** A “research team in a repo” where agents can work in parallel, produce auditable outputs, and keep going unattended—while avoiding contradictions and clobbering.

**Non-goals (at first):**
- Not building a complex “agents talk to each other” system.
- Not relying on shared chat as the coordination substrate.
- Not attempting fully automated merges without quality gates.

---

## 1.5) Research modes (required)

Every project must declare a mode:

- **Empirical**: data collection + cleaning + validation + estimation
- **Modeling**: formal model/spec + solution method + experiments/simulation + validation
- **Hybrid**: both (empirical informs model; model generates counterfactuals)

Mode determines which repo modules and contracts are mandatory.

---

## 2) The core architecture (Planner → Workers → Judge)

### Roles

- **Planner (human or agent)**
  - Reads the research plan + current repo state.
  - Decomposes work into small tasks with clear I/O contracts.
  - Assigns tasks to workstreams and defines ownership boundaries.

- **Workers (multiple agents)**
  - Each worker executes *one task at a time* in an isolated workspace (branch/worktree).
  - Writes outputs to the agreed paths and updates only the task’s `Status` + `Notes / Decisions`.
  - Does not coordinate directly with other workers.

- **Judge (human or agent)**
  - Runs deterministic checks (tests/validation/quality gates).
  - Reviews task success criteria against the outputs.
  - Accepts/requests revisions and merges via PRs/commits.

### Why this works

Parallelism fails when workers share mutable state (same branch, same files, “shared kanban brain” everyone edits).

This architecture prevents clashes by making conflicts structurally unlikely:
- **Isolation** (branches/worktrees) prevents accidental interference.
- **Contracts** (schemas, protocol lock, task specs) prevent definition drift.
- **Gates** prevent low-quality merges from propagating downstream.

---

## 3) “Repo as shared memory”: the coordination primitives

Everything important must live on disk in version control:

1. **`AGENTS.md` (the team contract)**
   - Non-negotiables, safety, scope rules, and “how we work here”.
   - Optional: add nested `AGENTS.md` in subdirectories for local rules.

2. **Contracts & locks (protocol/model spec)**
   - Canonical definitions that downstream work must not reinterpret.
   - Empirical: metrics, units, inclusion criteria, regimes, tolerances.
   - Modeling: notation, variables, constraints/objective, assumptions, regimes, solver config, evaluation metrics.
   - The contract files are the highest-authority reference for downstream tasks.

3. **Workstreams + ownership boundaries**
   - Workstream definitions (“who owns what”).
   - Explicit “Owns paths” / “Does NOT own” rules.

4. **Task files**
   - Single Markdown task file per unit of work.
   - **Planner** moves task files across lifecycle folders.
   - Workers update only `Status` + `Notes / Decisions` (and set `State:` accordingly).

5. **Handoff notes**
   - Short integration notes for cross-task dependencies (paths, commands, assumptions).

6. **Provenance manifests**
   - Every externally-derived artifact (data pull, instance set, experiment run) must have a manifest:
     - source/input identifiers
     - timestamp (UTC)
     - command used
     - file list + hashes (sha256)
     - software environment fingerprint (version info)

7. **Quality gates**
   - Deterministic checks that run fast (structure checks, contract checks, unit tests, basic sanity).

### Gate layers (recommended)
1) Structure gates: required files/dirs exist
2) Contract gates: protocol/model spec complete (no TODO stubs)
3) Task hygiene gates: valid task states/required sections
4) Repro gates: can rebuild key artifacts deterministically
5) Scientific sanity gates: domain-specific invariants (identities, constraints feasibility, baseline replication)

---

## 4) Reusable repo skeleton (minimum viable)

Recommended universal skeleton (copy to new projects as-is and then customize contracts + workstreams):

```text
.orchestrator/              # coordination control plane
docs/                       # narrative docs, protocol overview
contracts/                  # canonical specs (schemas/model specs/instances/experiments)
scripts/                    # gates + automation
src/                        # code (etl/analysis/model/validation)
reports/                    # figures/tables/paper/deck/status
AGENTS.md
Makefile
README.md
```

Recommended control-plane mode (bootstrapping): **PR-synchronized**
- Workers operate in isolated branches/worktrees.
- Status updates become visible when branches are pushed/PR’d (not real-time).

Mode-specific expansions:

- Empirical-heavy:
  - `src/etl/`, `data/raw/`, `data/processed/`, `contracts/schemas/`
- Modeling-heavy:
  - `src/model/`, `contracts/model_spec.*`, `contracts/instances/`, `contracts/experiments/`
- Hybrid:
  - both, plus an explicit link contract that maps data → model parameters

---

## 4.5) Universal contract set (standardize across projects)

To keep projects reusable, standardize a small set of canonical contract files that everything else depends on.

### 4.5.1 `contracts/project.yaml` (always)

Contains research mode and high-level invariants:

```yaml
project_name: "..."
mode: empirical | modeling | hybrid
primary_question: "..."
primary_outputs:
  - "reports/..."
risk_policy:
  stop_conditions:
    - "definition ambiguity"
    - "missing credentials"
    - "validation failure beyond tolerance"
```

### 4.5.2 Empirical contracts

- `contracts/schemas/`
  - `contracts/schemas/panel_schema.yaml`
  - `contracts/schemas/raw_<source>_schema.yaml`
- `contracts/data_dictionary.md`

### 4.5.3 Modeling contracts

- `contracts/model_spec.*` (model spec lock)
- `contracts/instances/` (benchmark instances / instance sets)
- `contracts/experiments/` (experiment specs: parameters, seeds, solver config)

Model spec template:

```md
# Model Spec Lock

## Objective / question
## Notation and sets
## Decision variables
## Constraints
## Objective function
## Assumptions (explicit)
## Baselines / benchmark cases
## Solver / method
- exact / heuristic / simulation / proof
- environment and versions
## Outputs (required)
- artifact paths
- evaluation metrics
```

---

## 4.6) Nested `AGENTS.md` placement rules (for reuse)

Always add these nested agent contracts:

1. `.orchestrator/AGENTS.md` — control plane discipline
2. `contracts/AGENTS.md` — “don’t casually edit the spec”
3. `scripts/AGENTS.md` — “gates must be deterministic & fast”
4. `src/AGENTS.md` — code standards and module boundaries
5. `reports/AGENTS.md` — output naming, reproducibility, and “no hand-edits”

Add these depending on project mode:

- Empirical: `src/etl/AGENTS.md`, `data/AGENTS.md`
- Modeling: `src/model/AGENTS.md`, `contracts/instances/AGENTS.md`, `contracts/experiments/AGENTS.md`

---

## 5) Task design: how to make tasks parallelizable

### The “interface-first” rule

A task is parallelizable when it has:
- clear inputs (which files/data it reads),
- clear outputs (paths it produces),
- a stable interface (schema, function signature, registry format),
- and strict file ownership boundaries.

### Task sizing guidelines

Prefer tasks that take **30–180 minutes** for an agent to complete end-to-end.
If a task is larger, split it until it becomes interface-driven.

### Task template

Use `.orchestrator/templates/task_template.md`.

**Non-negotiable sections:**
- `## Outputs` (paths are your integration contract)
- `## Success Criteria` (what “done” means)
- `## Status` and `## Notes / Decisions` (the only sections workers edit)

---

## 6) Isolation and integration (how to prevent clashes)

### Isolation mechanisms

Choose one (in increasing robustness):

1. **Separate branches** per task (minimum).
2. **Git worktrees** per task (recommended for parallel local execution).
3. **Remote sandbox** (Codespaces/VM/devcontainer) per worker for unattended runs.
4. **Cloud agent tasks** (where available) that produce PRs for review.

### Integration mechanism (preferred)

- Every task integrates via PR (or at least a branch merge after judge checks).
- The Judge runs `make gate` (and any task-specific commands) before merging.

---

## 7) Continuous operation (agents work while you’re away)

To run unattended you need:
- an always-on machine (local desktop that won’t sleep, a VM, or Codespaces),
- a session manager (e.g., `tmux`) for long-running terminals,
- headless/non-interactive agent execution settings,
- strict safety boundaries (allowed tools, sandboxing, secrets discipline).

**Important:** “auto-approve tools” should only run in a sandboxed environment that contains *only* the repo and no sensitive files.

---

## 8) Roadmap (incremental implementation plan)

### Stage 0 — Reusable coordination layer (this repo now)

**Deliverables**
- `.orchestrator/` lifecycle folders + templates
- `AGENTS.md` rules
- `contracts/` directory with templates (canonical specs live here)
- `reports/` directory baseline (predictable, reviewable outputs)
- Provenance manifest conventions (every “result” has a manifest + repro command)
- Nested `AGENTS.md` files in sensitive directories (control plane, contracts, scripts, src, reports)
- `docs/protocol.md` protocol lock (or `contracts/model_spec.*` for modeling projects)
- `make gate` runs fast and deterministically

**Success**
- A new agent can start a task without extra human context.

**Stage 0 exit gate (universal)**
- `make gate` passes
- Contracts exist and are non-stub (protocol/model spec complete enough to start work)
- Workstreams filled with ownership boundaries
- At least 2 example tasks exist with success criteria and allowed paths

### Stage 1 — Parallel manual swarm (2–6 workers)

**Deliverables**
- Workstreams filled in `.orchestrator/workstreams.md`
- Worktrees/branches per task
- Judge process (human or agent) running gates and merging

**Success**
- Multiple tasks progress in parallel without file conflicts or definition drift.

**Stage 1 exit gate**
- At least 3 tasks completed in parallel without:
  - definition changes midstream
  - cross-workstream file edits
- Judge can merge using only gates + task criteria (no detective work)

### Stage 2 — Semi-automation (“job controller” scripts)

**Deliverables**
- Script(s) that:
  - pick ready tasks from `.orchestrator/backlog/`,
  - create worktrees/branches,
  - launch headless workers,
  - collect outputs + update task status.

**Success**
- Overnight progress without manual task launching.

**Stage 2 exit gate**
- One overnight run produces:
  - ≥2 completed tasks
  - a status report under `reports/status/`
  - blocked tasks with minimal human questions

### Stage 3 — Continuous supervisor loop

**Deliverables**
- Scheduled “tick” (cron-like) to:
  - assign tasks,
  - re-run gates,
  - reopen tasks as blocked with actionable notes,
  - produce a daily summary.

**Success**
- System runs continuously and only pings you for true decision points.

### Stage 4 — Reliability + evaluation

**Deliverables**
- Formal gate suite (data checks, schema drift checks, regression tests).
- “Fresh start” policy for long-running agents (restart from disk state periodically).
- Metrics: throughput, defect rate, rework rate, time-to-merge.

**Success**
- Sustained multi-week runs with low drift and minimal rework.

---

## 9) New-project bootstrap checklist (copy/paste)

1. Copy the skeleton:
   - `AGENTS.md`, `.orchestrator/`, `contracts/`, `docs/`, `scripts/quality_gates.py`, `src/`, `reports/`, `Makefile`
2. Customize:
   - Set **Research mode** in `contracts/project.yaml` (empirical | modeling | hybrid)
   - Contracts & locks:
     - Empirical/hybrid: `docs/protocol.md`
     - Modeling/hybrid: `contracts/model_spec.*`
   - Workstreams (`.orchestrator/workstreams.md`)
3. Create 3 Phase 0 tasks:
   - Contract lock (protocol/model spec)
   - Quality gates expansion
   - First deliverable (empirical: first ETL; modeling: baseline solver on benchmark instances)
4. Run:
   - `make gate`
5. Start small:
   - 1 planner + 2 workers + 1 judge
6. Only then scale:
   - add more workers and increase parallel tasks

---

## 10) Universal bootstrap runbook (30 minutes)

1. Copy the skeleton into a new repo.
2. Fill `contracts/project.yaml` (mode, primary question, outputs, stop conditions).
3. Fill the primary lock:
   - Empirical: `docs/protocol.md`
   - Modeling: `contracts/model_spec.*`
4. Fill `.orchestrator/workstreams.md` with hard ownership boundaries.
5. Create three tasks in `.orchestrator/backlog/`:
   - contract lock
   - gate expansion
   - first deliverable
6. Run `make gate`.
7. Start with: 1 Planner + 2 Workers + 1 Judge.

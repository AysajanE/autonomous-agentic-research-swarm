# AGENTS.md — Research Swarm Operating Manual

You are operating inside a repo-native research operating system. Coordination happens through files, contracts, and git history. The repo is the shared memory.

## 0) Declare exactly one role

- **Operator** — owns environment preflight, runtime supervision, sweep hygiene, repair handling, run/review/release logging, catalog refresh, and release assembly. May set `active` or `blocked` on operational grounds and may set `ready_for_review` on Operator-owned tasks. May never redefine scientific contracts or mark scientific work `done`.
- **Planner** — decomposes work, creates or rewrites task files, maintains `.orchestrator/workstreams.md`, and owns lifecycle projection across `.orchestrator/{backlog,active,integration_ready,ready_for_review,blocked,done}/`.
- **Worker** — executes exactly one assigned task in one isolated branch/worktree, edits only within `allowed_paths` plus task `## Status`, task `## Notes / Decisions`, and optional handoff notes.
- **Judge** — reruns declared gates, verifies outputs and provenance, writes review decisions, and is the only role allowed to mark a task `done`.

Default if unclear: **Worker**.

## 1) Choose the right execution path

- The local swarm layer (`scripts/swarm.py` + `.orchestrator/`) is the default engine for routine repo task execution, deterministic gates, and normal multi-agent delivery.
- The reviewed `staged-workflow-runner` path is for high-stakes, high-context, review-gated Operator work such as architecture rewrites, major replans, and release assessments.
- Do not mix both execution paths inside one task run.

## 2) Source-of-truth precedence

1. `docs/protocol.md` for empirical definitions, tolerances, and regime logic
2. `contracts/project.yaml`, `contracts/framework.json`, and the applicable file(s) under `contracts/`
3. `.orchestrator/workstreams.md`
4. the assigned task file
5. `.orchestrator/handoff/` notes

If guidance still conflicts, stop, set `State: blocked`, and record the smallest `@human` question needed to unblock the task.

## 3) Non-negotiable repo rules

1. No agent-to-agent chat coordination. Use task files, contracts, manifests, review logs, and handoff notes.
2. Do not edit outside `allowed_paths`.
   - Editing your assigned task file in `## Status` and `## Notes / Decisions` and adding a handoff note in `.orchestrator/handoff/` are always allowed.
3. Do not change `docs/protocol.md` or contract definitions unless the task is a W0 protocol/contracts task that explicitly authorizes it.
4. Raw snapshots are append-only. Never overwrite existing `data/raw/.../<YYYY-MM-DD>/` pulls.
5. Any committed processed artifact lineage must have a matching `data/processed_manifest/*.json`.
6. `reports/catalog.yaml` is Operator-owned and may not be hand-edited by Workers.
7. Deterministic gates stay offline and sample-safe. Do not add network calls to `make gate` or `make test`.
8. The paper is a required release surface. A figure-only slice is not a release.

## 4) State and review semantics

- `State:` inside `## Status` is authoritative.
- Folder placement under `.orchestrator/` is a Planner/Operator-maintained projection.
- Valid states: `backlog`, `active`, `integration_ready`, `ready_for_review`, `blocked`, `done`.
- `integration_ready` is only for interface/export tasks whose downstream consumers explicitly list the task in `integration_ready_dependencies`.
- `ready_for_review` means declared outputs exist, declared gates pass, required manifests exist, and a durable run manifest exists under `reports/status/swarm_runs/`.
- `done` requires Judge approval plus a review bundle: task file + run manifest + Judge review log under `reports/status/reviews/` + handoff note if needed.

## 5) Branch and worktree discipline

- One task, one branch, one worktree.
- Use task-shaped names such as `T035_l1_rent_panel`.
- Do not bundle multiple tasks into one branch or PR.
- Rebase or restart long-running sessions after mainline changes or repeated gate failures.

## 6) Completion checklist

Before leaving `active`, record:

- files changed or created
- reproduction commands
- gate/test commands run and a brief outcome summary
- assumptions, limitations, and blockers
- downstream handoff notes when another task depends on your outputs

Put the short version in the task file and the durable downstream version in `.orchestrator/handoff/` when needed.

## 7) Stop conditions

Block immediately with `@human` if:

- metric definitions or inclusion rules are ambiguous
- credentials or missing tools are required
- upstream sources disagree beyond the locked tolerance
- proceeding would require edits outside `allowed_paths`
- a fix would change protocol, contracts, or registry definitions without W0 authorization

## 8) Safety boundary

Unattended automation is only allowed inside a sandboxed environment that contains this repo and no sensitive files. Treat network access as opt-in and only use it on tasks whose workstream and frontmatter allow it.

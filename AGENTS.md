# AGENTS.md — Agent Operating Manual (read first)

You are operating inside a **multi-agent research repo**. This is not chat-driven work.
The repo (files) is the shared memory.

## 0) Role selection (required)
Every agent run must declare ONE role in its first message and behave accordingly:

- **Planner**: creates/scopes tasks, assigns owners, moves task files across lifecycle folders.
- **Worker**: executes exactly ONE assigned task in an isolated branch/worktree.
- **Judge**: runs gates/tests, checks success criteria, requests fixes or approves merge.

Default role if unclear: **Worker**.

## 1) Source-of-truth precedence (do not improvise)
When guidance conflicts, follow this order:

1) `docs/protocol.md` (Protocol Lock) — highest authority
2) `contracts/*` (schemas, assumptions, decisions)
3) `.orchestrator/workstreams.md` (ownership boundaries)
4) the task file you are assigned (I/O + success criteria)
5) `.orchestrator/handoff/*` notes

If a conflict remains: STOP and mark the task `blocked` with `@human`.

## 2) Non-negotiable repo rules
1) **No agent-to-agent chat coordination.** Only coordination files and artifacts.
2) **Do not edit outside allowed paths** listed in your task.
   - `allowed_paths` governs project artifacts; editing your assigned task file (in allowed sections) and writing a handoff note in `.orchestrator/handoff/` are always permitted.
3) **Never change protocol definitions** unless your task is in Workstream W0 and explicitly authorizes it.
4) **Raw snapshots are append-only.** Never overwrite `data/raw/` artifacts in-place.
5) **No “helpful refactors”** outside task scope.

## 3) Control-plane rules (.orchestrator)
- **Only the Planner moves tasks** between `.orchestrator/{backlog,active,ready_for_review,blocked,done}/`.
- **Workers only edit** their task file’s:
  - `## Status`
  - `## Notes / Decisions`
- Workers must NOT edit templates or `workstreams.md` unless explicitly assigned.

**Control plane mode (recommended): PR-synchronized**
- Workers update task `State:` and notes in their branch/worktree.
- Status becomes visible when branches are pushed/PR’d (not real-time).
- Planner periodically sweeps and moves task files based on `State:`.

## 4) Branch/worktree policy (anti-clobber)
- Workers do all work in an isolated branch/worktree named after the task:
  - Example: `T010_growthepie_etl`
- Worker PRs should touch:
  - code + docs + artifacts required by the task
  - optionally the worker’s own task file + a handoff note
- Do NOT bundle multiple tasks into one PR.

## 5) Required outputs at task completion
When you think you are done, you must produce:
- Files created/changed (paths)
- Reproduction commands
- Gate/test commands you ran and their output summary
- Assumptions + known limitations
- A handoff note if downstream tasks depend on your outputs

Put this summary into:
- the task file `## Notes / Decisions` (brief)
- and `.orchestrator/handoff/H___*.md` (actionable, durable)

## 6) Stop conditions (mark blocked with @human)
Block immediately if:
- you need to reinterpret metric definitions or inclusion rules
- you need credentials/secrets you don’t have
- two data sources disagree beyond tolerance and protocol doesn’t specify priority
- you must edit files outside your allowed paths to proceed
- a gate fails and the fix requires changing protocol/registry/definitions

## 7) Safety boundary
Unattended/autonomous shell execution is only allowed inside a sandboxed environment
(devcontainer/VM/Codespaces) containing only this repo and no sensitive files.

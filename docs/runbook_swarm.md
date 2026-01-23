# Swarm Runbook (minimal, PR-synchronized control plane)

This is the “golden path” for running a Planner + multiple Workers + a Judge using this repo template.

For unattended automation (tmux supervisor loop), see `docs/runbook_swarm_automation.md`.

## Prereqs

- Work in a sandboxed environment (VM/devcontainer/Codespaces) containing only this repo.
- Ensure `make gate` passes on `main` before starting parallel work.

## 1) Planner: create/activate tasks

1. Create tasks under `.orchestrator/backlog/` using a template:
   - Generic: `.orchestrator/templates/task_template.md`
   - W0 protocol/contracts: `.orchestrator/templates/task_template_w0_protocol.md`
   - W1/W2 ETL: `.orchestrator/templates/task_template_w1_w2_etl.md`
2. When assigning a task, the Planner may either:
   - set `State: active` and run `make sweep` (recommended; keeps governance consistent), or
   - directly `git mv` the task file to `.orchestrator/active/` (Planner-only action).

Concurrency guidance:
- Prefer least-privilege `allowed_paths` (specific files/prefixes), not broad directories.
- Avoid running multiple tasks in the same workstream concurrently unless interfaces are locked.

## 2) Worker: create an isolated worktree per task (recommended)

From the repo root:

```bash
TASK_ID=T020
BRANCH=${TASK_ID}_short_name
git worktree add ../wt-${TASK_ID} -b ${BRANCH} .
cd ../wt-${TASK_ID}
```

Important: run your agent CLI from the task worktree directory so nested `AGENTS.md` files are applied.

## 3) Worker: run an agent session (headless example)

Use short, reset-friendly sessions:
- timebox: 90–180 minutes (or a fixed max-turn count)
- restart after any merge or after any failed `make gate`

**Claude Code (example; adjust flags to your installed version):**

```bash
claude -p "Role: Worker. Task: .orchestrator/active/${TASK_ID}__*.md. Follow AGENTS.md." \
  --output-format json \
  --max-turns 30 \
  --allowedTools "Bash(git*),Bash(make*),Bash(python*)"
```

**Codex CLI (example; adjust to your installed version):**

```bash
# Non-interactive worker run (matches the style used by scripts/swarm.py):
codex -a on-request exec --sandbox workspace-write -C . \
  "Role: Worker. Task: .orchestrator/active/${TASK_ID}__*.md. Follow AGENTS.md."

# Fully unattended (no approval prompts; external sandbox only):
codex -a never exec --sandbox workspace-write -C . \
  "Role: Worker. Task: .orchestrator/active/${TASK_ID}__*.md. Follow AGENTS.md."
```

## 4) PR-synchronized status cadence (recommended)

Because `.orchestrator/` is PR-synchronized (not live), keep the Planner informed:

- Push your branch at least every 30–60 minutes, and whenever you change `State:` or produce outputs.
- Open a PR early; use the PR description as a lightweight status log if needed.

## 5) Judge: verify and advance lifecycle

From a clean checkout (or the Worker PR branch):

```bash
make gate
make test
```

Then:
- If acceptable: set task `State: done` (or `ready_for_review` first), and the Planner sweeps the file into the matching folder.
- If revisions needed: write actionable feedback in `## Notes / Decisions` and set `State: active`.

## 6) Fresh start policy (drift control)

Long-running agent sessions drift. Use “fresh starts”:

- Restart Worker sessions after any merge to `main`.
- Restart if a session exceeds your timebox or fails gates twice.
- Prefer short tasks and frequent merges over long monolithic runs.

## 7) Running unattended overnight (practical note)

Codespaces and other hosted environments may stop on inactivity by default. If you need true overnight runs:

- use a VM + `tmux`, or
- configure your environment’s idle timeout appropriately, and verify it stays running before relying on it.

## 8) Safety defaults

- Prefer tool allowlists (`git`, `make`, `python`) over broad shell permissions.
- Avoid unattended runs on machines with secrets or sensitive files.

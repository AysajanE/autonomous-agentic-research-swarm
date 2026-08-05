# Autonomous Agentic Research Swarm

**Run AI coding agents on a research project without letting them wreck it.**

You give the swarm a queue of research tasks. It runs AI agents — Codex CLI or Claude
Code — one task at a time, each in its own git worktree, each allowed to touch only the
files its task declared. Work that fails its checks does not merge. No agent can approve
its own work. Every run leaves a manifest behind, so months later you can still answer
"who produced this number, from what input, and who checked it."

It is built for research that has to survive scrutiny: empirical data work, modeling and
simulation, or a hybrid of the two.

---

## See it work in 60 seconds

```bash
make demo
```

No API key. No network. No cost. Nothing in your repository changes.

The demo builds a throwaway research project in a temp directory and runs two Workers and
a Judge through the real runtime, using a scripted `mock` agent instead of a live model:

| | what happens |
|---|---|
| **Act 1** | A Worker writes exactly the file its task declared → gates pass, work is accepted, artifact appears |
| **Act 2** | A Worker tries to rewrite `contracts/project.yaml`, which its task does not own → **refused**, task blocked, file byte-identical |
| **Act 3** | The Judge is asked to approve the work this same session just produced → **refused**, with reasons written to a review log |

```
========================================================================
  Act 2 — a Worker that reaches out of scope
========================================================================
--- run-task result ----------------------------------------------------
  { "blocked_reasons": ["executor_failed"],
    "state_after": "blocked", "state_before": "backlog", "task_id": "T901" }
------------------------------------------------------------------------
  BLOCKED  the out-of-scope write was refused
           contracts/project.yaml is byte-identical
```

Those are the three failures that ruin unsupervised agent runs — scope creep, silent
merges, and self-approval — being stopped by the same code that runs in production.

---

## Who this is for

You are running AI agents on work where being wrong is expensive and being unable to
*prove* you were right is just as bad. A paper, a report, a model someone will act on.

If you just want an agent to refactor a service, this is far too much machinery. The
overhead here buys you provenance, and provenance is only worth paying for when someone
will eventually audit the result.

## The problem it solves

Agentic workflows tend to fail the same five ways:

- scope widens quietly — the agent "helpfully" edits things it was not asked to touch
- state lives in a chat window, so nothing survives the session
- review is bolted on afterwards, if at all
- parallel agents collide because nobody owns which files
- outputs exist with no clean chain of evidence back to their inputs

The fix here is one idea:

> **the repository is the shared memory**

Agents do not coordinate through conversation. They coordinate through task files,
contracts, manifests, review logs, and git history. Everything is inspectable because
everything is a file you can `cat`, `diff`, and `git log`.

---

## How it works

```
  .orchestrator/backlog/T042.md        a task file: scope, gates, allowed paths
             │
             ▼
   ┌──── Planner ─────┐                scopes work, writes tasks   (human-approved)
   │                  │
   ▼                  │
  swarm.py tick ──────┘                claims one ready task, takes a lease
   │
   ▼
  git worktree ../wt-T042              one task · one branch · one worktree
   │
   ▼
  Worker  (Codex CLI / Claude Code)    may write ONLY the task's allowed_paths
   │                                   out-of-scope write ⇒ run fails
   ▼
  declared gates                       `make gate`, tests, project checks
   │                                   offline by default; failure ⇒ state: blocked
   ▼
  Judge  (separate actor)              reruns gates, verifies provenance
   │                                   only role that can mark work `done`
   ▼
  reports/status/{swarm_runs,reviews}/ durable run manifest + review log
```

Four roles, enforced by the runtime rather than by good intentions:

| role | may do | may not |
|---|---|---|
| **Planner** | scope work, write task files, maintain the queue | execute tasks |
| **Worker** | execute exactly one task in one worktree | write outside `allowed_paths` |
| **Judge** | rerun gates, verify provenance, mark work `done` | review its own session's work |
| **Operator** | preflight, supervision, repair, release assembly | skip the review path |

### The unit of work is a task file, not a prompt

Every task is a markdown file whose YAML frontmatter *is* the contract:

```yaml
task_id: T060
title: "Analysis: release STR figures and tables from validated artifacts"
role: Worker
dependencies: ["T040", "T050"]
allowed_paths:                       # the only files this Worker may write
  - "src/analysis/build_str_release_outputs.py"
  - "reports/figures/str_ecosystem_timeseries.svg"
  - "reports/tables/str_regime_summary.csv"
disallowed_paths:
  - "docs/protocol.md"
  - "contracts/"
outputs:
  - "reports/tables/str_regime_summary.csv"
gates:
  - make gate                        # must pass or the task blocks
stop_conditions:
  - "Contract ambiguity"             # stop and ask; do not improvise
```

The `State:` field inside the file is authoritative. The `backlog/`, `active/`,
`blocked/`, `done/` folders are only a projection that tooling maintains — so a task
never gets "lost" because someone moved a file.

---

## What actually stops a bad run

These are enforced in code, not advice in a prompt. Most of them fire in `make demo`.

**Scope**
- A Worker writing outside its `allowed_paths` fails the run. The write does not land.
- One task, one branch, one worktree. Parallel agents cannot collide.
- Leases prevent two runners from claiming the same task.

**Review**
- Only a Judge marks work `done`, and the Judge reruns the gates itself.
- An actor-separation window blocks a session from reviewing work it just produced.
- Commits landing after a run manifest is sealed are detected (`post_manifest_commits`).

**Gate execution**
- Gates may only be `make <target>` or `python <repo-relative>.py`. Inline code
  (`-c`), module execution (`-m`), and stdin are rejected — an agent-authored gate can
  never become an arbitrary-code channel.
- Gates run with the network disabled by default and an allowlisted environment.

**Blast radius**
- Unattended runs refuse to start unless you have attested containment — *and* the
  runtime scans for readable AWS, SSH, gcloud, netrc, and Docker credentials and
  refuses if it finds any. The waiver must live in the signed attestation, never in an
  environment variable.
- `scaffold: true` fails closed: a pack still claiming to be a scaffold cannot ship
  real outputs.

**Scientific integrity**
- Preregistration locks can block analysis tasks until the analysis plan is frozen.
- A claim–evidence ledger ties manuscript numbers to the artifacts that produced them.
- Cross-family referees: work authored by one model family is reviewed by another.

---

## Run real agents

The demo uses `mock`. To use live models, swap the backend:

```bash
# one task, attended — you watch it
python3.11 scripts/swarm.py run-task --task-id T042 --executor-backend codex

# the whole ready queue
python3.11 scripts/swarm.py tick --executor-backend codex --max-workers 2

# crash-only supervisor loop (long-running)
python3.11 scripts/swarm.py supervise --executor-backend codex
```

**Engines.** Workers run through Codex CLI (`--executor-backend codex`). The Planner and
the referee panel run through Claude Code (`--planner-backend claude`,
`--referee-backend claude`). Models are pinned in `contracts/framework.json` under
`executors`, not hardcoded in the runtime.

**Cost.** Executor token usage is recorded per run and aggregated:

```bash
python3.11 scripts/swarm.py costs
```

Tasks declare their own ceiling in frontmatter — `budgets: {max_wall_clock: 1h,
max_tokens: 100000, max_cost_usd: 10}`.

**Before going unattended.** Read [`docs/operator_runbook.md`](docs/operator_runbook.md).
Unattended mode requires, deliberately:

```bash
python3.11 scripts/swarm.py attest-containment --attested-by <name>
python3.11 scripts/swarm.py ack-vendor-policy --vendor <vendor> --note <policy> --acked-by <name>
export SWARM_UNATTENDED_I_UNDERSTAND=1
```

and it will still refuse to run if your home directory has readable cloud credentials.
Run it in a sandbox or container that holds only this repository. `--codex-sandbox
danger-full-access` exists; it is not a default and you should have a reason.

---

## Getting started

### Prerequisites

- Python **3.11** (the Makefile calls `python3.11`; 3.9 will not work)
- `git`
- Optional: `quarto` (paper builds), `tmux` (supervisor sessions), `gh` (PRs)

### Install and verify

```bash
python3.11 -m pip install .
make demo     # 60-second narrated walkthrough, no cost
make gate     # deterministic contract + integrity gates
make test     # 554 offline tests
```

`make gate` prints one `ok=True` line per gate; many report `skipped: True` because they
only apply to other modes or task kinds. That is normal.

### Look at the live control plane

```bash
python3.11 scripts/swarm.py status --no-fetch
```

```
Swarm status
backlog: (none)
active: (none)
blocked: (none)
done: T000, T005, T010, T015, T020, ...
journal: events=1 malformed=0 escalations=0
```

> **Note:** `swarm.py tick` appends to the provenance journal at
> `reports/status/events/events.jsonl` **even with `--dry-run`**, and an uncommitted
> journal entry will make `make test` fail two release-integrity tests. If that happens,
> `git status` will show the file; remove or commit it. Prefer `status` for read-only
> inspection.

### Read the framework in this order

1. [AGENTS.md](AGENTS.md) — role boundaries and operating rules
2. [`contracts/framework.json`](contracts/framework.json) — capabilities, roles, states, engines
3. [`.orchestrator/workstreams.md`](.orchestrator/workstreams.md) — how work is grouped
4. [`docs/runbook_swarm.md`](docs/runbook_swarm.md) — the manual loop, step by step
5. [`docs/operator_runbook.md`](docs/operator_runbook.md) — supervision, escalation, attestation
6. [`contracts/project.yaml`](contracts/project.yaml) — the project currently instantiated

There is also a one-page visual: [`docs/swarm_workflow_poster.svg`](docs/swarm_workflow_poster.svg).

---

## Start your own research project

This repo ships the reusable kernel (`scripts/`) plus one reference project (`src/`).
To start a new project, generate a pack and replace the project-specific parts:

```bash
python3.11 scripts/swarm_init.py --mode empirical --output ../my-pack
make -C ../my-pack gate
```

Use `--mode modeling` or `--mode hybrid` for the other templates. Then:

1. Set `project.package_name`, paths, and workstream meanings in `contracts/pack.json`
2. Define the project in [`contracts/project.yaml`](contracts/project.yaml)
3. Fill the mode contracts — `docs/protocol.md` (empirical),
   [`contracts/model_spec.md`](contracts/model_spec.md) (modeling),
   [`contracts/hybrid_interface_v1.yaml`](contracts/hybrid_interface_v1.yaml) (hybrid)
4. Replace the `src/analysis/project_analysis.py` placeholder with your own science
5. Write your task queue under `.orchestrator/`
6. **Set `"scaffold": false` in `contracts/pack.json`** once the pack does real work

Step 6 is not optional. `scaffold: true` fails closed: as soon as the pack produces run
manifests, figures, tables, or processed data, `make gate` fails with
`scaffold_asserted_on_instantiated_repo`. That is the flag telling you the pack has
graduated from template to project.

You should not need to edit `scripts/` — that is the kernel. What you replace is `src/`
(your science) and the contracts. `swarm_init` gives you a contract-valid, orchestration-
testable scaffold; it deliberately does **not** generate a runnable analysis pipeline.

---

## Research modes

| mode | for | key contracts |
|---|---|---|
| **empirical** | source data → processed datasets → validation → analysis → manuscript | [`docs/protocol.md`](docs/protocol.md), [`registry/`](registry/), [`data/*_manifest/`](data/) |
| **modeling** | solvers, simulation, optimization, proofs | [`contracts/model_spec.md`](contracts/model_spec.md), [`contracts/instances/`](contracts/instances/), [`contracts/experiments/`](contracts/experiments/) |
| **hybrid** | empirical outputs feeding declared modeling instances | [`contracts/hybrid_interface_v1.yaml`](contracts/hybrid_interface_v1.yaml) |

In hybrid mode, modeling work consumes declared instance manifests — never arbitrary
empirical data paths. That bridge is the only sanctioned crossing.

## What is proven, and what is not

Being straight about this, because it changes whether you should adopt it:

**Exercised end to end on a real project.** The control plane, the swarm runtime, the
deterministic gate and Judge review path, and the full empirical mode — source data
through figures, tables, manuscript, paper build, and release manifest. The reference
project is an empirical study of L2-to-L1 rent, defined in
[`contracts/project.yaml`](contracts/project.yaml).

**Architecturally present, not yet exercised to the same depth.** Modeling runtime
maturity against a populated model spec and live instance set; hybrid runtime maturity
beyond the bridge contract itself.

So: the framework is designed for three modes, and the deep evidence today is empirical.

## Glossary

| term | meaning |
|---|---|
| **kernel** | the reusable machinery in `scripts/`. You do not edit it per project. |
| **pack** | one project's contracts, tasks, and science (`contracts/`, `src/`, `.orchestrator/`) |
| **gate** | a declared check that must pass for work to count |
| **lease** | a claim on a task, so two runners never take the same one |
| **projection** | folder placement, derived from the authoritative `State:` field |
| **run manifest** | the durable record of one execution under `reports/status/swarm_runs/` |
| **integration_ready** | a state for interface work downstream tasks need before full review |
| **STR** | Settlement Take Rate, the reference project's metric; pack-specific, not framework |

## Repository map

| path | contents |
|---|---|
| [`AGENTS.md`](AGENTS.md) | role boundaries and operating rules |
| [`.orchestrator/`](.orchestrator/) | task lifecycle, templates, handoffs, control-plane state |
| [`contracts/`](contracts/) | framework and project policy, schemas, instances, experiments |
| [`docs/`](docs/) | runbooks, prompts, protocol |
| [`scripts/`](scripts/) | the kernel: swarm runtime, gates, sweep, release assembly |
| [`src/`](src/) | pack-owned science: ETL, validation, analysis, modeling |
| [`data/`](data/), [`registry/`](registry/) | manifest-backed datasets and registry surfaces |
| [`reports/`](reports/) | validation, figures, tables, paper, releases, run manifests, reviews |
| [`tests/`](tests/) | fast offline verification |

## Design principles

- the repository is the shared memory
- task-file state is authoritative; folder placement is only a projection
- contracts outrank chat
- one task executes in one isolated worktree
- gates stay deterministic and offline by default
- review and release artifacts are required outputs, not metadata
- agents stop on ambiguity instead of widening scope

## License

See [LICENSE](LICENSE).

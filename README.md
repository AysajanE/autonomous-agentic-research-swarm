# Autonomous Agentic Research Swarm

Autonomous Agentic Research Swarm is a repo-native framework for executing research through explicit contracts, task state, review gates, and git-scoped work isolation.

This repository should be read framework-first. It currently includes an empirical research project that has been used as a reference implementation to exercise the framework end to end, but the repository itself is not defined by that single project.

The framework is designed to support three research modes:

- empirical
- modeling
- hybrid

Its central operating idea is simple:

> the repository is the shared memory

Agents do not rely on hidden conversational context to coordinate. They coordinate through task files, contracts, manifests, review logs, release artifacts, and git history. That makes the workflow inspectable, reproducible, and reviewable.

## Why This Exists

Most agentic workflows break down on the same points:

- work scope widens implicitly
- state lives in chat instead of durable files
- review and provenance are bolted on after the fact
- parallel work collides because ownership is unclear
- release outputs exist without a clean chain of evidence

This framework addresses those failures directly by treating the repository as both the execution substrate and the control plane.

## Framework Overview

The framework is built from a small set of explicit architectural pieces.

### Repo-native control plane

The local control plane lives under [`.orchestrator/`](.orchestrator/). Task markdown files carry the authoritative `State:` field, while folder placement under backlog, active, blocked, and done is only a projection maintained by tooling.

Relevant surfaces:

- [`.orchestrator/`](.orchestrator/)
- [`.orchestrator/workstreams.md`](.orchestrator/workstreams.md)
- [`scripts/sweep_tasks.py`](scripts/sweep_tasks.py)

### Contract-first execution

The framework separates framework policy from project policy.

- [`contracts/framework.json`](contracts/framework.json) defines framework capabilities, roles, states, task semantics, execution engines, and release policy.
- [`contracts/pack.json`](contracts/pack.json) defines project-owned names, paths, workstream meanings, and the compatible kernel range.
- [`contracts/kernel_interface.json`](contracts/kernel_interface.json) versions the kernel surfaces consumed by packs.
- [`contracts/project.yaml`](contracts/project.yaml) defines the currently instantiated research project.
- [`docs/protocol.md`](docs/protocol.md) locks empirical definitions when the current project is empirical.
- [`contracts/model_spec.md`](contracts/model_spec.md) is the modeling specification surface.
- [`contracts/hybrid_interface_v1.yaml`](contracts/hybrid_interface_v1.yaml) defines the only allowed empirical-to-modeling bridge.

That separation is deliberate. The framework is meant to be reusable across projects; the current project contract is only one instantiation.

### Explicit role separation

The operating model is built around four formal roles:

- `Planner`: scopes work, writes tasks, maintains workstreams, and manages lifecycle projection
- `Worker`: executes exactly one task in one isolated branch/worktree and edits only the allowed scope
- `Judge`: reruns declared gates, verifies outputs and provenance, and is the only role that can mark scientific work `done`
- `Operator`: owns preflight, supervision, repair handling, release assembly, and shared operational surfaces

The role model is enforced through [AGENTS.md](AGENTS.md), [`contracts/framework.json`](contracts/framework.json), and the prompt templates under [`docs/prompts/`](docs/prompts/).

### Worktree-scoped execution

The framework assumes strict task isolation:

- one task
- one branch
- one worktree

Tasks declare bounded ownership through fields such as:

- `allowed_paths`
- `outputs`
- `gates`
- `stop_conditions`

This keeps parallelism tractable and makes it clear when the correct action is to block rather than improvise.

### Deterministic gates and review bundles

The merge firewall is designed to stay offline and deterministic by default.

Primary runtime and gate surfaces:

- [`scripts/swarm.py`](scripts/swarm.py)
- [`scripts/quality_gates.py`](scripts/quality_gates.py)
- [`scripts/sweep_tasks.py`](scripts/sweep_tasks.py)
- [`scripts/release_assembly.py`](scripts/release_assembly.py)

Durable runtime and review artifacts live under:

- [`reports/status/swarm_runs/`](reports/status/swarm_runs/)
- [`reports/status/reviews/`](reports/status/reviews/)

The framework treats these review bundles as first-class outputs, not optional metadata.

### Two execution paths

The framework defines two execution paths.

- The default path is the local swarm runtime: [`scripts/swarm.py`](scripts/swarm.py) plus [`.orchestrator/`](.orchestrator/).
- The high-stakes path is the reviewed `staged-workflow-runner`, reserved for major replans, architecture rewrites, and release assessments under Operator control.

This separation prevents ordinary task execution from being overloaded with high-consequence synthesis work.

## Supported Research Modes

The framework supports the modes declared in [`contracts/framework.json`](contracts/framework.json).

### Empirical

Empirical mode is for workflows that move from source acquisition to processed datasets, validation artifacts, analytical outputs, manuscript source, and release surfaces.

Relevant framework surfaces include:

- [`docs/protocol.md`](docs/protocol.md)
- [`registry/`](registry/)
- [`data/raw_manifest/`](data/raw_manifest/)
- [`data/processed_manifest/`](data/processed_manifest/)
- [`reports/validation/`](reports/validation/)

### Modeling

Modeling mode is for solver, simulation, optimization, or proof-oriented workflows that require explicit instance and experiment definitions rather than informal inputs.

Relevant framework surfaces include:

- [`contracts/model_spec.md`](contracts/model_spec.md)
- [`contracts/instances/`](contracts/instances/)
- [`contracts/experiments/`](contracts/experiments/)
- [`src/model/`](src/model/)
- [`reports/models/`](reports/models/)

### Hybrid

Hybrid mode is for workflows where empirical outputs are transformed into declared modeling instances through a contract-bound interface.

Relevant framework surfaces include:

- [`contracts/hybrid_interface_v1.yaml`](contracts/hybrid_interface_v1.yaml)
- [`contracts/instances/`](contracts/instances/)
- [`contracts/experiments/`](contracts/experiments/)

The key rule is that modeling work consumes declared instance manifests, not arbitrary empirical data paths.

## Framework vs. Current Repository Instance

This repository currently contains both:

- the framework itself
- one active reference project instance

The current project instance, defined in [`contracts/project.yaml`](contracts/project.yaml), is an empirical research project on L2-to-L1 rent. It should be understood as a real end-to-end validation of the framework's empirical path, not as the definition of the framework.

At the current state of this repository, the strongest operational evidence is:

- the repo-native control-plane model is working
- the local swarm runtime is working
- the deterministic gate and Judge review path is working
- the empirical mode has been exercised through release assembly on a real project

What is present architecturally but not yet exercised to the same depth:

- full modeling runtime maturity on a populated model specification and live instance set
- full hybrid runtime maturity beyond the current bridge contract

That distinction matters. The framework supports empirical, modeling, and hybrid work by design, but the current deepest evidence comes from the empirical reference implementation.

## Repository Structure

- [`AGENTS.md`](AGENTS.md): role boundaries and operating rules
- [`.orchestrator/`](.orchestrator/): task lifecycle, templates, handoffs, and control-plane state
- [`contracts/`](contracts/): framework policy, project contract, model spec, hybrid interface, schemas, instances, and experiments
- [`docs/`](docs/): runbooks, prompts, and protocol documents
- [`scripts/`](scripts/): swarm runtime, quality gates, lifecycle sweep, and release assembly
- [`src/`](src/): implementation surfaces for ETL, validation, analysis, and modeling
- [`registry/`](registry/): registry surfaces for empirical projects
- [`data/`](data/): raw, processed, sample, and manifest-backed datasets
- [`reports/`](reports/): validation outputs, figures, tables, models, paper artifacts, release manifests, and review logs
- [`tests/`](tests/): fast offline verification

## How To Work With This Repo

You can use this repository in two different ways.

### 1. Operate the current reference instance

Use the repo as-is to inspect or extend the current empirical project that has been used to exercise the framework.

### 2. Use the framework for a different research project

Generate a compatible inactive pack scaffold, then replace its project-specific contract surfaces:

```sh
python3.11 scripts/swarm_init.py --mode empirical --output ../my-empirical-pack
make -C ../my-empirical-pack gate
```

- update `contracts/pack.json`
- update [`contracts/project.yaml`](contracts/project.yaml)
- update the relevant empirical, modeling, or hybrid contracts
- define the task queue under [`.orchestrator/`](.orchestrator/)
- keep the same role, state, gate, and review semantics

Use `--mode modeling` or `--mode hybrid` for the other program templates. The generated Makefile points at this repo-native kernel for M5a; physical installable packaging remains the separate M5b decision.

The framework is intended to generalize. The current empirical project is only one concrete instantiation.

## Getting Started

### Prerequisites

- Python `3.11`
- `git`
- `pip`

Useful optional tools:

- `quarto`
- `tmux`
- `gh`

### Install

```bash
python -m pip install .
```

### Validate the repository

```bash
make gate
make test
```

### Inspect control-plane status

```bash
python scripts/swarm.py plan --remote origin --base-branch main
```

### Read the framework in the right order

1. [AGENTS.md](AGENTS.md)
2. [`contracts/framework.json`](contracts/framework.json)
3. [`.orchestrator/workstreams.md`](.orchestrator/workstreams.md)
4. [`docs/runbook_swarm.md`](docs/runbook_swarm.md)
5. [`docs/runbook_swarm_automation.md`](docs/runbook_swarm_automation.md)
6. the current project contract in [`contracts/project.yaml`](contracts/project.yaml)

For modeling or hybrid work, then continue with:

- [`contracts/model_spec.md`](contracts/model_spec.md)
- [`contracts/hybrid_interface_v1.yaml`](contracts/hybrid_interface_v1.yaml)
- [`contracts/instances/`](contracts/instances/)
- [`contracts/experiments/`](contracts/experiments/)

## Design Principles

The framework is built around a small set of strong rules:

- the repository is the shared memory
- task-file state is authoritative
- folder placement is only a projection
- contracts outrank chat
- one task executes in one isolated worktree
- gates stay deterministic and offline by default
- review and release artifacts are required outputs
- agents should stop on ambiguity instead of widening scope informally

## Current Status

This repository now demonstrates a complete empirical reference run through figures, tables, manuscript source, paper build, catalog, and release manifest surfaces. That is evidence that the framework can carry a real research project end to end.

It is not yet evidence that every supported mode has equal runtime maturity. The framework is broader than the current reference project, and the README is written to reflect that boundary explicitly.

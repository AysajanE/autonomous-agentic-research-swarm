# Empirical Battle-Test Retrospective

Date: 2026-04-12

This document records the retrospective analysis prepared after the end-to-end empirical battle test of the Autonomous-Agentic Research Swarm. It is intended for internal team sharing and preserves the previously delivered analysis output in full.

## Recorded Analysis Output

**Assessment**
I read the live control-plane and battle-test record: [scripts/swarm.py](/Users/aeziz-local/Research/autonomous-agentic-research-swarm/scripts/swarm.py), [scripts/quality_gates.py](/Users/aeziz-local/Research/autonomous-agentic-research-swarm/scripts/quality_gates.py), [docs/runbook_swarm_automation.md](/Users/aeziz-local/Research/autonomous-agentic-research-swarm/docs/runbook_swarm_automation.md), the local autonomy docs in [.local/runbooks/operator_stepwise_swarm_workflow.md](/Users/aeziz-local/Research/autonomous-agentic-research-swarm/.local/runbooks/operator_stepwise_swarm_workflow.md) and [.local/prompts/operator_queue_driver_prompt.md](/Users/aeziz-local/Research/autonomous-agentic-research-swarm/.local/prompts/operator_queue_driver_prompt.md), the role prompts, the done-task chain from [T035](/Users/aeziz-local/Research/autonomous-agentic-research-swarm/.orchestrator/done/T035_onchain_l1_rent_etl_and_decomposition.md) through [T080](/Users/aeziz-local/Research/autonomous-agentic-research-swarm/.orchestrator/done/T080_release_candidate_assembly.md), the handoff notes, run/review manifests, validation reports, release manifest, and the actual paper source [reports/paper/index.qmd](/Users/aeziz-local/Research/autonomous-agentic-research-swarm/reports/paper/index.qmd). No files changed.

Main conclusion: the battle test successfully proved repo-native auditability, provenance, gating, and repairability. It did not prove full autonomy, strong planning, or scientist-grade analysis/writing. The framework is currently much stronger as a control plane than as an autonomous research strategist.

**What Happened**
- The repo was first rewritten into a v1 swarm through the local reviewed runner workflow under [docs/feedbacks/runner_task_packs/research_swarm_v1_rewrite/](/Users/aeziz-local/Research/autonomous-agentic-research-swarm/docs/feedbacks/runner_task_packs/research_swarm_v1_rewrite/), then battle-tested on the empirical L2-to-L1 rent project.
- The empirical queue ran roughly as `T025 -> T030 -> T035 -> T040 -> T045..T052 repairs -> T050 -> T060 -> T070 -> T080`.
- `T025` and `T030` exercised registry and vendor ETL successfully.
- `T035` became the real stress test: it ran for days, accumulated many blocked/active/run-manifest cycles, and effectively turned into a large exploratory repair campaign instead of a bounded worker task.
- `T045` to `T052` were a late decomposition of upstream scientific and attribution problems that should have been surfaced earlier.
- `T060`, `T070`, and `T080` completed the release path, but they were scoped as minimal release outputs, not as comprehensive analysis and paper-writing programs.

**Your 8 Issues**
1. Planner under-utilized: I agree strongly. In runtime terms, Planner is almost absent. In [scripts/swarm.py](/Users/aeziz-local/Research/autonomous-agentic-research-swarm/scripts/swarm.py), `--planner` only supports `heuristic`; it does not launch a Planner agent. The queue is selected heuristically, not replanned.
2. Operator loop/state machine: I agree strongly. The current `loop` only keeps calling `tick`; it does not automatically judge, merge, sweep, clean up, or escalate. Your local docs correctly describe the missing layer.
3. Planner should split tasks after selection: I agree strongly. `T035` is the clearest evidence. It should have been decomposed into at least attribution discovery, raw extraction/replay, enrichment/checkpointing, component auditability, and validation-facing rebuild tasks before a Worker was launched.
4. Prompts lack scientific rigor/integrity: I agree strongly. [docs/prompts/worker.md](/Users/aeziz-local/Research/autonomous-agentic-research-swarm/docs/prompts/worker.md), [planner.md](/Users/aeziz-local/Research/autonomous-agentic-research-swarm/docs/prompts/planner.md), and [judge.md](/Users/aeziz-local/Research/autonomous-agentic-research-swarm/docs/prompts/judge.md) focus on scope, paths, and manifests. They do not enforce evidence ranking, alternative-explanation checks, uncertainty discipline, claim-evidence mapping, or academic writing standards.
5. Worker needs a preliminary research phase: I agree. Right now the Worker prompt jumps straight to execution. In practice, `T035` did perform extensive preliminary investigation, but ad hoc inside the task, which made the task balloon.
6. Judge should be tighter and more comprehensive: I agree, with one constraint. Keep the current deterministic Judge core, but expand it. Today `judge-task` checks gates, outputs, manifests, and run bundles; it does not actually assess task success criteria or substantive output quality. Also, the Judge prompt exists, but the live Judge path is code-driven, not prompt-driven.
7. Analysis is too thin: I agree strongly. This is not just a prompt problem; it is a task-contract problem. [T060](/Users/aeziz-local/Research/autonomous-agentic-research-swarm/.orchestrator/done/T060_analysis_str_figures_and_tables.md) explicitly asked for a “minimum release analysis bundle,” and [src/analysis/build_str_release_outputs.py](/Users/aeziz-local/Research/autonomous-agentic-research-swarm/src/analysis/build_str_release_outputs.py) only emits two figures and one regime-summary table.
8. Paper is too thin and non-academic: I agree strongly. [reports/paper/index.qmd](/Users/aeziz-local/Research/autonomous-agentic-research-swarm/reports/paper/index.qmd) is release-note style, self-referential, authored by `autonomous-agentic-research-swarm`, and backed mostly by repo-local citations in [reports/paper/references.bib](/Users/aeziz-local/Research/autonomous-agentic-research-swarm/reports/paper/references.bib). It matches the current task scope, but not scientific paper standards.

**Additional Issues I Found**
- Planner and Judge are partly “paper roles” today. Their prompt files exist, but the runtime only actually invokes Worker and Operator prompts.
- The state machine is fragile for long-running tasks. `T035` and `T046` show false blocked/false ready states, and the full-autonomy handoff notes stale claimed-task detection.
- Manual and automatic execution are still awkwardly mixed. Several successful passes needed Operator backfill or repair of durable run manifests before review.
- Framework and project are not fully separated. Some framework behavior and prompts still encode the specific empirical battle-test DAG and artifact assumptions.
- Contract/doc drift is real. The repo hit stale handoff paths, legacy output references, and even a literal `YYYY-MM-DD` gate placeholder in `T080`.

**My View Of The Next Round**
1. Build a real Operator supervisor with explicit states and escalation rules. Your local Operator docs are the right direction.
2. Make Planner a required triage step for any task above a complexity threshold, not an optional repair role after failure.
3. Add a mandatory Worker reconnaissance note before implementation: scope understanding, risks, likely decomposition pressure, and proposed approach.
4. Expand Judge into deterministic-plus-substantive review, but keep it structured and bounded by task kind.
5. Redesign W6 and W7 as multi-task scientific programs. The current system is producing exactly what it asked for: a minimal release bundle, not a full scientific analysis and paper.

The deepest root cause is this: the swarm is already fairly good at controlled execution, but it still lacks a strong research-strategy layer. If the next round fixes only prompts, the outputs will improve a bit. If it fixes Planner, Operator, task decomposition, and scientific output contracts together, the framework will improve materially.

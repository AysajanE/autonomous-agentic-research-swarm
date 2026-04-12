# Research Swarm Post-Battle-Test Improvement Pack

This task pack is for the next improvement round of `autonomous-agentic-research-swarm` after the first full battle test.

The target is the framework in general, not a second pass on the `l2-l1-rent-analysis` instance. The empirical rent-analysis run is attached only as evidence of where the current swarm template held up and where it failed under real pressure.

## What This Pack Is For

Use this pack when the team wants the reviewed runner to design and finalize a drop-in improvement packet for the research-swarm framework itself, with explicit attention to future:

- empirical programs
- modeling programs
- hybrid programs

The improvement round is expected to preserve the strongest current components:

- repo-native shared-memory control plane
- contracts-first discipline
- deterministic offline gates
- worktree isolation
- durable run and review artifacts
- the split between the local swarm path and the reviewed high-stakes runner path

It is also expected to fix the architectural weaknesses exposed by the battle test:

- absent real Planner runtime behavior
- absent real Operator supervision loop
- weak decomposition pressure before large Worker tasks launch
- thin scientific rigor in prompts and task contracts
- thin substantive Judge review
- thin W6 analysis and W7 writing programs
- framework drift toward the single empirical reference instance

## Why This Pack Uses Three Stages

The task is large enough that one stage would collapse diagnosis, redesign, and packet authoring into one unstable pass, but small enough that more than three stages would mostly create review overhead.

The stages are:

1. `diagnosis_and_evidence_lock`
   Locks what actually failed, what must be preserved, and what must be improved across empirical, modeling, and hybrid work.
2. `target_architecture_and_change_contract`
   Converts the approved diagnosis into one exact target architecture and one exact file-level change contract.
3. `final_drop_in_packet`
   Emits the final implementation packet:
   - every new file as full contents
   - every changed file as a drop-in patch
   - every removal in an explicit remove ledger

Each stage has a distinct non-overlapping job:

- Stage 1 diagnoses and locks decisions. It does not lock the final file inventory or emit implementation files.
- Stage 2 locks the exact file inventory, per-file obligations, and red/green validation baseline. It does not emit the full packet.
- Stage 3 emits the full packet and may harden the approved draft, but it may not reopen approved architecture without explicit review authority.

## Manifest Curation Approach

The manifests are intentionally explicit and file-by-file.

- Stage 1 includes architecture, contracts, prompts, runtime scripts, modeling/hybrid contract surfaces, the April 12 operator retrospective, and the specific battle-test artifacts that reveal framework-wide failure modes.
- Stage 2 removes most historical battle-test evidence and narrows to the files most likely to change in the framework improvement packet.
- Stage 2 demotes `contracts/project.yaml` to reference-only context so the architecture contract does not drift back into the current empirical instance unless the approved diagnosis explicitly requires that file.
- Stage 3 keeps the likely final change targets and the minimum reference surfaces needed to harden the approved draft without reopening the design.

No stage uses broad repo-directory attachments for high-noise areas such as raw data, processed data, build outputs, or the full status-artifact tree.

## Tool Policy

- Stage 1: `web_search`
  Use only for current external benchmarks, current docs, and primary-source confirmation that could materially improve the architecture.
- Stage 2: `no_tools`
- Stage 3: `no_tools`

The final packet should be grounded in the approved internal architecture, not in fresh browsing late in the run.

## Pack Layout

- `shared_instructions.md`
- `corpus/`
- `prompts/`
- `inputs/`
- `tools/`
- `workflows/three_stage.workflow.json`

## Recommended Commands

Dry run stage 1:

```bash
cd "/Users/aeziz-local/Research/autonomous-agentic-research-swarm"

python3 "/Users/aeziz-local/staged-workflow-runner/automation/run_responses_v2.py" run \
  --root . \
  --workflow-file docs/feedbacks/runner_task_packs/research_swarm_post_battletest_improvement/workflows/three_stage.workflow.json \
  --dry-run
```

Run stage 1 live and wait:

```bash
cd "/Users/aeziz-local/Research/autonomous-agentic-research-swarm"

python3 "/Users/aeziz-local/staged-workflow-runner/automation/run_responses_v2.py" run \
  --root . \
  --workflow-file docs/feedbacks/runner_task_packs/research_swarm_post_battletest_improvement/workflows/three_stage.workflow.json \
  --skip-token-count \
  --wait
```

Create the approved stage-1 review bundle:

```bash
cd "/Users/aeziz-local/Research/autonomous-agentic-research-swarm"

python3 "/Users/aeziz-local/staged-workflow-runner/automation/create_review_bundle_v2.py" \
  --root . \
  --output <run_dir>/stage1.review_bundle.json \
  --workflow-id research_swarm_post_battletest_improvement \
  --source-stage-id diagnosis_and_evidence_lock \
  --source-run-id <run_id> \
  --primary-artifact-markdown <run_dir>/stages/01_diagnosis_and_evidence_lock/response.final.md \
  --response-artifact-json <run_dir>/stages/01_diagnosis_and_evidence_lock/response.final.json \
  --reviewer-notes <run_dir>/stage1.review.md
```

Continue with stage 2:

```bash
cd "/Users/aeziz-local/Research/autonomous-agentic-research-swarm"

python3 "/Users/aeziz-local/staged-workflow-runner/automation/run_responses_v2.py" run \
  --root . \
  --workflow-file docs/feedbacks/runner_task_packs/research_swarm_post_battletest_improvement/workflows/three_stage.workflow.json \
  --run-dir <run_dir> \
  --review-bundle <run_dir>/stage1.review_bundle.json \
  --skip-token-count \
  --wait
```

Repeat the same pattern for the stage-2 review bundle and then continue to stage 3.

## Operational Notes

- Keep stage review manual and explicit.
- Use `--skip-token-count` for live runs unless token preflight is known to be reliable again.
- Stage 1 may consult external sources, but Stages 2 and 3 should stay fully local.
- The final stage output is the packet the team should be able to apply directly without reinterpretation.
- No sidecar output is required for this pack.

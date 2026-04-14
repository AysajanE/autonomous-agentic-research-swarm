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
- Stage 2 removes most historical battle-test evidence and narrows to the direct architecture-control surfaces that should drive the change contract.
- Stage 2 keeps the direct framework control surfaces in `attached_repository_files`, while demoting worked examples, scientific-definition surfaces, supporting mode examples, and current-instance evidence to `reference_context` so stage 2 preserves completeness without letting secondary examples outrank the core framework contract.
- Stage 2 demotes `contracts/project.yaml` to reference-only context so the architecture contract does not drift back into the current empirical instance unless the approved diagnosis explicitly requires that file.
- Stage 3 preserves the same attached-authority boundary as stage 2, keeps low-authority coherence surfaces in `reference_context`, and does not re-elevate previously demoted files unless the stage-two review explicitly reopened them.

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
- `review_templates/`
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

Create the required stage-1 review bundle:

```bash
cd "/Users/aeziz-local/Research/autonomous-agentic-research-swarm"

cp \
  docs/feedbacks/runner_task_packs/research_swarm_post_battletest_improvement/review_templates/stage1_to_stage2_handoff_template.md \
  <run_dir>/stage1.stage2_handoff.md

# Then replace the template placeholders with the approved stage-1-to-stage-2 handoff.

python3 "/Users/aeziz-local/staged-workflow-runner/automation/create_review_bundle_v2.py" \
  --root . \
  --output <run_dir>/stage1.review_bundle.json \
  --workflow-id research_swarm_post_battletest_improvement \
  --source-stage-id diagnosis_and_evidence_lock \
  --source-run-id <run_id> \
  --primary-artifact-markdown <run_dir>/stages/01_diagnosis_and_evidence_lock/response.final.md \
  --response-artifact-json <run_dir>/stages/01_diagnosis_and_evidence_lock/response.final.json \
  --approved-handoff-markdown <run_dir>/stage1.stage2_handoff.md \
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

Create the required stage-2 review bundle:

```bash
cd "/Users/aeziz-local/Research/autonomous-agentic-research-swarm"

cp \
  docs/feedbacks/runner_task_packs/research_swarm_post_battletest_improvement/review_templates/stage2_to_stage3_handoff_template.md \
  <run_dir>/stage2.stage3_handoff.md

# Then replace the template placeholders with the approved stage-2-to-stage-3 handoff.

python3 "/Users/aeziz-local/staged-workflow-runner/automation/create_review_bundle_v2.py" \
  --root . \
  --output <run_dir>/stage2.review_bundle.json \
  --workflow-id research_swarm_post_battletest_improvement \
  --source-stage-id target_architecture_and_change_contract \
  --source-run-id <run_id> \
  --primary-artifact-markdown <run_dir>/stages/02_target_architecture_and_change_contract/response.final.md \
  --response-artifact-json <run_dir>/stages/02_target_architecture_and_change_contract/response.final.json \
  --approved-handoff-markdown <run_dir>/stage2.stage3_handoff.md \
  --reviewer-notes <run_dir>/stage2.review.md
```

Continue with stage 3:

```bash
cd "/Users/aeziz-local/Research/autonomous-agentic-research-swarm"

python3 "/Users/aeziz-local/staged-workflow-runner/automation/run_responses_v2.py" run \
  --root . \
  --workflow-file docs/feedbacks/runner_task_packs/research_swarm_post_battletest_improvement/workflows/three_stage.workflow.json \
  --run-dir <run_dir> \
  --review-bundle <run_dir>/stage2.review_bundle.json \
  --skip-token-count \
  --wait
```

## Operational Notes

- Keep stage review manual and explicit.
- Use `--skip-token-count` for live runs unless token preflight is known to be reliable again.
- Stage 1 may consult external sources, but Stages 2 and 3 should stay fully local.
- When preparing the stage-1 or stage-2 review bundle, include the concise reviewed handoff markdown. Do not launch stage 2 or stage 3 on raw prior-stage artifacts alone when a reviewed handoff is required.
- Stage 2 is intentionally authority-shaped: direct framework control surfaces stay attached at higher priority, while supporting examples and current-instance evidence remain available as lower-priority reference context.
- Stage-2 review should reject any approved inventory that cannot fit the current stage-3 packet budget. In this runner, `gpt-5.4-pro` is capped at `128000` `max_output_tokens`.
- In stages 2 and 3, validation tables are specified later checks, not claims that the checks were executed during packet design.
- The final stage output is the packet the team should be able to apply directly without reinterpretation.
- No sidecar output is required for this pack.

Produce the stage-two target architecture and exact change contract for the next `autonomous-agentic-research-swarm` improvement round.

This is the architecture-and-change-contract stage.

Its job is to convert the approved stage-one diagnosis into:

- one exact target architecture
- one explicit decision-to-architecture derivation chain
- one exact file inventory
- one exact per-file implementation contract
- one exact red/green validation baseline

This is not the final packet stage.

Authority and derivation rules:

- Use the approved stage-one review bundle and reviewer notes as the controlling authority above the attached repo files.
- The approved stage-one review bundle must include concise downstream handoff markdown for stage 2. Treat that handoff as the primary reviewed synthesis and do not compensate for its absence by widening authority or re-deriving architecture from raw artifacts alone.
- When the approved downstream handoff, the raw stage-one artifact, and the reviewer notes differ, treat the reviewer notes and locked decisions as controlling.
- Use the raw stage-one artifact as detail and evidence only where the approved handoff and reviewer notes do not already settle the downstream requirement.
- Use attached repo files to keep the contract minimum-change, repo-fit, and executable.
- Use reference-context files for supporting examples, cross-mode guardrails, and instance-awareness only. Do not let reference-only files expand the exact file inventory unless a locked decision requires that expansion.
- Do not reopen approved diagnosis decisions unless the reviewer explicitly reopened them.

Non-negotiable rules:

- Lock the exact change contract. Do not output broad architectural advice.
- Keep the redesign general across empirical, modeling, and hybrid work.
- Do not treat the excluded ad-hoc `.local` operator workflow notes as approved architecture inputs even if they appear cited in the carried-forward stage-one artifact.
- Treat any external benchmark signals carried forward from stage 1 as frozen, reviewed summaries. Use them as design hints only; do not browse, and do not invent external details that were not explicitly approved in the stage-one handoff.
- Do not default to rewriting current-instance surfaces such as `contracts/project.yaml`; only include them in the exact file inventory when an approved stage-one framework decision requires that update.
- If a file contract changes schema, prompt semantics, runtime behavior, or review semantics, the exact inventory must include the corresponding enforcement surface(s) and at least one validating test or validation row unless reviewer notes explicitly allow a docs-only clarification.
- If a file is not in the stage-two inventory, stage 3 must not touch it unless the stage-two review explicitly reopens the file set.
- The exact inventory must remain small enough that stage 3 can emit the full packet within the configured final-stage output budget.
- Do not browse. This stage is intentionally local and should rely on the approved stage-one handoff plus the attached repo files.
- Stage 2 may emit complete final contents only for the smallest necessary subset of boundary-locking files whose exact wording must be fixed now to keep stage 3 from drifting.
- Do not emit patch blocks, diff hunks, or a near-final implementation packet in this stage.
- Stage 2 must not emit the full final package.
- The validation matrix in this stage specifies the approved red/green baseline to run later. Do not claim that any check was executed in this stage.
- The final packet mode is locked:
  - `create` files end as full file contents in stage 3
  - `update` files end as drop-in patches in stage 3
  - `remove` paths end in the remove ledger in stage 3

Return these sections in this exact order:

## 1. Locked Architecture Summary

State:

- the approved target architecture in one paragraph
- the minimum-change thesis in one paragraph
- how the redesign stays general across empirical, modeling, and hybrid work in one paragraph
- the exact stage-2/stage-3 boundary in one paragraph

## 2. Decision-To-Architecture Derivation

Use this exact table:

| decision_id | locked_decision | architectural_consequence | repo_surfaces_required | validation_consequence |

Rules:

- `decision_id` must be a stable short id such as `D01`, `D02`, and so on.
- Normalize the approved handoff into the smallest stable set of decision IDs. Merge overlapping wording variants instead of proliferating near-duplicate rows.
- Every row must come from the approved stage-one handoff, reviewer notes, or a directly necessary repo-grounded architectural consequence of those approved decisions.
- `repo_surfaces_required` must name exact repo-relative paths or clearly delimited path groups.
- `validation_consequence` must explain what must become testable because of that decision.
- Every `decision_id` in section 2 must map to at least one inventory row and at least one validation consequence, unless the row explicitly states that it is a preserved-no-change guardrail.
- This section is the controlling derivation chain for the rest of the stage-two output.

## 3. Exact File Inventory

Use this exact table:

| path | action | category | final_packet_mode | derived_from_decision_ids | purpose |

Rules:

- `path` must be an exact repo-relative path.
- `action` must be `create`, `update`, or `remove`.
- `category` must be one of:
  - `control_plane`
  - `contract`
  - `docs`
  - `runtime`
  - `tests`
- `final_packet_mode` must be one of:
  - `full_new_file`
  - `drop_in_patch`
  - `remove_ledger`
- `derived_from_decision_ids` must cite the exact `decision_id` values from section 2 that require this file to be touched.
- Include only the files or paths that stage 3 is allowed to touch.

## 4. File Implementation Contracts

Use this exact table:

| path | derived_from_decision_ids | required_behavior | must_include | dependencies_or_interfaces | stage3_completion_rule |

Rules:

- Include one row for every row in the exact file inventory.
- The inventory and this table must match exactly.
- `derived_from_decision_ids` must match the file’s justification in the exact file inventory.
- `must_include` should name the exact sections, CLI behavior, schema fields, tests, or contract terms the final file must contain.
- `stage3_completion_rule` must make the stage-3 obligation unambiguous.

## 5. Boundary-Locking Exact Files

For every early-locked file, use this exact structure:

### File: `<repo-relative path>`

- action: `create` or `update`
- derived_from_decision_ids: `<comma-separated ids>`
- why required now: `<brief rationale>`
- stage-3 rule: `<what stage 3 must preserve or may tighten>`

````<language>
<complete final file contents>
````

Rules:

- Prefer `None.` unless exact full text must already be locked to prevent stage-3 drift.
- Prefer early-locking only short markdown or contract files whose wording itself is the contract. Avoid early-locking large updated runtime or test files unless exact wording must be frozen now.
- Emit complete final contents only for the smallest necessary subset of files whose exact wording must already be locked.
- Every file in this section must also appear in the inventory and in the file implementation contracts.
- If no early-locked file is required, write `None.` and ensure section 1 explains why.

## 6. Validation And Test Matrix

Use this exact table:

| phase | check_id | derived_from_decision_ids | command_or_method | expected_result | why_it_matters |

Rules:

- `phase` must be `red` or `green`.
- `derived_from_decision_ids` must cite the exact section-2 decisions the check is proving.
- Order rows so each `red` check appears before its corresponding `green` check.
- A `red` row must fail against the pre-change state and exercise new or changed behavior.
- A `green` row must pass after the final packet is applied.
- The validation matrix specifies the approved baseline to run later. Do not claim that any check was executed in this stage.
- This matrix becomes the controlling validation baseline for stage 3 unless the stage-two review explicitly reopens it.

## 7. Reviewer Focus

Use this exact table:

| focus_area | why_high_risk | what_to_audit |

## 8. Final Stage Charter

Format this section exactly as:

- `Preserve:`
  Then a flat bullet list.
- `Tighten first:`
  Then a flat bullet list.
- `Do not reopen:`
  Then a flat bullet list.

Quality bar:

- This should read like a real architecture-backed implementation contract.
- Do not omit derivation or validation.
- Do not collapse back into diagnosis or forward into the full final packet.
- Every inventory row and every validation row must trace cleanly to section 2.
- Keep stage 3 valuable by leaving it real hardening work inside the approved file set:
  - full packet emission
  - patch drafting
  - within-file hardening and patch-hunk pruning
  - consistency repair
  - acceptance tightening

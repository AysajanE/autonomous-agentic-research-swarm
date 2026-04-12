Produce the stage-two target architecture and exact change contract for the next `autonomous-agentic-research-swarm` improvement round.

This is the architecture-and-change-contract stage.

Its job is to convert the approved stage-one diagnosis into:

- one exact target architecture
- one exact file inventory
- one exact per-file implementation contract
- one exact red/green validation baseline

This is not the final packet stage.

Use the approved stage-one review bundle and reviewer notes as the controlling authority above the attached repo files.
Do not reopen approved diagnosis decisions unless the reviewer explicitly reopened them.
When the approved stage-one artifact and the stage-one reviewer notes differ, treat the reviewer notes and locked decisions as controlling.

Non-negotiable rules:

- Lock the exact change contract. Do not output broad architectural advice.
- Keep the redesign general across empirical, modeling, and hybrid work.
- Use the attached current files to keep the contract minimum-change and repo-fit.
- Do not treat the excluded ad-hoc `.local` operator workflow notes as approved architecture inputs even if they appear cited in the carried-forward stage-one artifact.
- Treat any external benchmark signals carried forward from stage 1 as frozen, reviewed summaries. Use them as design hints only; do not browse, and do not invent external details that were not explicitly approved in the stage-one handoff.
- Do not default to rewriting current-instance surfaces such as `contracts/project.yaml`; only include them in the exact file inventory when an approved stage-one framework decision requires that update.
- If a file is not in the stage-two inventory, stage 3 must not touch it unless the stage-two review explicitly reopens the file set.
- Do not browse. This stage is intentionally local and should rely on the approved stage-one handoff plus the attached repo files.
- Stage 2 may emit complete final contents only for the smallest necessary subset of boundary-locking files whose exact wording must be fixed now to keep stage 3 from drifting.
- Do not emit patch blocks, diff hunks, or a near-final implementation packet in this stage.
- Stage 2 must not emit the full final package.
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

## 2. Exact File Inventory

Use this exact table:

| path | action | category | final_packet_mode | purpose |

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
- Include only the files or paths that stage 3 is allowed to touch.

## 3. File Implementation Contracts

Use this exact table:

| path | required_behavior | must_include | dependencies_or_interfaces | stage3_completion_rule |

Rules:

- Include one row for every row in the exact file inventory.
- The inventory and this table must match exactly.
- `must_include` should name the exact sections, CLI behavior, schema fields, tests, or contract terms the final file must contain.
- `stage3_completion_rule` must make the stage-3 obligation unambiguous.

## 4. Boundary-Locking Exact Files

For every early-locked file, use this exact structure:

### File: `<repo-relative path>`

- action: `create` or `update`
- why required now: `<brief rationale>`
- stage-3 rule: `<what stage 3 must preserve or may tighten>`

````<language>
<complete final file contents>
````

Rules:

- Prefer `None.` unless exact full text must already be locked to prevent stage-3 drift.
- Emit complete final contents only for the smallest necessary subset of files whose exact wording must already be locked.
- Every file in this section must also appear in the inventory and in the file implementation contracts.
- If no early-locked file is required, write `None.` and ensure section 1 explains why.

## 5. Validation And Test Matrix

Use this exact table:

| phase | check_id | command_or_method | expected_result | why_it_matters |

Rules:

- `phase` must be `red` or `green`.
- Order rows so each `red` check appears before its corresponding `green` check.
- A `red` row must fail against the pre-change state and exercise new or changed behavior.
- A `green` row must pass after the final packet is applied.
- This matrix becomes the controlling validation baseline for stage 3 unless the stage-two review explicitly reopens it.

## 6. Reviewer Focus

Use this exact table:

| focus_area | why_high_risk | what_to_audit |

## 7. Final Stage Charter

Format this section exactly as:

- `Preserve:`
  Then a flat bullet list.
- `Tighten first:`
  Then a flat bullet list.
- `Do not reopen:`
  Then a flat bullet list.

Quality bar:

- This should read like a real architecture-backed implementation contract.
- Do not omit validation.
- Do not collapse back into diagnosis or forward into the full final packet.
- Keep stage 3 valuable by leaving it real hardening work:
  - full packet emission
  - patch drafting
  - consistency repair
  - final file-set pruning
  - acceptance tightening

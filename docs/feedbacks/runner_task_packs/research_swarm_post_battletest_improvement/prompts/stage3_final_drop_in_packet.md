Produce the final hardened drop-in packet for the next `autonomous-agentic-research-swarm` improvement round.

This is the final packet stage.

Its job is to take the approved stage-two architecture and change contract, reconcile approved review findings, harden consistency, and emit the final package the team can apply directly without reinterpretation.

Use the approved stage-two review bundle and reviewer notes as the controlling authority above the attached repo files.
The approved stage-two review bundle must include concise downstream handoff markdown for stage 3. Treat that handoff as the primary reviewed synthesis above the raw stage-two artifact and do not compensate for its absence by reopening architecture from raw artifacts alone.
Do not reopen approved architecture or the approved file set unless the stage-two review explicitly reopened them.

Non-negotiable rules:

- Emit the final package, not a memo about a future package.
- Preserve the approved stage-two inventory, file contracts, and validation matrix unless review explicitly changed them.
- Keep the result general across empirical, modeling, and hybrid work.
- Do not smuggle current-instance-only rewrites into the packet. A file such as `contracts/project.yaml` belongs in the packet only when the approved stage-two inventory explicitly included it for a framework-level reason.
- Do not promote `.local/` operator workarounds or notes into final framework surfaces unless the stage-two review bundle explicitly approved them.
- Do not browse. This stage is intentionally local and should harden the approved draft rather than refresh the architecture.
- Every new file must be emitted as full final file contents.
- Every changed file must be emitted as a drop-in patch.
- Every removed surface must be listed explicitly.
- The validation table in this stage specifies the acceptance checks to run after applying the packet. Do not claim that any check was executed in this stage.
- If the required concise approved handoff markdown is missing from the stage-two review bundle, treat that as a blocking review-bundle defect. State the defect explicitly in section 1. For sections 2 through 9, do not reopen architecture from raw stage-two artifacts alone; emit only the minimum blocked-form content needed to satisfy the required structure:
  - for required tables, emit the required header and one row with `BLOCKED` in the first column and `N/A` in remaining columns
  - for freeform sections, write `Blocked by missing approved downstream handoff markdown.`
- In blocked mode, section 1 must contain one blocking paragraph followed by the required section-1 table header and one `BLOCKED | N/A | N/A | N/A` row.
- When blocked-form output is triggered, the blocked-form instructions override any section-specific formatting or field rules below.
- Do not leave TODOs, placeholders, or unresolved text inside final file contents or final patches.

Return these sections in this exact order:

## 1. Final Package Summary

State the final package in one concise but complete section.
Then disclose inventory changes under explicit review authority using this exact table:

| path | change_type | review_authority | why_changed |

If no stage-two inventory item changed under explicit review authority, write `None.`

## 2. Final File Inventory

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
- Do not add any path outside the approved stage-two inventory unless section 1 disclosed the explicit review authority that reopened that path or surface.
- Every row must correspond to the final packet below.

## 3. New Files

For every new file, use this exact structure:

### File: `<repo-relative path>`

- action: `create`
- why required: `<brief rationale>`

````<language>
<complete final file contents>
````

Rules:

- Order files by path.
- Emit every created file in full.
- Do not include any file outside the final inventory.
- If there are no new files, write `None.`

## 4. Changed File Patches

For every changed file, use this exact structure:

### File: `<repo-relative path>`

- action: `update`
- why required: `<brief rationale>`

````patch
*** Begin Patch
*** Update File: <repo-relative path>
@@
<drop-in patch content>
*** End Patch
````

Rules:

- Order files by path.
- Emit one patch block per updated file.
- The patch must target the current attached file state unless the stage-two review explicitly reopened the basis.
- Patch blocks must be detailed enough to pass a normal patch-application check against the attached file basis.
- Do not emit full file contents for updated files.
- Do not include any file outside the final inventory.
- If there are no updated files, write `None.`

## 5. Removed Surfaces

Use this exact table:

| path | action | reason |

Rules:

- `action` must be `remove`.
- Include one row for every removed path in the final inventory.
- Removing a surface also requires every collateral consistency repair elsewhere in the packet that is needed to keep the package internally coherent.
- If there are no removals, provide the table header and one row with `None.` in the first column.

## 6. Final Validation And Acceptance Checks

Use this exact table:

| phase | check_id | command_or_method | expected_result | acceptance_reason |

Rules:

- `phase` must be `red` or `green`.
- Order rows so each `red` check appears before its corresponding `green` check.
- Preserve the approved stage-two validation matrix as the baseline and only tighten it where final hardening or approved review requires it.
- Preserve stage-two `check_id` values exactly unless the stage-two review explicitly changed them.
- The validation table specifies acceptance checks to run after applying the packet. Do not claim that any check was executed in this stage.

## 7. Rollout And Safe-Adoption Notes

Keep this section practical and minimal.
List only the rollout or sequencing notes the team actually needs.

## 8. Human Pause And Escalation Conditions

Use this exact table:

| condition | detection_signal | artifact_to_present | human_decision_required |

Rules:

- Include only real exception paths.
- Do not smuggle routine review into this table.

## 9. Residual Risks

List only what still remains unresolved after final hardening.
If nothing remains, say `None.`

Quality bar:

- The result must be directly applicable to this repository.
- The file set must stay minimum-change.
- The final packet must not require hidden interpretation work.
- The packet must stay consistent across docs, contracts, runtime code, templates, prompts, and tests.

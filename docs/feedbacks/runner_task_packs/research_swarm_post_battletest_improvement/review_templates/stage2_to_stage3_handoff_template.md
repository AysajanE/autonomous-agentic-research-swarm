# Stage 2 To Stage 3 Approved Handoff

Use this required file to give stage 3 a compact, approved final-packet brief ahead of the full stage-2 artifact.

Keep it short, specific, and review-grade. Preserve only what stage 3 must treat as locked, reopened under review authority, or explicitly excluded.

## 1. Final Packet Boundary

- Approved final package goal:
- Exact inventory preserved as approved:
- Allowed within-file hardening scope:
- Packet-budget fit statement:
- Touched-file-count fit:
- Large full-file emission risk:
- Large patch concentration risk:
- Early-locked exact files remain minimal:

## 2. Approved Exact Inventory

Use this exact table:

| path | action | final_packet_mode | downstream rule |

## 3. Review-Authorized Inventory Changes Or Reopenings

Use this exact table:

| path_or_scope | change_type | review_authority | downstream consequence |

If there are no review-authorized inventory changes or reopenings, write `None.`

## 4. Approved Validation Baseline

Use this exact table:

| check_scope_or_prior_id | keep_change_or_reopen | approved_check_id_or_rule | downstream rule |

When `keep_change_or_reopen = change`, `approved_check_id_or_rule` must name the newly approved stage-3 `check_id`, and `downstream rule` must note the prior identifier or scope being changed.

## 5. Early-Locked Exact Files To Preserve

Use this exact table:

| path | preserve_or_tighten | downstream rule |

If there are no early-locked exact files, write `None.`

## 6. Explicit Exclusions And Downstream Limits

- Excluded from downstream authority:
- Reference-only unless explicitly reopened:
- `.local` surfaces excluded unless explicitly approved:

## 7. Reviewer Cautions For Stage 3

- Audit first:
- Highest drift risk:
- Quality bar for approval:

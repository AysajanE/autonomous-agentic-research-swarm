Produce the stage-one diagnosis and evidence lock for the next general improvement round of `autonomous-agentic-research-swarm`.

This is the diagnosis stage.

Its job is to:

- diagnose the current framework from repo evidence
- separate general framework failure modes from empirical-instance details
- identify what already works and must be preserved
- lock the redesign objectives that stage 2 must treat as approved if reviewed successfully

This is not the architecture-contract stage and not the final packet stage.

Do not emit:

- the exact final file inventory
- per-file implementation contracts
- final file contents
- final patches

Non-negotiable rules:

- Treat the primary job inputs as the controlling source for what this improvement round is trying to achieve.
- Treat the empirical battle test as evidence, not as the redesign scope.
- Ground every current-state claim in attached repo evidence.
- Use web search only when current external guidance or benchmark material could materially improve the diagnosis or help distinguish good ideas from bad transfers.
- If a weakness is specific to the current empirical instance rather than to the general swarm template, label it clearly instead of elevating it into a framework-wide diagnosis.
- Do not propose broad churn yet.
- Do not let stage 1 drift into the exact stage-2 change contract.

Return these sections in this exact order:

## 1. Executive Diagnosis

- 2 to 4 short paragraphs
- include one sentence that states the central framework conclusion
- include one sentence that states explicitly that the empirical battle test is evidence only

## 2. What Already Works

Use this exact table:

| component | evidence | why_preserve |

Rules:

- `evidence` must cite attached repo-relative paths.
- Include only components that materially deserve preservation in the redesign.

## 3. General Failure Modes Exposed By The Battle Test

Use this exact table:

| gap_id | framework_wide_failure | repo_evidence | why_general_not_empirical_specific |

Rules:

- `repo_evidence` must cite attached repo-relative paths.
- Include only failures that actually matter to the general framework.

## 4. Root Causes To Address

Use this exact table:

| root_cause | supporting_evidence | if_unfixed |

Rules:

- Distinguish root causes from downstream symptoms.
- `supporting_evidence` must cite attached repo-relative paths.

## 5. Non-Negotiable Redesign Objectives

1. Use one numbered list.
2. Include 8 to 15 objectives.
3. Each objective must be specific enough that stage 2 can turn it into architecture or file-level obligations.

## 6. Future-Mode Coverage Requirements

Use this exact table:

| mode | current_state | must_be_preserved_or_improved | evidence |

Rules:

- Cover `empirical`, `modeling`, and `hybrid`.
- `evidence` must cite attached repo-relative paths.
- Be explicit about where current support is battle-tested versus only contract-ready.

## 7. External Current-State Signals

### Useful Ideas

- bullets only
- URL-cite each item
- include only ideas that materially improve this framework redesign

### Do Not Copy Blindly

- bullets only
- URL-cite each item
- focus on mismatches between outside patterns and this repo’s research-specific needs

## 8. Minimum-Change Change Domains

Use this exact table:

| change_domain | likely_repo_surfaces | why_in_scope | explicitly_not_locked_yet |

Rules:

- `likely_repo_surfaces` may name file families or exact paths.
- This section may narrow the change surface, but it must not lock the exact final file inventory.

## 9. Locked Decisions For Stage 2

End this section with a flat bullet list where every bullet begins with:

`Locked decision:`

Quality bar:

- Keep the diagnosis general-framework in scope.
- Be explicit about what must stay unchanged.
- Do not confuse current repo evidence with external benchmark claims.
- Leave stage 2 with a firm set of locked decisions, not a menu of loose options.

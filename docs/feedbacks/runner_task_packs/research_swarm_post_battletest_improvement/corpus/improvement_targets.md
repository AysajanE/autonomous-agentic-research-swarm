# Improvement Targets

The battle test established that the current swarm is stronger as a controlled execution plane than as an autonomous research strategist.

This improvement round should address the following framework-wide targets.

## Preserve

- repo-native auditability
- provenance and release lineage
- deterministic gating
- repairability
- explicit state and review semantics

## Improve

1. Planner must become a real runtime capability rather than a mostly paper role.
2. Planner must force decomposition before oversized Worker tasks launch.
3. Operator must become a real supervisory loop with explicit state handling, escalation rules, and cleanup discipline.
4. Worker tasks should begin with a bounded reconnaissance phase instead of jumping straight into execution.
5. Judge should remain deterministic at the core but expand to substantive task-success review where the task kind requires it.
6. Prompt and task-contract surfaces should enforce scientific rigor more explicitly:
   - evidence ranking
   - uncertainty discipline
   - alternative-explanation checks
   - claim-to-evidence alignment
7. Analysis and writing should be treated as real scientific programs, not as thin release wrappers.
8. The framework should stay general across empirical, modeling, and hybrid work rather than drifting toward the current empirical reference instance.
9. Contract and documentation drift should be harder to create and easier to detect.
10. Each approved improvement must become enforceable through the smallest coherent combination of contract, prompt, runtime, and test changes.

## Generality Requirement

The redesign must remain valid for future work that is:

- empirical only
- modeling only
- hybrid across empirical and modeling layers

The current empirical battle test is evidence, not the redesign boundary.

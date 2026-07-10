# Golden regression tier

This package is the CI-visible golden tier described in plan §9.3. It is the optimization target for prompt and contract changes: changes should preserve these named end-to-end behaviors. A held-out adversarial tier will be added at M5 and is intentionally not represented here.

Scenarios use stable `G01_...` identifiers. To add one, choose the next identifier, add one `test_GNN_...` method to the appropriate golden module, construct a hermetic repository with `GoldenRepo`, and document the behavior in the scenario name and assertions.

Golden scenarios exercise end-to-end command flows such as `cmd_tick`, `cmd_run_task`, `cmd_judge_task`, and deterministic gates. They must not substitute assertions against runtime internals for those command flows.

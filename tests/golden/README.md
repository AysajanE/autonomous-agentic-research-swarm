# Golden regression tier

This package is the CI-visible golden tier described in plan §9.3. It is the optimization target for prompt and contract changes: changes should preserve these named end-to-end behaviors. The M5 [`tests/held_out/`](../held_out/) adversarial tier is intentionally separate and runs only at milestone gates.

Scenarios use stable `G01_...` identifiers. To add one, choose the next identifier, add one `test_GNN_...` method to the appropriate golden module, construct a hermetic repository with `GoldenRepo`, and document the behavior in the scenario name and assertions.

Golden scenarios exercise end-to-end command flows such as `cmd_tick`, `cmd_run_task`, `cmd_judge_task`, and deterministic gates. They must not substitute assertions against runtime internals for those command flows.

## Chaos & reconciliation (tier-b)

`scripts/swarm_chaos.py` injects supervisor and worker SIGKILL failures into a prepared synthetic queue, then checks orphan recovery, serial merges, journal integrity, and spend-ledger reconciliation. Run a longer manual soak with `python scripts/swarm_chaos.py --repo <fixture-repo> --cycles 50 --kill-supervisor-at-cycle 20 --kill-worker --json`; bounded CI coverage lives in `tests/test_m1_chaos_reconcile.py`.

# Held-out milestone evaluation

This package is the M5 Goodhart control (plan §9.3). It is distinct from the
CI-visible `tests/golden/` optimization target: its cases live in `cases.py`,
whose name does not match default `test*.py` discovery, so prompt/contract
tuning never sees them. Run it only with `make eval-heldout` at milestone gates.

**Distinctness contract (verified, not aspirational).** Every case in `cases.py`
is constructed *independently* — it imports NONE of the CI-visible test modules
(no `golden.test_golden_*`, `test_m3a_modeling_battery`, `test_m3b_referee`, or
`test_m4c_replication`), building its fixtures only from the generic
`GoldenRepo`/`runtime_test_utils` harness (neither is discovered by `make test`)
plus an inline attack. And every case asserts a failure mode (gate + reason) that
is — verified by grep over `tests/**/test_*.py` — exercised by **neither** the
`make test` suite **nor** the seeded-defect drill rotation:

| Case | Gate | Distinct reason |
|---|---|---|
| wrong-monotonicity comparative static | `theoretical_falsification` | `comparative_static_violated` (drill uses `inequality_violated`) |
| wrong-signed explicit derivative | `theoretical_falsification` | `comparative_static_violated` (explicit-derivative branch) |
| future-dated citation evidence | `citation_integrity` | `citation_snapshot_directory_after_as_of` |
| schema-violating bridge instance | `instance_manifest_conformance` | `instance_manifest_schema_violation` |
| citation snapshot key forgery | `citation_integrity` | `citation_snapshot_key_mismatch` (drill uses unresolved/retraction/url) |

Because a regression these cases detect does not already fail the suite that
produced the optimiser's signal, a prompt/contract change over-fit to the
visible suite is still caught here.

**Not faked.** Live-referee held-out judgement cannot be produced offline
without prescribing the verdict into a mock (which would test plumbing, not
detection), so it is deliberately **not** included here; it is a
tier-c/live-calibration concern (on-demand/BT2), flagged rather than simulated.

The owner must adversarially refresh this tier at every milestone with fresh
cases aimed at the mechanisms then in use, preserving the distinctness contract
above. Do not copy held-out cases into per-PR prompt tuning or the visible
golden suite.

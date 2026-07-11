# tests/

Unit tests and small deterministic checks (should run on data in `data/samples/` only).

- Keep tests fast.
- No network calls.
- Prefer testing identities/invariants (accounting checks, schema checks, etc.).

## Golden tier

The CI-visible golden tier in [`tests/golden/`](golden/) protects named end-to-end command behaviors and is the optimization target for prompt and contract changes. See [`tests/golden/README.md`](golden/README.md) for scenario conventions and contribution rules. The distinct [`tests/held_out/`](held_out/) tier is the milestone-only Goodhart control and is intentionally excluded from default discovery.

# tests/

Unit tests and small deterministic checks (should run on data in `data/samples/` only).

- Keep tests fast.
- No network calls.
- Prefer testing identities/invariants (accounting checks, schema checks, etc.).

## Golden tier

The CI-visible golden tier in [`tests/golden/`](golden/) protects named end-to-end command behaviors and is the optimization target for prompt and contract changes. See [`tests/golden/README.md`](golden/README.md) for scenario conventions and contribution rules; a separate held-out adversarial tier arrives at M5.

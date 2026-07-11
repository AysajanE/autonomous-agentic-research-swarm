# Held-out milestone evaluation

This package is the M5 Goodhart control. It is distinct from the CI-visible
`tests/golden/` optimization target: its cases live in `cases.py`, whose name
does not match default `test*.py` discovery. Run it only with
`make eval-heldout` at milestone gates.

The five M5 cases target the current computed-paper, referee,
program-conformance, replication-bridge, and compliance mechanisms. The owner
must adversarially refresh this tier at every milestone with fresh artifacts
and tasks aimed at the mechanisms then in use. Do not copy held-out cases into
per-PR prompt tuning or the visible golden suite.

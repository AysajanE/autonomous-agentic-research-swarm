---
schema_version: research_swarm.prereg_lock.v1
phase: lock_b
status: draft
locked_at_utc: null
locked_sha256: null
locked_by: null
lock_version: 0
---

# Lock B — concrete instance-set binding

After bridge generation, list every concrete instance manifest exactly once and
replace `pending` with its SHA-256 before activating this lock. A binding line
has the two-line form shown below.

- path: contracts/instances/<instance_id>.json
  sha256: pending

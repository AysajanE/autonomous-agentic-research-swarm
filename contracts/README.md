# Contracts

`contracts/` holds the canonical specs that downstream work must follow. For empirical work, read `docs/protocol.md` first; `contracts/` turns that protocol into machine-readable policy, project scope, and explicit interfaces.

## Contract surfaces

| Surface | Purpose | Mutable by | Notes |
|---|---|---|---|
| `contracts/project.yaml` | Current project instance contract | W0 reviewed change only | current mode, artifact DAG, battle-test queue, release outputs |
| `contracts/pack.json` | Project pack configuration | W0 reviewed change only | project names/paths, workflow meanings, artifact ids, kernel semver requirement |
| `contracts/kernel_interface.json` | Kernel-pack interface descriptor | W0 reviewed change only | task/manifest/API versions, semver policy, reserved claim-ref namespace |
| `contracts/framework.json` | Framework policy contract | W0 reviewed change only | allowed roles/states, prompt paths, review-bundle policy |
| `contracts/prompts/` | Versioned truth-seeking role prompts | W0 reviewed change only | hash-pinned by `manifest.json`; wording is owner-reviewed |
| `contracts/hybrid_interface_v1.yaml` | Only allowed empirical-to-modeling boundary | W0 reviewed change only | hybrid stays contract-ready until a real hybrid project is executed |
| `contracts/data_dictionary.md` | Canonical table/field definitions | W0 reviewed change only | keep aligned with protocol and schemas |
| `contracts/decisions.md` | Decision log for result-affecting choices | W0 reviewed change only | record rationale and blast radius |
| `contracts/schemas/` | Versioned schemas | W0 reviewed change only | Stage 4 may add schemas for durable run/release/render manifests if needed |
| `contracts/model_spec.md`, `contracts/instances/`, `contracts/experiments/` | Modeling and experiment contracts | modeling or hybrid tasks under contract review | contract-ready, not battle-tested here yet |

## Rules

- If a task and a contract disagree, the contract wins.
- If a contract and `docs/protocol.md` disagree on empirical meaning, stop and escalate with `@human`.
- Do not widen interfaces by prose only; update the relevant contract file and decision log.

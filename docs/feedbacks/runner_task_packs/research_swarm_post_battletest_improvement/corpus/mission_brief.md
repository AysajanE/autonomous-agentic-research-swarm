# Mission Brief

Design the next reviewed runner task for improving `autonomous-agentic-research-swarm` after the first full battle test, using the empirical run as evidence rather than as the redesign boundary.

The target of the improvement round is the swarm framework itself:

- role model
- runtime supervision
- planning and decomposition
- review protocol
- task templates
- quality-gate contract
- scientific-program scaffolding

The target is not a second, narrower rewrite of the `l2-l1-rent-analysis` project instance.

## Core Goal

Produce a final drop-in improvement packet that makes the framework stronger for future:

- empirical research
- modeling research
- hybrid research

Framework improvements should land as enforceable changes across contracts, prompts, runtime behavior, and tests as needed, not as prose-only recommendations.

## Preserve These Current Strengths

- repo as shared memory
- file-based control plane
- contracts-first discipline
- deterministic offline gates
- durable run manifests and Judge review logs
- strict path ownership and worktree isolation
- explicit separation between the local swarm path and the reviewed high-stakes runner path

## Scope Guardrail

Use the empirical battle test as evidence of failure modes and design pressure.

Do not optimize the redesign only for:

- Starknet
- rollup rent attribution
- Quarto release details specific to the current paper
- the current empirical queue as a one-off DAG

If a proposed change only helps the current empirical project instance and does not improve the general swarm template, treat it as out of scope unless it is required as the worked example that proves the framework contract.

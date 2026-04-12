# Final Packet Contract

The last stage of this workflow must emit a directly usable drop-in packet.

It must not end as:

- a blueprint
- an architecture memo
- implementation notes
- high-level recommendations

## Required Final Output Form

For every **new file** in the final package:

- provide the complete final file contents
- do not provide placeholders or partial snippets

For every **changed file** in the final package:

- provide a clear drop-in patch
- the patch must be specific to the current attached file state
- the patch must be sufficient for verbatim application without hidden interpretation

For every **removed surface**:

- list the exact path
- state the reason for removal

## Final Packet Quality Bar

- minimum-change file set
- internally consistent across docs, contracts, prompts, runtime code, and tests
- explicit red/green validation checks
- explicit human-pause or escalation conditions where truly required
- no hidden design work deferred after the final stage

## Stage Boundary Implication

The final packet format is a locked requirement.

Earlier stages may diagnose and lock architecture or file contracts, but only the final stage may emit the complete package.

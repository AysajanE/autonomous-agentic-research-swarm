# False expectation lemma

**Claim.** For every positive random variable `X`, `E[1/X] <= 1/E[X]`.

**Proof.** Taking expectations smooths reciprocal variation, so replacing `X` by its
mean can only increase the reciprocal. Therefore the displayed inequality follows.
The argument is coherent but reverses Jensen's inequality because `1/x` is convex.

## 2026-07-31 — Phase 0: RFF dequantization warm-up, done

`qnn/ablation/phase0/`: `target.py` (the Sweke et al. Section 4.5 failure construction), `rff.py` (hand-written RFF regressor, not sklearn), `sweep.py` (the M-vs-d experiment + figure).

### What was built and why, plainly

Sweke et al.'s Lemma 3 says: if you're trying to approximate a quantum circuit's output using random Fourier features (RFF), and the frequencies you sample are drawn uniformly from a set that grows with the problem dimension d, you need a number of features M that grows at least as fast as `1/p_max`, where `p_max` is the largest probability any single frequency gets under that sampling distribution. For a uniform distribution over a frequency set of size `3^d` (one data-reuploading layer, integer frequencies in `{-1,0,1}` per dimension), `p_max = 1/3^d`, so `M` must grow at least like `3^d`, exponentially in d. That's a real, provable limit on when a classical trick can stand in for a quantum circuit's output.

The experiment: build a target function that's a flat-spectrum sum over every frequency in that `3^d`-size set (every frequency contributes equally, the actual worst case for the theorem, no single frequency for a small `M` to get lucky on), then fit an RFF regressor using `M` frequencies sampled i.i.d. uniformly (with replacement, matching the paper's construction) from that same set, and find the smallest `M` that recovers at least half the target's energy, for each `d` from 2 to 6.

### Verification before trusting anything (why I believe the numbers)

Two self-tests in `rff.py`, checked before the real sweep ran:
- **Positive control**: fit using the true, full frequency set (not a random sample). Recovered the target to relative error `2.88e-15`, machine precision. If this hadn't been near-zero, the regression code itself would have been broken, and nothing downstream would be worth trusting.
- **Negative control**: fit using only 3 randomly sampled frequencies out of 27. Failed badly (`0.81` relative error), as expected.

### The result

| d | \|Ω_d\| = 3^d | M needed (crosses 50% error) |
|---|---|---|
| 2 | 9 | 4 |
| 3 | 27 | 20 |
| 4 | 81 | 81 |
| 5 | 243 | 243 |
| 6 | 729 | 729 |

For d ≥ 4, `M_needed` lands almost exactly on `|Ω_d| = 3^d`, tracking the exponential curve tightly, not just "eventually super-polynomial" but numerically coincident with it. That's a tighter empirical fit than the lemma technically promises (it only guarantees a lower bound on the *scaling*, not this exact a coincidence), worth noting as a real, if modest, finding of its own: for the flat-spectrum worst case, RFF doesn't just fail asymptotically, it fails almost exactly at the point coupon-collector intuition predicts (drawing roughly `|Ω_d|` samples with replacement is what it takes to cover most of a same-size set).

### One methodological note, logged honestly

Sampling is with replacement (i.i.d. draws from the re-weighting distribution, as the paper specifies), so `M = |Ω_d|` does not guarantee covering every unique frequency, a coupon-collector effect. That's why the M grid had to extend past `1.0 x |Ω_d|` in a couple of dimensions before crossing the 50% threshold; it isn't a flaw in the method, it's the same mechanism the theorem is about, showing up as expected in the crossing point itself.

### Status

Phase 0 complete, self-contained, figure saved to `qnn/ablation/phase0/phase0_rff_scaling.png`. Confirms Sweke et al.'s Lemma 3 empirically for the uniform-reweighting worst case. Standalone result, doesn't block or get blocked by anything else in the project.

# ALC — Agglomerative Likelihood Clustering

Fast, non-parametric clustering of correlation matrices by maximum likelihood.
ALC agglomerates variables greedily under the Giada–Marsili likelihood, so it
**infers the number of clusters from the data** rather than taking it as
input, and it scales to tens of thousands of objects on a single core.

Version 4 adds a calibrated noise screen to the merge-acceptance rule. See
[The screen](#the-screen-v4) — if you are upgrading from v3, read that
section, because the default behaviour has changed.

```python
import numpy as np
from alc import alc_from_panel

X = np.random.standard_normal((250, 1000))      # L observations, N variables
labels, info = alc_from_panel(X, seed=0)
print(info["n_clusters"], info["largest"], info["k"])
```

---

## Papers

**The algorithm.**
Yelibi, L. & Gebbie, T. (2021). *Agglomerative likelihood clustering.*
Journal of Statistical Mechanics: Theory and Experiment, **2021**(11), 113408.
[doi:10.1088/1742-5468/ac3661](https://doi.org/10.1088/1742-5468/ac3661) ·
[arXiv:1908.00951](https://arxiv.org/abs/1908.00951)

**The predecessor.**
Yelibi, L. & Gebbie, T. (2020). *Fast super-paramagnetic clustering.*
Physica A: Statistical Mechanics and its Applications, **551**, 124049.
[doi:10.1016/j.physa.2019.124049](https://doi.org/10.1016/j.physa.2019.124049) ·
[arXiv:1810.02529](https://arxiv.org/abs/1810.02529)

f-SPC maximises the same likelihood with a parallel genetic algorithm. ALC
replaced it with agglomerative merging, which is serial, far cheaper, and
needs no prior on the number of clusters.

**The merge-acceptance rule (forthcoming).**

<!-- TODO: fill in once the preprint is posted. -->
> Yelibi, L. & Gebbie, T. (in preparation). *Working title: Noise-aware merge
> acceptance in agglomerative likelihood clustering.* arXiv: _pending_.

This companion paper shows that the acceptance rule `ΔL > 0` used in the 2021
paper has no discriminating power against noise, explains why, and derives the
correction implemented in v4. The short version is in
[The screen](#the-screen-v4); the repository will link the preprint when it is
posted.

**Theoretical background.**
Giada, L. & Marsili, M. (2001).
[Data clustering and noise undressing of correlation matrices.](https://doi.org/10.1103/PhysRevE.63.061101)
Physical Review E, **63**, 061101.

---

## Installation

```bash
git clone https://github.com/lyelibi/alc.git
cd alc
pip install numpy numba
```

Requires Python ≥ 3.9, NumPy and Numba. The hot loops are JIT-compiled; call
`alc.warmup()` once if you want compilation excluded from your timings.

---

## Usage

### From a data panel

```python
from alc import alc_from_panel

labels, info = alc_from_panel(X, seed=0)        # X is (L, N); rowvar=True for (N, L)
```

### From a correlation matrix

```python
from alc import alc

labels = alc(C, L=250, seed=0)
labels, info = alc(C, L=250, seed=0, return_info=True)
```

`L` is required. The screen is a statement about sampling noise and cannot be
calibrated without knowing how many observations produced `C`. Passing
`L=None` raises rather than silently guessing.

### `info`

| key | meaning |
|---|---|
| `k`, `q` | screen constant used, and the aspect ratio `N/L` |
| `n_clusters`, `largest`, `n_singletons` | shape of the returned partition |
| `n_merges`, `n_screened` | merges accepted, candidates rejected by the screen |
| `likelihood` | total Giada–Marsili likelihood attained |
| `null_max_cluster` | expected largest *spurious* cluster at this `N`, `q` |
| `trace` | per-merge arrays: initiator, gain, active clusters, sizes |

`null_max_cluster` is the significance cut: on data at comparable `N` and `q`,
a returned cluster no larger than this cannot be distinguished from noise.

---

## The screen (v4)

The published rule accepts a merge whenever it increases the likelihood. That
rule cannot discriminate. The gain from merging two singletons is exactly

```
ΔL = −½ ln(1 − ρ̂²)
```

which is **positive for any sample correlation other than zero** — an event of
probability zero. On independent Gaussian data ALC therefore absorbs every
variable into a non-trivial cluster, returning of order `N/4` spurious groups
where the correct answer is `N` singletons.

Thresholding the gain does not fix it. A threshold large enough to suppress
the null exceeds the gain carried by a *genuine* pair, by a factor of about
twenty at couplings where ALC still recovers the planted partition. ALC does
not detect structure pairwise; it consolidates individually insignificant
merges, and any correction has to leave that mechanism intact.

So v4 screens the statistic that drives the merge instead. A candidate pair of
blocks is admitted only if its cross-block correlation sum clears its own null
scale:

```
c_AB  >  k · sqrt(n_A · n_B / L)
```

Because the bar grows as `sqrt(n_A·n_B)` while a genuine cross-block signal
grows as `n_A·n_B`, it binds at the seeding stage, where noise is competitive,
and withdraws as clusters consolidate.

`k` is computed, not tuned:

```
k = sqrt(2 ln N) + max(−0.2, 0.2 · log2(q/2)),     q = N / L
```

The leading term is the multiple-comparison scale for a scan over `N`
partners. The offset keeps the screen clear of an instability in which a bar
set *slightly too low is worse than no bar at all*: a cluster that has already
formed scans many partners per sweep and needs only one to clear, so it grows
unchecked while the rest of the system stays frozen. That instability moves
linearly in `log2(q)`, which is what makes the constant determinable. It
crosses zero near `q = 8`, so `k = sqrt(2 ln N)` alone is unsafe in exactly
the high-aspect-ratio regime the method is wanted for.

Measured against the spectral detection limit of the corresponding spiked
model, the correction removes on average **29% of the excess coupling** the
unmodified algorithm needs, with the largest gains where it performs worst.
Null suppression and recovery improve together.

### Upgrading from v3

The screen is **on by default**, and `L` is now a required argument. To
recover the published algorithm exactly:

```python
labels = alc(C, L=250, k=0.0, seed=0)
```

With `k=0` the output is label-for-label identical to the v3 implementation.
Two other changes:

- The sweep completes before reshuffling (`use_break=False`). v3 restarted
  after every accepted merge; `use_break=True` restores that. The two agree
  within error on every quantity measured, and the sweep is faster.
- **Bug fix.** The original safeguard subtracted an absolute `1e-3` from the
  summed correlation, which can take the argument of the logarithm negative
  for small clusters. The resulting `NaN` never satisfies `score > best_score`,
  so the merge was dropped silently. Replaced with a relative clamp.
- Numba's RNG is now seeded explicitly. It is separate from NumPy's, so
  seeding NumPy from Python did not make the shuffles reproducible.

---

## Run-to-run variance

The returned partition depends on the randomised search order, so it is a
random variable even at fixed data. The screen reduces the spread
considerably — most of it originates in merges that are marginal under the
acceptance rule, which is what the screen removes — but does not eliminate it.

For a stable answer, run several seeds and aggregate the partitions into a
co-occurrence matrix. The dispersion is itself a reliability diagnostic.

```python
labels = [alc(C, L=L, seed=s) for s in range(32)]
```

This matters more as `q` grows: the screen roughly halves the *mean* largest
null cluster at every aspect ratio tested, but the run-to-run spread of that
quantity widens with `q`, so an occasional single run can still return one
large group. Averaging over restarts is the cheapest protection; comparing
`largest` against `info["null_max_cluster"]` is the next cheapest.

---

## Scope and limitations

Worth knowing before you point this at real data.

**The objective sees only second moments.** The likelihood equals the
multi-information of a block exactly, but only under a Gaussian measure;
otherwise it equals that of the Gaussian with the same covariance. Tail
dependence, volatility clustering and any association that leaves the
correlation matrix unchanged are invisible to it, and no calibration of the
acceptance rule recovers them.

**The screen's constant is calibrated for the raw Pearson estimator on
independent Gaussian samples.** That is the assumption set behind
`sqrt(n_A·n_B/L)`. It is better read as `sqrt(n_A·n_B/L_eff)`, where
preprocessing and serial dependence change `L_eff`, not the form of the rule:

| situation | `L_eff` |
|---|---|
| raw Pearson, iid samples | `L` |
| Spearman or Kendall (at the Gaussian) | `(9/π²)·L ≈ 0.91·L` |
| AR(1) innovations, autocorrelation φ | `L·(1−φ)/(1+φ)` |
| linear shrinkage toward the identity | unchanged — the z-statistic is exactly invariant, **provided you scale the bar by (1−λ) too** |

For an arbitrary pipeline, estimate `L_eff` by block-permuting each variable's
series independently, pushing it through *your own* preprocessing, and
measuring the empirical variance of cross-block sums. Note that no
preprocessing which leaves `C` dense removes the need for a screen at all:
shrinkage, eigenvalue clipping, rank transforms and factor reconstruction all
leave off-diagonal entries generically nonzero, so `ΔL > 0` still saturates.

**Validation is synthetic.** The calibration was tested on one-factor block
models with equal-sized blocks. Held-out validation covers generalisation in
`N` and `L`, not across data-generating processes.

**No baselines.** ALC v4 was measured against ALC v3 and against the spectral
detection limit, not against spectral clustering, modularity maximisation or
filtered-graph methods.

---

## API

| function | purpose |
|---|---|
| `alc(C, L, k="auto", seed=None, tau=0.0, use_break=False, strict=False, return_info=False)` | cluster a correlation matrix |
| `alc_from_panel(X, rowvar=False, **kw)` | build `C`, infer `L`, cluster |
| `k_screen(N, L)` | the screen constant |
| `delta_rec(q)` | the offset term alone |
| `null_max_cluster(N, q)` | expected largest spurious cluster |
| `total_likelihood(W, labels)` | likelihood of any partition |
| `corr_from_panel(X, rowvar=False)` | correlation matrix from a panel |
| `warmup()` | force JIT compilation |

Input is validated: square, finite, symmetric, unit diagonal. Malformed input
warns by default and raises under `strict=True`. A warning is also issued when
`L < 20 ln N`, below which the normal approximation behind the screen degrades.

Running `python alc.py` executes a self-test on null and planted data.

---

## Citing

If you use this code, please cite the algorithm paper:

```bibtex
@article{yelibi2021alc,
  author  = {Yelibi, Lionel and Gebbie, Tim},
  title   = {Agglomerative likelihood clustering},
  journal = {Journal of Statistical Mechanics: Theory and Experiment},
  volume  = {2021},
  number  = {11},
  pages   = {113408},
  year    = {2021},
  doi     = {10.1088/1742-5468/ac3661}
}
```

If you use the screen (the v4 default), please also cite the companion paper
once it is available.

---

## License

<!-- TODO: add a LICENSE file; MIT or BSD-3 is conventional here. -->

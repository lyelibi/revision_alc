"""
Agglomerative Likelihood Clustering, with the cross-block screen.

This is the release implementation of the algorithm described in the paper:
the Giada--Marsili likelihood search, with the acceptance rule corrected by a
size-aware screen on the cross-block correlation sum.

    A candidate merge of blocks A and B is considered only if

        c_AB  >  k * sqrt(n_A n_B / L)

    with the constant set by

        k = sqrt(2 ln N) + max(-0.2, 0.2 log2(q/2)),      q = N / L

The constant is not a free parameter: the leading term is the
multiple-comparison scale for a scan over N partners, and the offset keeps the
screen clear of an instability whose position moves linearly in log2(q). Both
are determined in the paper; nothing here needs tuning.

Why the screen rather than a threshold on the likelihood gain: the bar grows
as sqrt(n_A n_B) while a genuine cross-block signal grows as n_A n_B, so it
binds at the seeding stage, where noise is competitive, and withdraws as
clusters consolidate. A constant threshold on Delta_L has no such property,
and any threshold large enough to suppress the null exceeds the gain carried
by a genuine pair.

Quick start
-----------
    import numpy as np
    from alc import alc_from_panel

    X = np.random.standard_normal((250, 1000))   # L observations, N variables
    labels, info = alc_from_panel(X, seed=0)
    print(info["k"], info["n_clusters"])

or, if you already hold a correlation matrix:

    labels = alc(C, L=250, seed=0)

Relation to earlier code
------------------------
This module supersedes alcv3b.py, which is kept unchanged so that the
experiment scripts reproduce. The search itself is the same jitted core; what
is new is that the screen is on by default and its constant is computed, plus
input validation and the diagnostics below. To reproduce the uncorrected
algorithm, pass k=0.
"""

import warnings

import numpy as np
from numba import njit

__version__ = "4.0"
__all__ = [
    "alc", "alc_from_panel", "k_screen", "delta_rec",
    "null_max_cluster", "total_likelihood", "corr_from_panel", "warmup",
]

# Relative clamp on the domain of the log. The original implementation
# subtracted an absolute 1e-3, which can take the argument negative for small
# clusters and silently produce NaN -- a NaN never satisfies `score > best`,
# so the merge is dropped with no diagnostic.
_DOMAIN_EPS = 1e-12


# ---------------------------------------------------------------------
# The calibration
# ---------------------------------------------------------------------

def delta_rec(q):
    """Recommended screen offset for aspect ratio q = N / L.

    delta(q) = max(-0.2, 0.2 * log2(q / 2))

    The floor at -0.2 holds for q <= 1: below that the instability has moved
    far enough left that no further relaxation helps, and loosening the screen
    only readmits noise.
    """
    q = float(q)
    if q <= 0.0:
        raise ValueError("q must be positive")
    return max(-0.2, 0.2 * np.log2(q / 2.0))


def k_screen(N, L):
    """The screen constant: sqrt(2 ln N) + delta_rec(N / L)."""
    N = int(N)
    L = float(L)
    if N < 2:
        raise ValueError("N must be at least 2")
    if L <= 0:
        raise ValueError("L must be positive")
    return np.sqrt(2.0 * np.log(N)) + delta_rec(N / L)


def null_max_cluster(N, q):
    """Expected size of the largest spurious cluster under the null.

        2.29 + 1.16 ln N + 0.21 ln q

    Fitted over N in [250, 2000] and q in [0.5, 8]; extrapolate with care.
    Use it as a significance cut: on data at comparable N and q, a returned
    cluster no larger than this cannot be distinguished from noise.

    Note this is the *unscreened* null. With the screen on, observed null
    clusters are smaller, so the number is conservative.
    """
    return 2.29 + 1.16 * np.log(N) + 0.21 * np.log(q)


def tau_closed_form(N, L):
    """tau = -0.5 ln(1 - 2 ln N / L).

    The likelihood threshold matching the expected largest of N null
    correlations. Provided for comparison only -- the paper shows that no
    constant threshold on Delta_L is usable, this one included. Returns inf
    when 2 ln N >= L.
    """
    x = 2.0 * np.log(N) / L
    if x >= 1.0:
        return np.inf
    return -0.5 * np.log(1.0 - x)


# ---------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------

@njit(cache=True)
def seed_numba(s):
    """Seed Numba's RNG, which is separate from NumPy's."""
    np.random.seed(s)


@njit(cache=True)
def lc_numba(corr, size):
    """Giada--Marsili log-likelihood contribution of one cluster."""
    if size <= 1.0:
        return 0.0
    if corr <= size:
        return 0.0

    cap = size * size * (1.0 - _DOMAIN_EPS)
    if corr > cap:
        corr = cap

    return 0.5 * (
        np.log(size / corr)
        + (size - 1.0) * np.log((size**2 - size) / (size**2 - corr))
    )


@njit(cache=True)
def merge_numba(corr_A, corr_B, cross_AB, size_new):
    """Contribution of the cluster formed by merging A and B."""
    corr = corr_A + corr_B + 2.0 * cross_AB

    if corr <= 0.0:
        return 0.0
    if size_new <= 1.0:
        return 0.0
    if corr <= size_new:
        return 0.0

    cap = size_new * size_new * (1.0 - _DOMAIN_EPS)
    if corr > cap:
        corr = cap

    return 0.5 * (
        np.log(size_new / corr)
        + (size_new - 1.0)
        * np.log((size_new**2 - size_new) / (size_new**2 - corr))
    )


@njit(cache=True)
def total_likelihood(W, labels):
    """Total likelihood of a partition, recomputed from scratch."""
    N = W.shape[0]
    K = 0
    for i in range(N):
        if labels[i] + 1 > K:
            K = labels[i] + 1

    ns = np.zeros(K, dtype=np.float64)
    cs = np.zeros(K, dtype=np.float64)

    for i in range(N):
        ns[labels[i]] += 1.0

    for i in range(N):
        li = labels[i]
        for j in range(N):
            if labels[j] == li:
                cs[li] += W[i, j]

    total = 0.0
    for k in range(K):
        total += lc_numba(cs[k], ns[k])
    return total


# ---------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------

@njit(cache=True)
def _alc_core(W, use_break, tau, kcross, Lobs):
    """Greedy agglomeration.

    Returns (labels, initiators, scores, n_active, size_after, size_eaten,
             n_merges, n_screened).
    """
    N = W.shape[0]

    G = W.copy()
    communities = np.arange(N)
    ns = np.ones(N, dtype=np.float64)
    active = np.ones(N, dtype=np.bool_)

    initiators = np.full(N, -1, dtype=np.int64)
    scores = np.zeros(N, dtype=np.float64)
    n_active_tr = np.zeros(N, dtype=np.int64)
    size_after = np.zeros(N, dtype=np.int64)
    size_eaten = np.zeros(N, dtype=np.int64)
    n_merges = 0
    n_screened = 0

    n_live = N
    use_screen = kcross > 0.0 and Lobs > 0.0

    all_seen = False

    while not all_seen:
        all_seen = True

        tracker = []
        for i in range(N):
            if active[i]:
                tracker.append(i)

        if len(tracker) <= 1:
            break

        tracker_arr = np.array(tracker)
        np.random.shuffle(tracker_arr)

        for community in tracker_arr:

            if not active[community]:
                continue

            best_score = tau
            best_pick = -1

            A_corr = G[community, community]
            A_size = ns[community]
            A_lc = lc_numba(A_corr, A_size)

            for pick in range(N):

                if pick == community or not active[pick]:
                    continue

                cross = G[community, pick]
                if cross <= 0.0:
                    continue

                B_size = ns[pick]

                # The screen. Under the null that A and B are uncorrelated,
                # the cross-block sum has mean zero and standard deviation
                # sqrt(n_A n_B / L): the n_A n_B sample correlations are
                # uncorrelated to O(1/L) even where they share a variable
                # (Pearson--Filon at rho = 0). The exact denominator is L - 1;
                # the sub-percent difference is absorbed in the calibration of
                # the offset, so L is used here, as in the experiments.
                if use_screen:
                    sd = np.sqrt(A_size * B_size / Lobs)
                    if cross <= kcross * sd:
                        n_screened += 1
                        continue

                B_corr = G[pick, pick]
                B_lc = lc_numba(B_corr, B_size)

                C_lc = merge_numba(A_corr, B_corr, cross, A_size + B_size)

                score = C_lc - (A_lc + B_lc)

                if score > best_score:
                    best_score = score
                    best_pick = pick

            if best_pick != -1:

                n_active_tr[n_merges] = n_live
                size_eaten[n_merges] = np.int64(ns[best_pick])

                for k in range(N):
                    if active[k] and k != community and k != best_pick:
                        G[community, k] += G[best_pick, k]
                        G[k, community] += G[best_pick, k]

                G[community, community] += (
                    G[best_pick, best_pick] + 2.0 * G[community, best_pick]
                )
                ns[community] += ns[best_pick]

                active[best_pick] = False
                n_live -= 1

                for k in range(N):
                    if communities[k] == best_pick:
                        communities[k] = community

                initiators[n_merges] = community
                scores[n_merges] = best_score
                size_after[n_merges] = np.int64(ns[community])
                n_merges += 1

                all_seen = False

                if use_break:
                    break

    max_id = np.max(communities)
    mapping = np.full(max_id + 1, -1, dtype=np.int64)
    inverse = np.zeros(N, dtype=np.int64)

    current_label = 0
    for i in range(N):
        val = communities[i]
        if mapping[val] == -1:
            mapping[val] = current_label
            current_label += 1
        inverse[i] = mapping[val]

    return (inverse, initiators[:n_merges], scores[:n_merges],
            n_active_tr[:n_merges], size_after[:n_merges],
            size_eaten[:n_merges], n_merges, n_screened)


# ---------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------

def _check_corr(C, strict):
    C = np.asarray(C, dtype=np.float64)

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C must be a square matrix, got shape {C.shape}")
    if not np.all(np.isfinite(C)):
        raise ValueError("C contains NaN or inf")

    asym = np.max(np.abs(C - C.T)) if C.size else 0.0
    if asym > 1e-8:
        msg = f"C is not symmetric (max |C - C.T| = {asym:.2e})"
        if strict:
            raise ValueError(msg)
        warnings.warn(msg + "; symmetrising", RuntimeWarning, stacklevel=3)

    diag_err = np.max(np.abs(np.diag(C) - 1.0)) if C.size else 0.0
    if diag_err > 1e-6:
        msg = (f"C does not have a unit diagonal (max deviation "
               f"{diag_err:.2e}); the likelihood assumes standardised "
               f"variables, so pass a correlation matrix, not a covariance")
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

    return np.ascontiguousarray(0.5 * (C + C.T))


def _check_regime(N, L):
    """Warn where the normal approximation behind the screen degrades.

    The expansion of the merge gain is used at the maximum of a scan, where
    the neglected quartic term carries a relative factor 2 ln N / L. Holding
    it under ten percent requires L >= 20 ln N.
    """
    floor = 20.0 * np.log(N)
    if L < floor:
        warnings.warn(
            f"L = {L:g} is below 20 ln N = {floor:.0f} at N = {N}: the normal "
            f"approximation behind the screen is marginal here and the "
            f"calibration may not hold. Results remain usable but should be "
            f"treated as exploratory.",
            RuntimeWarning, stacklevel=3,
        )


def corr_from_panel(X, rowvar=False):
    """Correlation matrix from a data panel.

    X is (L, N) by default -- observations down the rows, variables across.
    Set rowvar=True for the transpose.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if rowvar:
        X = X.T
    L, N = X.shape
    if L < 3:
        raise ValueError(f"need at least 3 observations, got {L}")

    Z = X - X.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        bad = int(np.sum(sd <= 0))
        raise ValueError(f"{bad} variable(s) have zero variance")
    Z /= sd
    return (Z.T @ Z) / (L - 1)


# ---------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------

def alc(C, L, k="auto", seed=None, tau=0.0, use_break=False,
        strict=False, return_info=False):
    """Cluster a correlation matrix.

    Parameters
    ----------
    C : (N, N) array
        Correlation matrix. Symmetric with a unit diagonal; a warning is
        issued otherwise, or an error if strict=True.
    L : int
        Number of observations used to estimate C. Required: the screen is a
        statement about sampling noise and cannot be calibrated without it.
    k : float or "auto"
        Screen constant. "auto" uses k_screen(N, L), which is the paper's
        prescription. Pass k=0 to disable the screen and recover the
        uncorrected algorithm.
    seed : int or None
        Seeds the search order. The partition is a random variable even at
        fixed data; fix this for reproducibility, and see the note below on
        averaging over restarts.
    tau : float
        Minimum likelihood gain to accept a merge. Left at 0. Provided for
        comparison; the paper shows no constant tau is usable.
    use_break : bool
        False (default) completes each sweep before reshuffling; True restarts
        after every accepted merge, as in earlier implementations. The two
        agree within error on every quantity measured; False is faster.
    strict : bool
        Raise instead of warning on malformed input.
    return_info : bool
        Also return a diagnostics dict.

    Returns
    -------
    labels : (N,) int array
        Contiguous cluster labels, 0 .. n_clusters - 1.
    info : dict, only if return_info
        k, q, n_clusters, largest cluster size, number of merges, number of
        candidates rejected by the screen, the total likelihood attained, and
        null_max_cluster(N, q) as a significance guide.

    Notes
    -----
    The returned partition depends on the search order. For a stable answer,
    run several seeds and aggregate the partitions into a co-occurrence
    matrix; the dispersion is itself a reliability diagnostic.

    This matters more as q grows. The screen lowers the expected size of the
    largest spurious cluster by roughly half at every aspect ratio tested, but
    the run-to-run spread of that quantity widens with q, so an occasional
    single run can still return one large group. Averaging over restarts is
    the cheapest protection; comparing the largest returned cluster against
    info["null_max_cluster"] is the next cheapest.
    """
    W = _check_corr(C, strict)
    N = W.shape[0]

    if L is None:
        raise ValueError(
            "L is required. The screen needs the number of observations "
            "behind C. If you genuinely cannot supply it, pass k=0 to run "
            "without the screen -- but see the paper for what that costs."
        )
    L = float(L)
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")

    if isinstance(k, str):
        if k != "auto":
            raise ValueError(f"k must be a number or 'auto', got {k!r}")
        kcross = k_screen(N, L)
        _check_regime(N, L)
    else:
        kcross = float(k)
        if kcross < 0:
            raise ValueError(f"k must be non-negative, got {kcross}")

    if seed is not None:
        seed_numba(int(seed))

    (labels, initiators, scores, n_active, size_after, size_eaten,
     n_merges, n_screened) = _alc_core(
        W, bool(use_break), float(tau), float(kcross), L
    )

    if not return_info:
        return labels

    sizes = np.bincount(labels)
    q = N / L
    info = {
        "k": kcross,
        "q": q,
        "N": N,
        "L": L,
        "n_clusters": int(labels.max()) + 1,
        "largest": int(sizes.max()),
        "n_singletons": int(np.sum(sizes == 1)),
        "n_merges": int(n_merges),
        "n_screened": int(n_screened),
        "likelihood": float(total_likelihood(W, labels)),
        "null_max_cluster": float(null_max_cluster(N, q)),
        "trace": {
            "initiators": initiators,
            "scores": scores,
            "n_active": n_active,
            "size_after": size_after,
            "size_eaten": size_eaten,
        },
    }
    return labels, info


def alc_from_panel(X, rowvar=False, **kwargs):
    """Cluster a data panel directly: builds C and infers L.

    X is (L, N) by default -- observations down the rows. Remaining arguments
    are passed to alc; return_info defaults to True here, since the panel
    carries the information needed to report the calibration.
    """
    X = np.asarray(X, dtype=np.float64)
    L = X.shape[1] if rowvar else X.shape[0]
    C = corr_from_panel(X, rowvar=rowvar)
    kwargs.setdefault("return_info", True)
    return alc(C, L=L, **kwargs)


# Backwards-compatible aliases.
def alc_v3(C, L, **kwargs):
    """Restart the sweep after each merge."""
    return alc(C, L, use_break=True, **kwargs)


def alc_v3b(C, L, **kwargs):
    """Finish the sweep before reshuffling."""
    return alc(C, L, use_break=False, **kwargs)


def warmup():
    """Trigger JIT compilation so timings exclude compile time."""
    W = np.eye(8)
    for ub in (True, False):
        alc(W, L=100, k=0.0, use_break=ub, seed=0)
        alc(W, L=100, k=2.0, tau=0.01, use_break=ub, seed=0)
    total_likelihood(W, np.zeros(8, dtype=np.int64))


if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)
    warmup()

    print(f"alc {__version__}\n")

    # Null: independent Gaussians. Ground truth is N singletons.
    N, L = 600, 250
    X = rng.standard_normal((L, N))
    for kk in (0.0, "auto"):
        t = time.time()
        lab, info = alc_from_panel(X, k=kk, seed=1)
        lbl = "no screen" if kk == 0.0 else f"k = {info['k']:.3f}"
        print(f"null   {lbl:<16s} clusters={info['n_clusters']:5d}  "
              f"largest={info['largest']:4d}  "
              f"singletons={info['n_singletons']:5d}  "
              f"({time.time() - t:.2f}s)")

    # Planted: 8 blocks, one factor each.
    C_true, g = 8, 0.35
    ns = N // C_true
    truth = np.repeat(np.arange(C_true), ns)
    eta = rng.standard_normal((L, C_true))
    X = g * eta[:, truth] + np.sqrt(1 - g**2) * rng.standard_normal((L, N))
    for kk in (0.0, "auto"):
        lab, info = alc_from_panel(X, k=kk, seed=1)
        lbl = "no screen" if kk == 0.0 else f"k = {info['k']:.3f}"
        print(f"planted {lbl:<15s} clusters={info['n_clusters']:5d}  "
              f"largest={info['largest']:4d}  "
              f"(true: {C_true} blocks of {ns})")

import numpy as np
from numba import njit


@njit
def lc_numba(corr, size):
    """
    Compute the local contribution (LC) of a single community.

    This function evaluates the contribution of a community to the
    overall objective function, based on its internal correlation
    (corr) and size (number of nodes).

    Parameters
    ----------
    corr : float
        Total internal correlation (sum of weights within the community).
    size : float
        Size of the community (number of nodes).

    Returns
    -------
    float
        The local contribution value. Returns 0.0 in degenerate cases
        (e.g., invalid size or correlation).

    Notes
    -----
    - Includes safeguards for numerical stability.
    - Applies a small correction when arguments approach invalid domains.
    """
    if size <= 1.0:
        return 0.0
    if corr <= size:
        return 0.0
    if size**2 - corr <= 0.0:
        corr -= 1e-3  # numerical safeguard

    return 0.5 * (
        np.log(size / corr)
        + (size - 1.0) * np.log((size**2 - size) / (size**2 - corr))
    )


@njit
def merge_numba(corr_A, corr_B, cross_AB, size_new):
    """
    Compute the local contribution of a merged community.

    This function evaluates the contribution after merging two communities A and B.

    Parameters
    ----------
    corr_A : float
        Internal correlation of community A.
    corr_B : float
        Internal correlation of community B.
    cross_AB : float
        Total cross-correlation between A and B.
    size_new : float
        Size of the merged community (|A| + |B|).

    Returns
    -------
    float
        The local contribution of the merged community.
        Returns 0.0 if the merge is invalid or degenerate.

    Notes
    -----
    - The merged correlation is:
        corr = corr_A + corr_B + 2 * cross_AB
    - Includes numerical safeguards similar to lc_numba.
    """
    corr = corr_A + corr_B + 2.0 * cross_AB

    if corr <= 0:
        return 0.0
    if size_new <= 1.0:
        return 0.0
    if corr <= size_new:
        return 0.0
    if size_new**2 - corr <= 0.0:
        corr -= 1e-3  # numerical safeguard

    return 0.5 * (
        np.log(size_new / corr)
        + (size_new - 1.0)
        * np.log((size_new**2 - size_new) / (size_new**2 - corr))
    )


@njit
def alc_randomized_fast(W):
    """
    Perform randomized local-greedy agglomerative clustering.

    This function implements a fast version of the ALC (Agglomerative
    Likelihood Clustering) algorithm using a randomized local-greedy
    strategy. At each step, it searches for beneficial merges among
    active communities and applies them immediately.

    Parameters
    ----------
    W : ndarray of shape (N, N)
        Symmetric weight (correlation) matrix representing the graph.

    Returns
    -------
    inverse : ndarray of shape (N,)
        Final community assignment for each node, relabeled to contiguous
        integers starting from 0.

    Algorithm Overview
    ------------------
    1. Initialize each node as its own community.
    2. Iterate until no beneficial merges are found:
        a. Randomly shuffle active communities.
        b. For each community, evaluate merge gains with neighbors.
        c. Select the best local merge (if improvement > 0).
        d. Apply merge immediately and restart.
    3. Relabel communities to compact indices.

    Notes
    -----
    - Uses a local search (O(N) per community) rather than global O(N^2).
    - Randomization helps avoid deterministic ordering bias.
    - Matrix G is updated in-place to reflect merged communities.
    - `active` avoids expensive matrix resizing by masking inactive nodes.
    """
    N = W.shape[0]

    # Working copy of the correlation matrix
    G = W.copy()

    # Community labels (initially each node is its own community)
    communities = np.arange(N)

    # Sizes of each community
    ns = np.ones(N, dtype=np.float64)

    # Boolean mask indicating active communities
    active = np.ones(N, dtype=np.bool_)

    all_seen = False

    while not all_seen:
        all_seen = True

        # Build list of active communities
        tracker = []
        for i in range(N):
            if active[i]:
                tracker.append(i)

        # Stop if only one community remains
        if len(tracker) <= 1:
            break

        # Shuffle active communities for randomized processing
        tracker_arr = np.array(tracker)
        np.random.shuffle(tracker_arr)

        for community in tracker_arr:

            # Skip if this community was deactivated during iteration
            if not active[community]:
                continue

            best_score = 0.0
            best_pick = -1

            # Current community stats
            A_corr = G[community, community]
            A_size = ns[community]
            A_lc = lc_numba(A_corr, A_size)

            # Evaluate merges with all other active communities
            for pick in range(N):

                if pick == community or not active[pick]:
                    continue

                cross = G[community, pick]

                # Only consider positive interactions
                if cross <= 0:
                    continue

                B_corr = G[pick, pick]
                B_size = ns[pick]
                B_lc = lc_numba(B_corr, B_size)

                # Compute merged contribution
                C_lc = merge_numba(
                    A_corr, B_corr, cross, A_size + B_size
                )

                # Gain from merging
                score = C_lc - (A_lc + B_lc)

                if score > best_score:
                    best_score = score
                    best_pick = pick

            # Apply best merge immediately (greedy step)
            if best_pick != -1 and best_score > 0:

                # Merge adjacency structure
                for k in range(N):
                    if active[k] and k != community and k != best_pick:
                        G[community, k] += G[best_pick, k]
                        G[k, community] += G[best_pick, k]

                # Update internal correlation and size
                G[community, community] += (
                    G[best_pick, best_pick]
                    + 2.0 * G[community, best_pick]
                )
                ns[community] += ns[best_pick]

                # Deactivate merged community
                active[best_pick] = False

                # Update node-to-community assignments
                for k in range(N):
                    if communities[k] == best_pick:
                        communities[k] = community

                # Restart search after merge
                all_seen = False
                break

    # Relabel communities to contiguous integers [0, ..., K-1]
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

    return inverse

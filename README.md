# Agglomerative Likelihood Clustering (ALC) — Fast Local-Greedy (Numba)

This repository implements a fast, randomized **Agglomerative Likelihood Clustering (ALC)** algorithm using a local-greedy merge strategy compiled with Numba.

The method partitions a weighted graph by maximizing a principled likelihood-based objective defined over communities.

---

## Overview

Given a symmetric weight matrix \( W \in \mathbb{R}^{N \times N} \), the goal is to partition nodes into communities such that intra-community connectivity is maximized in a statistically meaningful way.

Unlike standard clustering methods (e.g., modularity or k-means), this approach is based on an **additive likelihood objective**, evaluated per community.

---

## Objective Function

A partition \( \mathcal{P} = \{C_1, \dots, C_K\} \) defines:

- Community size: \( n_C = |C| \)
- Internal correlation:  
  \[
  c_C = \sum_{i,j \in C} W_{ij}
  \]

The total objective is:

\[
\mathcal{L}(\mathcal{P}) = \sum_{C \in \mathcal{P}} \ell(c_C, n_C),
\]

where the per-community contribution is:

\[
\ell(c, n) = \frac{1}{2} \left[
\log\!\left(\frac{n}{c}\right)
+ (n - 1)\log\!\left(\frac{n^2 - n}{n^2 - c}\right)
\right]
\]

defined for:

\[
n > 1, \quad c > n, \quad c < n^2.
\]

---

## Algorithm

The algorithm follows a **randomized local-greedy agglomerative procedure**:

1. Initialize each node as its own community.
2. Iterate until convergence:
   - Randomly shuffle active communities
   - For each community \( A \), evaluate merge candidates \( B \)
   - Compute merge gain:
     \[
     \Delta =
     \ell(c_A + c_B + 2c_{AB},\, n_A + n_B)
     - \big(\ell(c_A, n_A) + \ell(c_B, n_B)\big)
     \]
   - Merge the best pair if \( \Delta > 0 \)
3. Stop when no improving merges exist.

This yields a **locally optimal partition**.

---

## Key Features

- **Fast**: Numba-compiled inner loops
- **Local search**: avoids full \( O(N^2) \) scans per iteration
- **Randomized ordering**: reduces bias from deterministic traversal
- **In-place updates**: efficient memory usage

---

## Usage

```python
from your_module import alc_randomized_fast

# W: symmetric NxN weight (correlation) matrix
labels = alc_randomized_fast(W)

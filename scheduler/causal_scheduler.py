"""Counterfactual PRB scheduling (paper eq. (9), Fig. 4).

A cell serves K UEs and must divide a PRB budget B between them. The paper
poses this as

    max_{t}  sum_u w_u * Yhat_t^(u)     s.t.  sum_u t^(u) <= B,

where Yhat_t^(u) is the UE's *counterfactual* goodput under allocation t —
what that specific UE would achieve, given its own latent channel, device
and traffic state. This module turns that statement into a working
scheduler (dp_allocate: exact search over a per-UE utility table) and the
two non-learned references of Fig. 4.

Policies
--------
equal         B/K to everyone. The incumbent allocation.
proportional  allocate in proportion to each UE's *observed* goodput. This
              is what a KPI-driven xApp does, and it is exactly the
              correlational mistake the paper warns about: a UE looks good
              partly because its neighbour load is low, not because it
              converts PRBs efficiently.
dp_allocate   maximise the summed utility (predicted or true counterfactual
              goodput) under the budget by exact dynamic programming; used
              by the H-SCM, the MLP and the oracle alike.
"""
import numpy as np

K_MIN, K_MAX = 1, 100


def proportional_allocate(observed, budget, k_min=K_MIN, k_max=K_MAX):
    """Allocate proportionally to an observed KPI (the correlational policy)."""
    w = np.clip(np.asarray(observed, dtype=float), 1e-9, None)
    raw = k_min + (budget - len(w) * k_min) * w / w.sum()
    alloc = np.clip(np.round(raw), k_min, k_max).astype(int)
    return _repair(alloc, budget, k_min, k_max)


def equal_allocate(K, budget, k_min=K_MIN, k_max=K_MAX):
    alloc = np.full(K, min(max(budget // K, k_min), k_max), dtype=int)
    return _repair(alloc, budget, k_min, k_max)


def _repair(alloc, budget, k_min, k_max):
    """Nudge an allocation until it exactly meets the budget."""
    alloc = alloc.astype(int).copy()
    while alloc.sum() > budget:
        cand = np.where(alloc > k_min)[0]
        if not len(cand):
            break
        alloc[cand[np.argmax(alloc[cand])]] -= 1
    while alloc.sum() < budget:
        cand = np.where(alloc < k_max)[0]
        if not len(cand):
            break
        alloc[cand[np.argmin(alloc[cand])]] += 1
    return alloc


def dp_allocate(utility, budget, k_min=K_MIN, k_max=K_MAX):
    """EXACT optimum of  max sum_u utility[u, k_u]  s.t.  sum_u k_u <= budget.

    Multiple-choice knapsack by dynamic programming — O(K * B * k_max), which
    the paper quotes as O(KB). Unlike greedy this needs no concavity
    assumption, so it is the correct optimizer for a true oracle: the oracle
    must maximise the *exact* simulator utility, not a smoothed stand-in.

    Using the same routine for the learned utility keeps the comparison
    optimizer-matched: any gap is then attributable to the causal model,
    not to the search.
    """
    K = utility.shape[0]
    NEG = -np.inf
    dp = np.full((K + 1, budget + 1), NEG)
    dp[0, 0] = 0.0
    choice = np.zeros((K, budget + 1), dtype=np.int32)
    for u in range(K):
        for k in range(k_min, min(k_max, budget) + 1):
            if not np.isfinite(utility[u, k]):
                continue
            cand = dp[u, :budget + 1 - k] + utility[u, k]
            better = cand > dp[u + 1, k:]
            dp[u + 1, k:][better] = cand[better]
            choice[u, k:][better] = k
    b = int(np.nanargmax(np.where(np.isfinite(dp[K]), dp[K], NEG)))
    if not np.isfinite(dp[K, b]):
        raise ValueError(f"no feasible allocation: {K} UEs, budget {budget}, "
                         f"k in [{k_min}, {k_max}]")
    alloc = np.zeros(K, dtype=int)
    for u in range(K - 1, -1, -1):
        alloc[u] = choice[u, b]
        b -= alloc[u]
    return alloc



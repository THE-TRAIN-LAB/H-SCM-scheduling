"""Causal discovery building blocks (paper Sec. III-C1).

The discovery experiment (experiments/step11_protocol_discovery.py) runs PC
with protocol background knowledge (causal.protocol_knowledge) and the
stages implemented here:

  find_aliases, find_degenerate   guards that keep Fisher-z defined. A pair
                                  of exact functional copies, or a column
                                  that is an exact linear combination of
                                  earlier ones, makes the correlation matrix
                                  singular and PC cannot run. No column of
                                  the model triggers either guard; both are
                                  kept so that an extended model fails
                                  loudly rather than silently.
  orient                          protocol precedence orients every
                                  cross-layer edge downward (exogenous < L1
                                  < L2 < L3); DirectLiNGAM orients the edges
                                  PC left undirected within a layer.
  interventional_augmentation     direct-effect regressions on pooled
                                  observational and do() rows propose the
                                  cross-layer edges the observational CI
                                  tests cannot keep.
  evaluate                        precision / recall / F1 / SHD against the
                                  ground-truth DAG.

This experiment tests whether the pipeline recovers the DAG that generated
the synthetic data. It does not establish that this DAG is the true causal
structure of a real RAN.
"""
import numpy as np
import pandas as pd

NODE_LAYER = {
    # exogenous = 0
    "cell_load": 0, "shadow_fading": 0, "hw_impairment": 0, "traffic_burst": 0,
    "snr": 0, "doppler": 0, "delay_spread": 0, "eps_sinr": 0, "eps_sched": 0,
    "eps_queue": 0,
    # endogenous, intra-layer rank encoded by list order
    "sinr_eff": 1, "mcs_idx": 1, "mod_order": 1, "code_rate": 1, "bler": 1,
    "se_phy": 1,
    "num_prb": 2, "harq_retx": 2, "mac_tput": 2,
    "queue_dep": 3, "pkt_loss": 3, "goodput": 3,
}
ALIAS_R = 1.0 - 1e-5   # exact functional copies only; near-copies such as
                       # goodput ~ mac_tput (r ~ 0.996) stay in the CI tests

# Parents the base do() grid of config/simulator.yaml forces, keyed by parent
# with the regime pair that contrasts it. interventional_augmentation tests
# these parents when no `targets` are given; the discovery experiment passes
# the parents of its expanded grid instead.
CONTRASTS = {
    "num_prb": ("do(num_prb=10)", "do(num_prb=100)"),
    "cell_load": ("do(cell_load=0.1,num_prb=100)", "do(cell_load=0.9,num_prb=100)"),
    "mcs_idx": ("do(mcs_idx=0)", "do(mcs_idx=11)"),
    "harq_retx": ("do(harq_retx=0)", "obs"),
    "snr": ("do(snr=4)", "do(snr=26)"),
    "shadow_fading": ("do(shadow_fading=-14)", "do(shadow_fading=14)"),
}


def find_aliases(obs, columns):
    """Exact functional copies (|r| > ALIAS_R): return (kept_columns, alias_edges)."""
    corr = obs[columns].corr().abs().values
    aliases, drop = [], set()
    for i, a in enumerate(columns):
        for j, b in enumerate(columns):
            if i < j and corr[i, j] > ALIAS_R:
                parent, child = sorted([a, b], key=lambda v: (NODE_LAYER[v],
                                                              columns.index(v)))
                if child not in drop:
                    aliases.append((parent, child, corr[i, j]))
                    drop.add(child)
    return [c for c in columns if c not in drop], aliases


def find_degenerate(obs, columns, tol=1e-9):
    """Columns that are exact linear combinations of earlier ones.

    find_aliases catches pairwise copies (|r| ~ 1). A column that is an exact
    linear combination of two or more others can have unremarkable pairwise
    correlations while the correlation matrix is still singular, in which
    case Fisher-z raises "Data correlation matrix is singular" and PC cannot
    run. No column of the model is such a combination, so this stage is a
    guard like find_aliases.

    Returns (kept_columns, degenerate) with degenerate a list of
    (column, [predictors], r2). Columns are visited in protocol-precedence
    order, so the dependent variable dropped is always the later one.

    The predictors are reported for diagnostics only and are not turned into
    edges. find_aliases attaches its pairwise copy as an edge, which is
    defensible for an exact 1-1 restatement; doing the same here would hand
    discovery the whole parent set of the column for free. A degenerate
    column is removed from the CI tests and its edges must be earned by the
    interventional stage like any other.
    """
    order = sorted(columns, key=lambda v: (NODE_LAYER[v], columns.index(v)))
    kept, degenerate = [], []
    for c in order:
        if not kept:
            kept.append(c)
            continue
        A = np.column_stack([np.ones(len(obs))] + [obs[k].values.astype(float)
                                                   for k in kept])
        y = obs[c].values.astype(float)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot > 0 and float(np.sum(resid ** 2)) / ss_tot < tol:
            used = [k for k, b in zip(kept, beta[1:]) if abs(b) > 1e-12]
            degenerate.append((c, used, 1.0 - float(np.sum(resid ** 2)) / ss_tot))
        else:
            kept.append(c)
    return [c for c in columns if c in set(kept)], degenerate


def orient(directed, undirected, obs, columns):
    """Precedence + DirectLiNGAM orientation -> fully directed edge list."""
    import lingam
    edges, flipped = [], 0
    for a, b in directed:
        if NODE_LAYER[a] > NODE_LAYER[b]:   # PC oriented against precedence
            edges.append((b, a))
            flipped += 1
        else:
            edges.append((a, b))
    # split undirected: cross-layer -> precedence; intra-layer -> DirectLiNGAM
    lingam_needed = {}
    for a, b in undirected:
        if NODE_LAYER[a] != NODE_LAYER[b]:
            edges.append((a, b) if NODE_LAYER[a] < NODE_LAYER[b] else (b, a))
        else:
            lingam_needed.setdefault(NODE_LAYER[a], []).append((a, b))
    for layer, pairs in lingam_needed.items():
        members = [c for c in columns if NODE_LAYER[c] == layer]
        model = lingam.DirectLiNGAM()
        model.fit(obs[members].values.astype(float))
        order = {members[idx]: pos for pos, idx in enumerate(model.causal_order_)}
        for a, b in pairs:
            edges.append((a, b) if order[a] < order[b] else (b, a))
    return sorted(set(edges)), flipped


def interventional_augmentation(edges, obs, intv, columns, p_add=1e-6,
                                targets=None):
    """Constructive use of the do() regimes (the paper's cross-layer stage).

    Observational PC cannot keep L -> sinr_eff: given num_prb, cell_load has
    almost no residual variance (r(L, prb) = -0.97), so the partial
    correlation dies at alpha = 0.01. Under do(), that collinearity is
    broken. For each manipulable parent X and each downstream candidate Y:
    pool the observational rows with the regimes where X is forced but Y is
    free, regress Y ~ [X + ALL topological predecessors of Y] (protocol
    precedence gives the order, so predecessors include every potential
    mediator), and add X -> Y only when X's coefficient survives at
    p < p_add — a direct-effect test, not a total-effect test.
    Returns (augmented_edges, table).
    """
    import statsmodels.api as sm
    edges = list(edges)
    topo = {c: i for i, c in enumerate(columns)}
    rows = []
    # `targets` defaults to the parents of the base grid (CONTRASTS); the
    # discovery experiment passes the parents its expanded grid forces.
    for x in (CONTRASTS if targets is None else targets):
        if x not in topo:
            continue
        forced_x = intv[intv.regime.str.contains(f"{x}=", regex=False)]
        for y in columns:
            if y == x or NODE_LAYER[y] < NODE_LAYER[x] or (x, y) in edges:
                continue
            pool_intv = forced_x[~forced_x.regime.str.contains(f"{y}=",
                                                               regex=False)]
            if len(pool_intv) < 30:
                continue   # X never forced with Y free -> not testable
            pool = pd.concat([obs, pool_intv], ignore_index=True)
            mediators = [c for c in columns
                         if topo[c] < topo[y] and c != x]
            X = sm.add_constant(pool[[x] + mediators].values.astype(float))
            try:
                fit = sm.OLS(pool[y].values.astype(float), X).fit()
                p = float(fit.pvalues[1])
            except Exception:
                continue
            if p < p_add:
                edges.append((x, y))
                rows.append({"edge": f"{x} -> {y}", "p_value": p,
                             "n_interventional_rows": len(pool_intv)})
    return sorted(set(edges)), pd.DataFrame(rows)


def evaluate(recovered, truth):
    """Skeleton + directed precision/recall/F1 and SHD vs ground truth."""
    rec_d, tru_d = set(recovered), set(truth)
    rec_s = {frozenset(e) for e in rec_d}
    tru_s = {frozenset(e) for e in tru_d}

    def prf(tp, fp, fn):
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        return round(prec, 3), round(rec, 3), round(f1, 3)

    sp, sr, sf = prf(len(rec_s & tru_s), len(rec_s - tru_s), len(tru_s - rec_s))
    dp, dr, df = prf(len(rec_d & tru_d), len(rec_d - tru_d), len(tru_d - rec_d))
    shd = len(rec_s - tru_s) + len(tru_s - rec_s) + sum(
        1 for e in rec_d if frozenset(e) in tru_s and e not in tru_d)
    return {
        "n_true_edges": len(tru_d), "n_recovered_edges": len(rec_d),
        "skeleton": {"precision": sp, "recall": sr, "f1": sf},
        "directed": {"precision": dp, "recall": dr, "f1": df},
        "shd": shd,
        "missing_edges": sorted(f"{a} -> {b}" for a, b in tru_d - rec_d),
        "extra_edges": sorted(f"{a} -> {b}" for a, b in rec_d - tru_d),
        "backdoor_L_to_num_prb": ("cell_load", "num_prb") in rec_d,
        "backdoor_L_to_sinr_eff": ("cell_load", "sinr_eff") in rec_d,
    }

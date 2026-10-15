"""Unit-level counterfactual inference by abduction-action-prediction (paper Sec. III-C3).

    Step 1  Abduction   u_hat_i = x_i (-|/) f_hat_i(pa(x^(u)))   [kind-aware]
    Step 2  Action      remove all edges into T; set T <- t
    Step 3  Prediction  x*_j = f_hat_j(pa*(x*)) (+|*) u_hat_j     in topological
                        order, for descendants of T only

Noise persistence: every abducted u_hat is carried into step 3, and
non-descendants of T keep their factual values — so the unit's channel,
device and traffic fingerprint survive the intervention, while cell load's
path to sinr_eff stays active even when its path to num_prb is severed.

Latent interventions. A latent has no column, so do() on it needs a
realisation. Two kinds are possible:

  proxy realisation      an observed variable is a bijection of the latent,
                         so do(latent = v) is rewritten on that column. No
                         observable in this model inverts any latent, so
                         LATENT_PROXY is empty.

  residual realisation   the latent enters the graph at exactly one node,
                         inside that node's structural noise term, so the
                         node's abducted residual is the latent's fingerprint
                         and the action is applied to the residual itself
                         (LATENT_RESIDUAL, below). This needs no proxy and
                         invents no variable: abduction already recovers
                         u_hat and step 3 re-applies it, so intervening means
                         substituting a different u in that same step. This
                         is what CF-2, do(xi_hw = 0), uses.

Everything here consumes telemetry-mode fits only: the latents are recovered
by abduction, never read from the data.
"""
import os

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from .fitting import IDENTITY, SUPPORT, residual_kind

# Latents realisable on an observed proxy: name -> (column, fn). No observable
# in this model is a bijection of a latent, so this dict is empty and
# do(traffic_burst = v), for instance, is rejected rather than silently
# ignored.
LATENT_PROXY = {}

# Latents realisable on their own abducted residual:
#   name -> (node, u_of(v, cfg))
# The requirement is strict: the latent must enter the graph at exactly one
# node, and it must be the whole of that node's noise term, so that inverting
# the node's residual identifies it. xi_hw qualifies -- the simulator computes
#
#     mac_tput = C0 * num_prb * se_phy * (1 + kappa_hw * xi_hw)
#                / (1 + kappa_harq * harq_retx)
#
# and mac_tput is fitted with a multiplicative residual, so
# u_MAC = mac_tput / f_hat(pa) ~= 1 + kappa_hw * xi_hw. do(xi_hw = v) is then
# u_MAC <- 1 + kappa_hw * v, and do(xi_hw = 0) -- "normalise this device to
# nominal RF quality" -- is simply u_MAC <- 1.
#
# Sign convention: se_eff = se_phy * (1 + kappa_hw * xi_hw), so xi_hw is a
# hardware-quality offset, not an impairment magnitude. Normalising repairs a
# below-nominal device and degrades an above-nominal one; the per-unit effect
# is monotone decreasing in xi_hw and crosses zero at xi_hw = 0.
#
# xi_sf does not qualify even though it also has a single entry point: it is
# confounded with eps_1 inside the sinr_eff residual (u = xi_sf + eps_1), so
# the residual identifies the sum, not the latent.
LATENT_RESIDUAL = {
    "hw_impairment": ("mac_tput",
                      lambda v, cfg: 1.0 + float(cfg["mac"]["kappa_hw"]) * v),
}


def load_graph(root, edges=None, nodes=None):
    """Graph used for the topological order and for descendant sets.

    edges=None reads the ground-truth DAG. Passing a recovered edge list makes
    do() propagate over the graph that was actually discovered -- so a missed
    edge means a descendant that never gets recomputed, which is the failure
    mode an end-to-end evaluation has to expose rather than hide.
    """
    if edges is None:
        with open(os.path.join(root, "config", "dag_edges.yaml")) as f:
            edges = [tuple(e) for e in yaml.safe_load(f)["edges"]]
    g = nx.DiGraph()
    if nodes is not None:
        g.add_nodes_from(nodes)          # keep isolated nodes addressable
    g.add_edges_from(edges)
    return g


class CounterfactualEngine:
    """Abduction-action-prediction over a fitted StructuralModel (telemetry mode)."""

    def __init__(self, model, root, edges=None, nodes=None):
        self.model = model
        if edges is None:
            edges = getattr(model, "edges", None)     # inherit the model's graph
        self.g = load_graph(root, edges=edges, nodes=nodes)
        self.order = [n for n in nx.topological_sort(self.g)
                      if n in model.parents]
        # LATENT_RESIDUAL needs the coupling constant kappa_hw; it is read
        # from the configuration, never hardcoded
        with open(os.path.join(root, "config", "simulator.yaml")) as f:
            self.cfg = yaml.safe_load(f)
        # per-node count of values a support projection had to move; a
        # non-zero entry is a diagnostic, not a success (see SUPPORT)
        self.projected = {}

    def _bounds(self, node):
        """Physical support of `node` (see SUPPORT in causal.fitting)."""
        return SUPPORT.get(node, (None, None))

    # -- step 1 -----------------------------------------------------------
    def abduct(self, factual, fold_of=None):
        return self.model.residuals(factual, fold_of=fold_of)

    def _valid_do_targets(self):
        """Names do() can act on: fitted nodes, observable features, latents
        with a declared proxy, or latents realisable on a fitted node's
        residual. A latent with NONE of those (e.g. shadow_fading) would
        silently change nothing — no telemetry model reads it — so it must be
        rejected, not ignored."""
        observable = set(self.model.parents)
        for pa in self.model.parents.values():
            observable |= set(pa)
        residual_ok = {name for name, (node, _) in LATENT_RESIDUAL.items()
                       if node in self.model.parents}
        return observable | set(LATENT_PROXY) | residual_ok

    # -- steps 2 + 3 -------------------------------------------------------
    def predict(self, factual, do, residuals, fold_of=None):
        """Counterfactual frame for the units in `factual` under `do`."""
        do = dict(do)
        valid = self._valid_do_targets()
        bad = set(do) - valid
        if bad:
            raise ValueError(
                f"do() target(s) {sorted(bad)} are latent without a proxy "
                f"(or unknown) — the telemetry-mode SCM cannot realise them. "
                f"Valid targets: {sorted(valid)}")
        cf = factual.copy()
        res = dict(residuals)        # local copy: residual actions edit it

        # step 2: realise the action. Value targets have their column pinned
        # and are NOT recomputed; residual targets keep their equation and
        # have their NOISE replaced, so the node itself IS recomputed.
        pinned, affected = set(), set()
        for name, val in do.items():
            if name in LATENT_RESIDUAL:
                node, u_of = LATENT_RESIDUAL[name]
                res[node] = np.full(len(cf), float(u_of(val, self.cfg)))
                affected |= {node} | nx.descendants(self.g, node)
            elif name in LATENT_PROXY:
                proxy, fn = LATENT_PROXY[name]
                cf[proxy] = fn(val)
                pinned.add(proxy)
                affected |= nx.descendants(self.g, proxy)
            else:
                cf[name] = val
                pinned.add(name)
                affected |= nx.descendants(self.g, name)
        affected -= pinned

        # step 3: recompute in topological order, re-applying abducted noise
        # (every node keeps ITS OWN factual residual — only the intervened
        # latent's is replaced; zeroing them all would be a different query)
        for node in self.order:
            if node not in affected:
                continue                      # noise persistence
            kind = residual_kind(node)
            if kind == "identity":
                # Computed from its parents, never fitted-plus-residual, so
                # the definition holds exactly in the counterfactual world too.
                cf[node] = IDENTITY[node](cf)
                continue
            f = self.model.predict_node(node, cf, fold_of=fold_of)
            if kind == "multiplicative":
                cf[node] = f * res[node]
            elif kind == "carry":
                cf[node] = res[node]          # own latent state, unchanged
            elif kind == "scaled":
                # The residual is a(pa)*(xi - m) + eps: one part scales with
                # the mechanism's own mean, one does not. Carrying it
                # unchanged keeps the traffic term at the factual utilization,
                # which is wrong whenever the action moves the service rate --
                # and wrong in both directions, so clipping cannot repair it.
                # Rescale the share that scales, carry the share that doesn't.
                f0 = self.model.predict_node(node, factual, fold_of=fold_of)
                w = self.model.scale_weight(node, f0, fold_of=fold_of)
                ratio = f / np.maximum(f0, 1e-9)
                cf[node] = f + (w * ratio + (1.0 - w)) * res[node]
            else:
                cf[node] = f + res[node]
            lo, hi = self._bounds(node)
            if lo is None and hi is None:
                continue
            v = np.asarray(cf[node], dtype=float)
            n_bad = 0
            if lo is not None:
                n_bad += int((v < lo).sum())
            if hi is not None:
                n_bad += int((v > hi).sum())
            self.projected[node] = self.projected.get(node, 0) + n_bad
            cf[node] = np.clip(v, lo, hi)
        return cf

    def query(self, factual, do, fold_of=None):
        """Convenience: abduct then predict."""
        return self.predict(factual, do,
                            self.abduct(factual, fold_of=fold_of),
                            fold_of=fold_of)


def latent_fingerprints(residuals, factual, kappa_hw):
    """Physical latents recovered by abduction (never read from the data).

    xi_sf_hat : residual of sinr_eff (= xi_sf + eps_1 by eq. (1))
    xi_tb_hat : not recoverable. xi_tb enters only queue_dep, where it is
                confounded with eps_3, and no observable inverts it.
    xi_hw_hat : from the multiplicative mac_tput residual, (r - 1)/kappa_hw
                (kappa_hw from config mac.kappa_hw). This is the inverse of
                LATENT_RESIDUAL["hw_impairment"] and the stratifier of CF-2.

    Caveat: u_MAC absorbs f_hat's own error as well as the hardware state, so
    xi_hw_hat is over-dispersed, and selecting a subpopulation on it selects
    partly on estimation error; the estimated gain in such a bin is
    optimistic. Subpopulations are therefore defined from telemetry — see
    peer_underperformers().
    """
    out = {"xi_sf_hat": residuals["sinr_eff"],
           }
    if "mac_tput" in residuals:
        out["xi_hw_hat"] = (residuals["mac_tput"] - 1.0) / float(kappa_hw)
    return out


def peer_underperformers(frame, ratio=0.95, prb_bins=10):
    """Observable P_2 for CF-2: UEs delivering less than `ratio` x the median
    service rate of peers in the same radio state (same MCS index, same PRB
    decile).

    Paper Sec. II defines every subpopulation P_k through observable
    constraints, and xi_hw_hat is a latent. This rule uses only mcs_idx,
    num_prb and mac_tput — all telemetry — and no fitted model, so it is
    computable at a live gNB. Holding the radio state fixed is what makes it
    select device-limited UEs: a coarser rule such as "good SINR, plenty of
    PRBs, low goodput" mostly selects the low end of the MCS range, since
    se_phy still spans the whole MCS table within a SINR half-plane. Not
    selecting on the model residual also avoids the winner's curse described
    in latent_fingerprints().
    """
    prb = pd.qcut(frame.num_prb, prb_bins, duplicates="drop")
    med = frame.groupby([frame.mcs_idx, prb],
                        observed=True).mac_tput.transform("median")
    return (frame.mac_tput / med) < ratio


def cf_statistics(factual_y, cf_y, name=""):
    """KS test, Wasserstein-1 and means for a counterfactual query."""
    from scipy import stats
    ks = stats.ks_2samp(factual_y, cf_y)
    return {"query": name, "n": len(factual_y),
            "factual_mean": float(np.mean(factual_y)),
            "cf_mean": float(np.mean(cf_y)),
            "delta_mean": float(np.mean(cf_y) - np.mean(factual_y)),
            "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
            "wasserstein_1": float(stats.wasserstein_distance(factual_y, cf_y))}


def stratify(values, effect, edges, labels):
    """Per-stratum mean effect (used for the paper's spread ratios)."""
    rows = []
    for (lo, hi), lab in zip(edges, labels):
        sel = (values >= lo) & (values < hi)
        rows.append({"stratum": lab, "n": int(sel.sum()),
                     "mean_effect": float(np.mean(effect[sel])) if sel.any()
                     else np.nan})
    return pd.DataFrame(rows)

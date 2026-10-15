"""Structural fitting and counterfactual inference on the recovered graph
(paper Sec. III-C2, III-C3 and IV: the held-out R^2 values and Fig. 3).

Discovery is run for one realization (seed 42); the recovered graph supplies
the parent sets for structural fitting (eq. 11) and the causal order for
abduction-action-prediction. Two fitted models: a full-data fit for the
held-out validation against the simulator's per-UE outcomes on the 5000 oracle
units, and a cross-fitted one for CF-1/2/3 so that no UE is abducted by a
model that saw it.

Inputs: data/observational.csv, data/oracle_paired.csv.gz, config/*.yaml.
Outputs: results/step13_val.csv (held-out R^2 per arm), step13_queries.csv
(CF-1/2/3), step13_strata.csv (mean gain by inferred residual range),
step13_cf_arrays.npz (per-UE factual and counterfactual goodput).

Usage: python experiments/step13_counterfactuals.py [--seed 42]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from causal.counterfactual import (CounterfactualEngine,             # noqa: E402
                                   cf_statistics, latent_fingerprints,
                                   peer_underperformers, stratify)
from causal.discovery import evaluate                                # noqa: E402
from causal.fitting import StructuralModel, observable_edges         # noqa: E402
from simulator.ran_simulator import COLUMNS, ENDOGENOUS, RANSimulator  # noqa: E402
from step11_protocol_discovery import (discover, reference_levels,   # noqa: E402
                                       targeted_regimes)

ROOT = os.path.join(os.path.dirname(__file__), "..")
HW_EDGES = [(-99, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 99)]
HW_LABELS = ["xi_hw<-1.5", "-1.5..-0.5", "-0.5..0.5", "0.5..1.5", ">1.5"]
TREATMENT, OUTCOME = "num_prb", "goodput"


def rung3(obs, oracle, edges, kappa_hw):
    """Held-out validation and CF-1/2/3. Returns (validation, queries,
    strata, per-UE arrays)."""
    # Two models: a full-data fit for the held-out oracle validation, and a
    # cross-fitted one for CF-1/2/3 so no UE is abducted by a model that saw it.
    sm_full = StructuralModel(ROOT, mode="telemetry", edges=edges,
                              nodes=ENDOGENOUS).fit(obs, cv_folds=3)
    eng_full = CounterfactualEngine(sm_full, ROOT, edges=edges, nodes=COLUMNS)
    sm = StructuralModel(ROOT, mode="telemetry", edges=edges,
                         nodes=ENDOGENOUS).fit_crossfit(obs, folds=5)
    eng = CounterfactualEngine(sm, ROOT, edges=edges, nodes=COLUMNS)
    fold = sm.fold_of
    res = eng.abduct(obs, fold_of=fold)
    fps = latent_fingerprints(res, obs, kappa_hw)
    sf_hat = fps["xi_sf_hat"]
    hw_hat = fps["xi_hw_hat"]         # CF-2's stratifier

    # -- learned vs oracle on the 5000 held-out UEs --------------------
    fact = oracle[oracle.arm == "factual"].reset_index(drop=True)
    ores = eng_full.abduct(fact)
    val = []
    for arm, (do, col) in {"do(num_prb=10)": ({"num_prb": 10}, "goodput"),
                           "do(num_prb=50)": ({"num_prb": 50}, "goodput"),
                           "do(num_prb=100)": ({"num_prb": 100}, "goodput"),
                           "do(hw_impairment=0)": ({"hw_impairment": 0.0},
                                                   "goodput")}.items():
        truth = oracle[oracle.arm == arm].reset_index(drop=True)[col].values
        pred = eng_full.predict(fact, do, ores)[col].values
        err = pred - truth
        val.append({"arm": arm, "outcome": col,
                    "r2": 1 - np.sum(err**2) / np.sum((truth - truth.mean())**2),
                    "corr": float(np.corrcoef(pred, truth)[0, 1]),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "oracle_effect": float(np.mean(truth) - fact[col].mean())})

    # -- CF-1/2/3 -----------------------------------------------------
    out, strata, arrays = [], [], {}

    def cf(mask, do, col, name, key, edges_, labels):
        c = eng.predict(obs[mask], do,
                        {k: v[mask.values] for k, v in res.items()},
                        fold_of=fold[mask.values])
        s = cf_statistics(obs.loc[mask, col].values, c[col].values, name)
        s["outcome"] = col          # the figure reads the unit from here
        out.append(s)
        arrays[f"{name}_fact"] = obs.loc[mask, col].values
        arrays[f"{name}_cf"] = c[col].values
        delta = c[col].values - obs.loc[mask, col].values
        st = stratify(key[mask.values], delta, edges_, labels)
        st["query"] = name
        strata.append(st)

    cf((obs.goodput >= 5) & (obs.goodput <= 40), {"num_prb": 100}, "goodput",
       "CF-1", sf_hat, [(-99, -7), (-7, -3), (-3, 3), (3, 7), (7, 99)],
       ["xi_sf<-7", "-7..-3", "-3..3", "3..7", ">7"])
    cf(peer_underperformers(obs), {"hw_impairment": 0.0},
       "goodput", "CF-2", hw_hat, HW_EDGES, HW_LABELS)
    cf((obs.snr >= 10) & (obs.snr <= 20) & (obs.goodput < 15),
       {"num_prb": 100}, "goodput", "CF-3", sf_hat,
       [(-99, -5), (-5, 5), (5, 99)],
       ["xi_sf<-5 (channel-limited)", "-5..5", ">+5 (PRB-limited)"])
    return (pd.DataFrame(val), pd.DataFrame(out),
            pd.concat(strata, ignore_index=True), arrays)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "simulator.yaml")))
    # scored against the observable subgraph, as in step11
    truth = observable_edges([tuple(e) for e in yaml.safe_load(
        open(os.path.join(ROOT, "config", "dag_edges.yaml")))["edges"]])
    sim = RANSimulator(cfg)
    kappa_hw = float(cfg["mac"]["kappa_hw"])

    lv = reference_levels(sim)
    rec = sorted(discover(sim, cfg, a.seed, lv, targeted_regimes(lv), 60)[0])
    m = evaluate(rec, truth)
    print(f"recovered graph (seed {a.seed}): {len(rec)} edges, "
          f"{len(set(rec) & set(truth))} correct, "
          f"{len(set(rec) - set(truth))} spurious "
          f"(P {m['directed']['precision']:.3f} / R {m['directed']['recall']:.3f})")

    obs = pd.read_csv(os.path.join(ROOT, "data", "observational.csv"))
    oracle = pd.read_csv(os.path.join(ROOT, "data", "oracle_paired.csv.gz"))

    vr, cr, sr, ar = rung3(obs, oracle, rec, kappa_hw)
    np.savez(os.path.join(ROOT, "results", "step13_cf_arrays.npz"),
             **ar)

    print("\n  learned vs oracle (5000 held-out UEs)")
    print(f"  {'arm':<24}{'R2':>10}{'corr':>10}{'RMSE':>10}")
    print("  " + "-" * 54)
    for _, x in vr.iterrows():
        print(f"  {x['arm']:<24}{x['r2']:>10.3f}{x['corr']:>10.3f}{x['rmse']:>10.3f}")
    print("\n  CF-1/2/3 on the 500 observational UEs")
    print(f"  {'query':<7}{'n':>5}{'factual':>10}{'CF':>10}{'delta':>10}{'KS':>8}")
    print("  " + "-" * 50)
    for _, x in cr.iterrows():
        print(f"  {x['query']:<7}{int(x['n']):>5}{x['factual_mean']:>10.2f}{x['cf_mean']:>10.2f}"
              f"{x['cf_mean'] - x['factual_mean']:>+10.2f}{x['ks_stat']:>8.2f}   Mbps")

    vr.to_csv(os.path.join(ROOT, "results", "step13_val.csv"), index=False)
    cr.to_csv(os.path.join(ROOT, "results", "step13_queries.csv"), index=False)
    sr.to_csv(os.path.join(ROOT, "results", "step13_strata.csv"), index=False)
    print("\nwrote results/step13_val.csv, step13_queries.csv, step13_strata.csv, "
          "step13_cf_arrays.npz")

if __name__ == "__main__":
    main()

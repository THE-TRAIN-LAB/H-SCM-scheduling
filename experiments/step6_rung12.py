"""Population estimates of E[Y | do(T = t)] (paper Fig. 2).

naive OLS      mu_obs(t) = E[Y | T = t], OLS of goodput on num_prb alone
adjusted OLS   mu_int(t) = E_L[E[Y | T = t, L]] under a linear outcome model
H-SCM          the mean of the per-UE counterfactuals Y_t(u) of the cross-fitted
               H-SCM on the recovered graph (the model behind Fig. 3)
oracle         mean goodput under do(num_prb = t) on the 5000 paired oracle
               units (data/oracle_paired.csv.gz)

All estimators use the 500 observational UEs only. Requires
results/step11_recovered_dag.yaml (run step11 first). Outputs
results/rung2_dose_response.csv, read by plot_confounding_recovered.py.
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from causal.counterfactual import CounterfactualEngine                 # noqa: E402
from causal.fitting import StructuralModel                             # noqa: E402
from simulator.ran_simulator import COLUMNS, ENDOGENOUS                # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
T_GRID = [10, 50, 100]


def naive_ols(obs, t):
    b, a = np.polyfit(obs.num_prb, obs.goodput, 1)
    return a + b * t


def adjusted_ols(obs, t):
    """Backdoor adjustment E_L[E[Y|T=t,L]] under a linear outcome model."""
    X = np.column_stack([np.ones(len(obs)), obs.num_prb, obs.cell_load])
    beta, *_ = np.linalg.lstsq(X, obs.goodput.values, rcond=None)
    return beta[0] + beta[1] * t + beta[2] * obs.cell_load.mean()


def hscm_engine(obs):
    """The cross-fitted H-SCM of Fig. 3 (recovered graph, telemetry mode)."""
    edges = [tuple(e) for e in yaml.safe_load(open(os.path.join(
        ROOT, "results", "step11_recovered_dag.yaml")))["recovered_edges"]]
    sm = StructuralModel(ROOT, mode="telemetry", edges=edges,
                         nodes=ENDOGENOUS).fit_crossfit(obs, folds=5)
    eng = CounterfactualEngine(sm, ROOT, edges=edges, nodes=COLUMNS)
    return eng, eng.abduct(obs, fold_of=sm.fold_of), sm.fold_of


def main():
    obs = pd.read_csv(os.path.join(ROOT, "data", "observational.csv"))
    oracle = pd.read_csv(os.path.join(ROOT, "data", "oracle_paired.csv.gz"))
    print(f"r(cell_load, num_prb) = {np.corrcoef(obs.cell_load, obs.num_prb)[0, 1]:+.3f}\n")
    eng, res, fold = hscm_engine(obs)
    rows = []
    for t in T_GRID:
        big = oracle.loc[oracle.arm == f"do(num_prb={t})", "goodput"]
        mu_n, mu_a, truth = naive_ols(obs, t), adjusted_ols(obs, t), big.mean()
        mu_h = float(eng.predict(obs, {"num_prb": int(t)}, res,
                                 fold_of=fold).goodput.mean())
        rows.append({"t": t, "naive_ols": mu_n, "adjusted_ols": mu_a, "hscm": mu_h,
                     "oracle_hiN": truth, "oracle_hiN_se": big.std() / np.sqrt(len(big)),
                     "bias_naive_pct": 100 * (mu_n - truth) / truth,
                     "bias_adj_pct": 100 * (mu_a - truth) / truth,
                     "bias_hscm_pct": 100 * (mu_h - truth) / truth})
    dose = pd.DataFrame(rows)
    dose.to_csv(os.path.join(ROOT, "results", "rung2_dose_response.csv"), index=False)
    for r in dose.itertuples(index=False):
        print(f"t={r.t:3d}: naive {r.naive_ols:6.2f} ({r.bias_naive_pct:+.1f}%)  "
              f"adjusted {r.adjusted_ols:6.2f} ({r.bias_adj_pct:+.1f}%)  "
              f"H-SCM {r.hscm:6.2f} ({r.bias_hscm_pct:+.1f}%)  "
              f"oracle {r.oracle_hiN:6.2f} (SE {r.oracle_hiN_se:.3f})")
    print("\nwrote results/rung2_dose_response.csv")


if __name__ == "__main__":
    main()

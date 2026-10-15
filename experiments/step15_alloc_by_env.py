"""Counterfactual PRB scheduling (paper eq. (9), Fig. 4).

Four UEs share a cell's load-dependent PRB budget B(L) = floor(N_PRB (1 - alpha_L L))
with at least k_min PRBs each. Two learned policies are trained once on the same
exploration log of the nominal environment and then evaluated, without
refitting, in four environments that shift only the root-variable distributions.

The exploration log: LOG_CELLS nominal cells x 4 UEs, each observed in two
scheduling epochs with the same context (load, radio link, device and traffic
factors) and fresh mechanism noise -- epoch 1 under the incumbent equal split,
epoch 2 under a random split of the budget. It is the kind of log an exploring
scheduler produces.

  hscm_struct    the H-SCM: structural mechanisms on the recovered graph
                 (results/step11_recovered_dag.yaml), the MAC service rate
                 fitted in the proportional-in-T form mu = T * g(parents), fitted
                 on the log's rows as unpaired samples. At decision time each
                 UE's residual state is abducted from the telemetry realized
                 under the incumbent allocation and carried into the prediction
                 for every candidate allocation.
  noncausal_mlp  the MLP (PyTorch, two hidden layers of 64 units): a direct
                 predictor of next-epoch goodput from the UE's full telemetry
                 vector of the current epoch (all 16 observable variables,
                 post-allocation ones included) and the candidate allocation,
                 trained on the log's epoch pairs. It therefore holds exactly
                 the measurements the H-SCM uses; re-fitted at four more
                 initialisations (noncausal_mlp_rs*) so the reported interval
                 covers training noise.

Two rung-2 references isolate what the counterfactual rung adds:

  hscm_rung2     the same fitted H-SCM mechanisms and the same search, but
                 abduction skipped: every residual is set to its population
                 default (0 for additive residuals, 1 for the multiplicative MAC
                 residual), so the prediction is the conditional causal response
                 E[Y | do(T=t), x] through the UE's observed telemetry. The only
                 difference from hscm_struct is the counterfactual rung.
  cond_gbm       the conventional data-driven scheduler of paper eq. (6): a GBM
                 regression of goodput on (T, x) with x = [L, gamma0, fD, tau_rms]
                 (the context the scheduler observes), fitted on the same log.

All learned policies optimize their predicted goodput by exact search (dynamic
programming over the budget). Two non-learned references: proportional (PRBs
proportional to observed goodput) and equal (equal split, the incumbent). The
oracle optimizes the simulator's true counterfactual goodput and is the upper
bound. Every allocation is executed in the ground-truth simulator; the score
is the realized total cell goodput relative to the oracle's.

Inputs: config/simulator.yaml, results/step11_recovered_dag.yaml.
Outputs results/step15_alloc_by_env.csv (per environment and realization),
read by plot_alloc_by_env.py.
"""
import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd
import yaml
import torch
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from causal.counterfactual import CounterfactualEngine            # noqa: E402
from causal.fitting import (GBM_PARAMS, StructuralModel,          # noqa: E402
                            observable, residual_kind)
from scheduler.causal_scheduler import (dp_allocate,              # noqa: E402
                                        equal_allocate,
                                        proportional_allocate)
from simulator.ran_simulator import COLUMNS, RANSimulator         # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")

# Evaluation environments: each is a pure covariate shift of the root-variable
# distributions in config/simulator.yaml; no structural constant is touched.
# Each override is "<section>.<key>[.<subkey>] : value". Nominal values are in
# the comment so the size of every shift is visible in one place.
ENVIRONMENTS = {
    "E0_nominal": {},
    "E1_high_load": {                       # more congestion, burstier traffic
        "observational.cell_load.a": 5.0,           # Beta(2,2) -> Beta(5,2)
        "observational.cell_load.b": 2.0,           #   mean 0.50 -> 0.71
        "observational.traffic_burst.sigma2": 0.30,  # 0.12 -> 0.30 (burstier)
    },
    "E2_poor_channel": {                    # same supports, degraded half
        "observational.snr_db.high": 16.0,           # U(2,28) -> U(2,16)
        "observational.shadow_fading_db.var": 81.0,  # sd 7 -> 9 dB
        "observational.doppler_hz.low": 100.0,       # U(5,200) -> U(100,200)
        "observational.delay_spread_ns.low": 165.0,  # U(30,300) -> U(165,300)
    },
    "E3_heterogeneous": {                   # same means, much wider spread
        "observational.hw_impairment.var": 9.0,      # sd 1.5 -> 3.0
        "observational.shadow_fading_db.var": 100.0,  # sd 7 -> 10 dB
        "observational.traffic_burst.sigma2": 0.40,  # 0.12 -> 0.40
    },
}


def env_cfg(base, overrides):
    """Config overlay: only the root-variable distributions in ENVIRONMENTS move."""
    cfg = copy.deepcopy(base)
    for path, val in overrides.items():
        node, *rest = path.split(".")
        d = cfg[node]
        for k in rest[:-1]:
            d = d[k]
        d[rest[-1]] = val
    return cfg

OBS = observable(COLUMNS)                 # the 16 telemetry variables
CONTEXT = ["cell_load", "snr", "doppler", "delay_spread"]   # x^(u) of eq. (6)
NOISE = ["eps_sinr", "eps_sched", "eps_queue"]
LOG_CELLS, LOG_SEED = 500, 2026           # exploration log: 2000 UEs x 2 epochs
MLP_INITS = (43, 44, 45, 46)            # extra initialisations of the MLP
MLP_ARMS = ["noncausal_mlp"] + [f"noncausal_mlp_rs{r}" for r in MLP_INITS]
METHODS = (["oracle", "hscm_struct", "hscm_rung2", "cond_gbm", "noncausal_mlp",
            "proportional", "equal"]
           + [f"noncausal_mlp_rs{r}" for r in MLP_INITS])


def neutral(res):
    """Population-default residuals: the rung-2 reading of the same SCM."""
    return {k: (np.ones_like(v) if residual_kind(k) == "multiplicative"
                else np.zeros_like(v)) for k, v in res.items()}


class TorchMLP:
    """MLP regressor in PyTorch: standardized inputs, two hidden layers of
    64 ReLU units, Adam (lr 1e-3),
    L2 penalty alpha = 1e-4 on the weights (not the biases), mini-batches of
    200, at most 3000 epochs, and a plateau rule (stop when the epoch
    loss has not improved by tol = 1e-4 for 10 consecutive epochs). Weights
    and biases start Glorot-uniform; `seed` fixes initialisation and shuffling."""

    def __init__(self, hidden=(64, 64), lr=1e-3, alpha=1e-4, batch_size=200,
                 max_iter=3000, tol=1e-4, n_iter_no_change=10, seed=42):
        self.hidden, self.lr, self.alpha = hidden, lr, alpha
        self.batch_size, self.max_iter = batch_size, max_iter
        self.tol, self.n_iter_no_change, self.seed = tol, n_iter_no_change, seed

    def fit(self, X, y):
        torch.set_num_threads(2)
        g = torch.Generator().manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-12
        Xt = torch.from_numpy((X - self.mu_) / self.sd_); yt = torch.from_numpy(y)
        sizes = [X.shape[1], *self.hidden, 1]; layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            lin = torch.nn.Linear(a, b); bound = float(np.sqrt(6.0 / (a + b)))
            with torch.no_grad():
                lin.weight.uniform_(-bound, bound, generator=g)
                lin.bias.uniform_(-bound, bound, generator=g)
            layers.append(lin)
            if i < len(self.hidden): layers.append(torch.nn.ReLU())
        self.net_ = torch.nn.Sequential(*layers)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        n = len(Xt); bs = min(self.batch_size, n); best, stale = np.inf, 0
        weights = [m.weight for m in self.net_ if isinstance(m, torch.nn.Linear)]
        for _ in range(self.max_iter):
            perm = torch.randperm(n, generator=g); total = 0.0
            for s in range(0, n, bs):
                idx = perm[s:s + bs]
                pred = self.net_(Xt[idx]).squeeze(1)
                loss = 0.5 * torch.mean((pred - yt[idx]) ** 2) \
                    + 0.5 * self.alpha * sum((w ** 2).sum() for w in weights) / n
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(idx)
            epoch_loss = total / n
            if epoch_loss > best - self.tol: stale += 1
            else: stale = 0
            best = min(best, epoch_loss)
            if stale >= self.n_iter_no_change: break
        self.n_iter_ = _ + 1
        return self

    def predict(self, X):
        X = (np.asarray(X, dtype=np.float32) - self.mu_) / self.sd_
        with torch.no_grad():
            return self.net_(torch.from_numpy(X)).squeeze(1).numpy().astype(float)


def cell_budget(cfg, L, n_ues, k_min):
    base = int(cfg["mac"]["base_prb"])
    budget = max(int(np.floor(base * (1.0 - float(cfg["mac"]["alpha_L"]) * L))),
                 n_ues * k_min)                      # eq. (2) without noise
    return budget, min(budget - (n_ues - 1) * k_min, base)


def run_alloc(sim, u, alloc):
    """Execute a per-UE PRB vector in the simulator for the units in u."""
    return pd.concat([sim.propagate({k: v[[i]] for k, v in u.items()},
                                    do={"num_prb": int(alloc[i])})
                      for i in range(len(alloc))], ignore_index=True)


def redraw_noise(sim, u, rng):
    """Same context, fresh mechanism noise: the next scheduling epoch."""
    fresh = sim.sample_exogenous(len(u["cell_load"]), rng)
    u2 = {k: v.copy() for k, v in u.items()}
    for k in NOISE:
        u2[k] = fresh[k]
    return u2


def build_log(sim, cfg, n_cells, n_ues, k_min, rng):
    """Two-epoch exploration log of nominal cells (see module docstring)."""
    e1, e2 = [], []
    for _ in range(n_cells):
        u = sim.sample_exogenous(n_ues, rng)
        o = cfg["observational"]["cell_load"]
        L = float(rng.beta(o["a"], o["b"]))
        u["cell_load"] = np.full(n_ues, L)
        budget, k_max = cell_budget(cfg, L, n_ues, k_min)
        a1 = equal_allocate(n_ues, budget, k_min, k_max)           # incumbent
        w = rng.dirichlet(np.full(n_ues, 0.5))                      # exploration
        a2 = np.minimum(k_min + np.floor(w * (budget - n_ues * k_min)).astype(int), k_max)
        e1.append(run_alloc(sim, u, a1))
        e2.append(run_alloc(sim, redraw_noise(sim, u, rng), a2))
    e1 = pd.concat(e1, ignore_index=True); e2 = pd.concat(e2, ignore_index=True)
    e1["regime"], e2["regime"] = "obs", "do(num_prb=explore)"
    return e1, e2


def frozen_models(cfg, n_ues, k_min):
    """Fit both learned policies ONCE, on the same exploration log."""
    sim = RANSimulator(cfg)
    e1, e2 = build_log(sim, cfg, LOG_CELLS, n_ues, k_min,
                       np.random.default_rng(LOG_SEED))
    pool = pd.concat([e1, e2], ignore_index=True)        # the log's rows, unpaired
    edges = [tuple(e) for e in yaml.safe_load(open(os.path.join(
        ROOT, "results", "step11_recovered_dag.yaml")))["recovered_edges"]]
    sm = StructuralModel(ROOT, mode="telemetry", edges=edges,
                         nodes=OBS, factored=True).fit(pool, cv_folds=5)   # mu = T * g_hat
    X = np.column_stack([e1[OBS].values.astype(float),       # epoch-1 telemetry
                         e2.num_prb.values.astype(float)])   # + next allocation
    y = e2.goodput.values                                    # -> next goodput
    Xc = pool[["num_prb"] + CONTEXT].values.astype(float)     # eq. (6): (T, x)
    return {
        "hscm_struct": CounterfactualEngine(sm, ROOT, edges=edges, nodes=COLUMNS),
        "cond_gbm": GradientBoostingRegressor(**GBM_PARAMS).fit(Xc, pool.goodput.values),
        "noncausal_mlp": TorchMLP(seed=42).fit(X, y),
        **{f"noncausal_mlp_rs{r}": TorchMLP(seed=r).fit(X, y) for r in MLP_INITS},
    }, e1, e2


def one_cell(sim, M, cfg, rng, n_ues, k_min):
    o = cfg["observational"]["cell_load"]
    u = sim.sample_exogenous(n_ues, rng)
    L = float(rng.beta(o["a"], o["b"]))
    u["cell_load"] = np.full(n_ues, L)          # one cell -> one load state

    budget, k_max = cell_budget(cfg, L, n_ues, k_min)
    if k_max < k_min:
        return None

    # the incumbent allocation the cell is actually running
    fact_alloc = equal_allocate(n_ues, budget, k_min, k_max)
    fact = run_alloc(sim, u, fact_alloc)
    def util(fn):
        t = np.full((n_ues, k_max + 1), -np.inf)
        for k in range(k_min, k_max + 1):
            t[:, k] = fn(k)
        return t

    tables = {"oracle": util(
        lambda k: sim.propagate(u, do={"num_prb": int(k)}).goodput.values)}
    eng = M["hscm_struct"]
    res = eng.abduct(fact)                 # UE state from the incumbent world
    tables["hscm_struct"] = util(
        lambda k: eng.predict(fact, {"num_prb": int(k)}, res).goodput.values)
    res0 = neutral(res)                    # rung 2: same mechanisms, no abduction
    tables["hscm_rung2"] = util(
        lambda k: eng.predict(fact, {"num_prb": int(k)}, res0).goodput.values)
    xctx = fact[CONTEXT].values.astype(float)
    tables["cond_gbm"] = util(lambda k: M["cond_gbm"].predict(
        np.column_stack([np.full(n_ues, float(k)), xctx])))
    # the MLP reads the UE's full telemetry vector of the incumbent epoch and
    # the candidate allocation -- the same measurements the H-SCM abducts from
    xfact = fact[OBS].values.astype(float)
    for name in MLP_ARMS:
        tables[name] = util(lambda k, m=M[name]: m.predict(
            np.column_stack([xfact, np.full(n_ues, float(k))])))

    alloc = {n: dp_allocate(t, budget, k_min, k_max) for n, t in tables.items()}
    alloc["proportional"] = proportional_allocate(fact.goodput.values, budget,
                                                  k_min, k_max)
    alloc["equal"] = fact_alloc
    # every policy is EXECUTED in the ground-truth simulator; J is realized,
    # never the utility the policy believed it would get
    out = {"budget": budget, "cell_load": L}
    for name, a in alloc.items():
        out[f"J_{name}"] = float(run_alloc(sim, u, a).goodput.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--cells", type=int, default=25)
    ap.add_argument("--ues", type=int, default=4)
    ap.add_argument("--kmin", type=int, default=10)
    a = ap.parse_args()

    base = yaml.safe_load(open(os.path.join(ROOT, "config", "simulator.yaml")))
    M, e1, e2 = frozen_models(base, a.ues, a.kmin)
    print(f"Policies trained once on the nominal exploration log: {LOG_CELLS} cells x {a.ues} UEs "
          f"x 2 epochs = {len(e1) + len(e2)} rows ({len(e2)} epoch pairs); "
          f"PRB {e1.num_prb.min()}-{e1.num_prb.max()} (epoch 1), "
          f"{e2.num_prb.min()}-{e2.num_prb.max()} (epoch 2)\n")
    rows = []
    for env, ov in ENVIRONMENTS.items():
        cfg = env_cfg(base, ov)
        sim = RANSimulator(cfg)
        for s in range(a.seeds):
            rng = np.random.default_rng(58_000 + 131 * s)   # evaluation seeds
            per = [one_cell(sim, M, cfg, rng, a.ues, a.kmin)
                   for _ in range(a.cells)]
            per = [p for p in per if p]
            d = pd.DataFrame(per)
            r = {"env": env, "seed": s, "n_cells": len(d)}
            for m in METHODS:
                r[f"J_{m}"] = d[f"J_{m}"].mean()
                r[f"regret_{m}"] = float(
                    ((d.J_oracle - d[f"J_{m}"]) / d.J_oracle).mean())
            rows.append(r)
    df = pd.DataFrame(rows)
    out = os.path.join(ROOT, "results", "step15_alloc_by_env.csv")
    df.to_csv(out, index=False)

    print(f"mean loss vs the oracle allocation (%), {a.seeds} realizations x "
          f"{a.cells} cells per environment (MLP pooled over its initialisations)")
    inits = [""] + [f"_rs{r}" for r in MLP_INITS]
    for env in ENVIRONMENTS:
        e = df[df.env == env]
        mlp = np.concatenate([e[f"regret_noncausal_mlp{t}"].values for t in inits])
        print(f"  {env:<17} H-SCM {100 * e.regret_hscm_struct.mean():5.2f}   "
              f"no-abduction {100 * e.regret_hscm_rung2.mean():5.2f}   "
              f"eq.(6) {100 * e.regret_cond_gbm.mean():5.2f}   "
              f"MLP {100 * mlp.mean():5.2f}   proportional {100 * e.regret_proportional.mean():5.2f}   "
              f"equal {100 * e.regret_equal.mean():5.2f}")
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()

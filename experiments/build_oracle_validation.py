"""Paired interventional oracle: the held-out ground truth for the per-UE
counterfactual validation (paper Sec. IV, results/step13_val.csv).

N held-out UEs are drawn once (one exogenous realization U^(u) each) and
propagated through the simulator under every arm with the same U, so the
per-UE treatment effects Y_t(u) - Y_t'(u) are exact. The arms cover the
counterfactual queries of the paper: do(T = 10, 50, 100) for CF-1, CF-3 and
the Fig. 2 ground truth, and hardware normalization do(xi_hw = 0) for CF-2.

Input: config/simulator.yaml. Output: data/oracle_paired.csv.gz (N units x
5 arms), the ground truth for the per-UE counterfactual validation and for the
oracle means of Fig. 2.

Usage: python experiments/build_oracle_validation.py [--n 5000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulator.ran_simulator import RANSimulator  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEED = 20250721          # distinct from the dataset seeds (42, 1000+i)
# do(xi_hw = 0) is realised on the multiplicative residual of mac_tput, which
# the simulator computes as f(pa) * (1 + kappa_hw * xi_hw).
ARMS = {"factual": None,
        "do(num_prb=10)": {"num_prb": 10},
        "do(num_prb=50)": {"num_prb": 50},
        "do(num_prb=100)": {"num_prb": 100},
        "do(hw_impairment=0)": {"hw_impairment": 0.0}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "config", "simulator.yaml")) as f:
        cfg = yaml.safe_load(f)

    sim = RANSimulator(cfg)
    rng = np.random.default_rng(SEED)
    u = sim.sample_exogenous(args.n, rng)             # one draw, shared by all arms

    frames = []
    for name, do in ARMS.items():
        df = sim.propagate(u, do=do)
        df.insert(0, "unit", np.arange(args.n))
        df.insert(1, "arm", name)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)

    out = os.path.join(ROOT, "data", "oracle_paired.csv.gz")
    full.to_csv(out, index=False, compression="gzip")
    print(f"wrote {out}: {len(full)} rows = {args.n} units x {len(ARMS)} arms")

    # pairing check: exogenous columns identical across arms
    piv = full.pivot(index="unit", columns="arm", values="shadow_fading")
    assert np.allclose(piv.std(axis=1), 0), "arms are not paired!"
    print("pairing verified: identical U across arms")

    print(f"\noracle means (N={args.n}, SE in brackets):")
    for name in ARMS:
        g = full.loc[full.arm == name, "goodput"]
        print(f"  {name:>21}: {g.mean():7.2f} +- {g.std():6.2f} [SE {g.std() / np.sqrt(len(g)):.3f}]")

if __name__ == "__main__":
    main()

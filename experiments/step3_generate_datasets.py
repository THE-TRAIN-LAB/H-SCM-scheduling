"""Observational dataset of the paper (Sec. IV): 500 UEs sampled from
config/simulator.yaml under the load-dependent scheduler of eq. (2).

Output: data/observational.csv, 500 rows x 23 columns (22 node columns + the
regime label), seed 42. It is the training set of the structural fitting
(step13) and of the Fig. 2 estimators (step6).
"""
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulator.ran_simulator import RANSimulator, COLUMNS  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEED_OBS = 42


def generate(cfg):
    sim = RANSimulator(cfg)
    return sim.sample(int(cfg["observational"]["n_samples"]), seed=SEED_OBS,
                      regime="obs")


def main():
    with open(os.path.join(ROOT, "config", "simulator.yaml")) as f:
        cfg = yaml.safe_load(f)

    obs = generate(cfg)
    assert list(obs.columns) == COLUMNS + ["regime"]
    assert obs.shape == (500, len(COLUMNS) + 1), obs.shape
    assert not obs.isna().any().any()

    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    obs.to_csv(os.path.join(ROOT, "data", "observational.csv"), index=False)
    print(f"dataset: {obs.shape[0]} rows x {obs.shape[1]} cols (observational)")

    r = np.corrcoef(obs.cell_load, obs.num_prb)[0, 1]
    print(f"\nr(cell_load, num_prb) on the 500 obs rows: {r:+.3f}")

if __name__ == "__main__":
    main()

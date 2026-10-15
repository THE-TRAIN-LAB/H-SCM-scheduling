# hscm-ran

Code, data and results for

> **From Confounding to Counterfactuals: A Cross-Layer Causal Model for RAN Scheduling**
> Artin Elhamirad, Christo Kurisummoottil Thomas (Worcester Polytechnic Institute),
> Ali Tajer (Rensselaer Polytechnic Institute)

A hierarchical structural causal model (H-SCM) of a single-cell downlink RAN spans the
PHY, MAC and network layers and separates cell-load confounding from UE-specific
response heterogeneity. PRB allocations are selected from per-UE counterfactual
goodput, obtained by constrained causal discovery, structural-equation fitting and
abduction–action–prediction. Every number and figure in the paper is produced by a
script in `experiments/` from the configuration in `config/`.

## Layout

| directory | contents |
|---|---|
| `simulator/` | the H-SCM as a simulator: Sionna 1.2 PHY (CDL-C, cached BLER table), MAC and network layers of paper Sec. II |
| `config/` | simulator parameters (paper Table I), the variable dictionary and the ground-truth DAG |
| `causal/` | causal discovery with protocol knowledge and mediator blocking, structural fitting, counterfactual inference (Sec. III) |
| `scheduler/` | exact-search PRB allocation from per-UE goodput tables |
| `experiments/` | the scripts that produce the paper's results and figures |
| `data/`, `results/` | the generated dataset, oracle, result tables and figures (`results/figures/fig1`–`fig4`) |
| `tests/` | unit tests of the simulator and the causal pipeline |

| paper | script | output |
|---|---|---|
| Fig. 1, DAG | `plot_dag.py` | `results/figures/fig1_dag.png` |
| Fig. 2, confounding analysis | `step6_rung12.py`, `plot_confounding_recovered.py` | `results/rung2_dose_response.csv`, `fig2_confounding.png` |
| discovery accuracy | `step11_protocol_discovery.py` | `results/step11_per_seed.csv`, `step11_recovered_dag.yaml` |
| held-out R², Fig. 3 | `step13_counterfactuals.py`, `plot_cf_effects.py` | `results/step13_*.csv`, `fig3_counterfactual_gains.png` |
| Fig. 4, scheduling | `step15_alloc_by_env.py`, `plot_alloc_by_env.py` | `results/step15_alloc_by_env.csv`, `fig4_scheduling.png` |

## Reproduce

```bash
conda create -n hscm-ran python=3.11 -y && conda activate hscm-ran
pip install -r requirements.txt

python experiments/build_bler_lut.py                      # MCS-specific BLER curves (reuses data/bler_lut.npz)
python experiments/step3_generate_datasets.py             # the 500 observational UEs
python experiments/build_oracle_validation.py --n 5000    # 5000 held-out UEs with their true counterfactuals
python experiments/step11_protocol_discovery.py --seeds 50    # causal discovery, 50 realizations
python experiments/step6_rung12.py                        # Fig. 2 estimates
python experiments/step13_counterfactuals.py              # held-out R^2 and CF-1/2/3
python experiments/step15_alloc_by_env.py --seeds 50 --cells 25  # scheduling in four environments
python experiments/plot_dag.py
python experiments/plot_confounding_recovered.py
python experiments/plot_cf_effects.py
python experiments/plot_alloc_by_env.py
python -m pytest tests/
```

All artifacts are seeded and regenerate bit-for-bit, except the BLER table
`data/bler_lut.npz`, which is committed and reused whenever its configuration
fingerprint matches; rebuilding it from scratch draws a fresh Monte-Carlo sample.
Verified on macOS arm64 (CPU) with Sionna 1.2.2 / TensorFlow 2.21.

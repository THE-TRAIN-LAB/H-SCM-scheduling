"""L2 MAC layer (paper Sec. II-A). All functions vectorized over units.

Structural equations owned by this module:

    num_prb   = clip(floor(base_prb * (1 - alpha_L*L)) + eps_T, 1, base_prb)  (eq. (2))
    harq_retx = sum_{k=1..K} bler^k          expected HARQ retransmissions     (eq. (3))
    bler_res  = bler^(1 + K)                 residual decoding failure after the
                                             initial transmission and K
                                             retransmissions (used by
                                             network.packet_loss)
    se_eff    = se_phy * (1 + kappa_hw * xi_hw)
    mac_tput  = C0 * num_prb * se_eff / (1 + kappa_harq * harq_retx)

HARQ model. Every attempt fails independently with probability bler, up to
K = mac.harq_max_rounds retransmissions:
  - harq_retx = sum_{k=1..K} bler^k is the expected retransmission count, a
    fractional quantity in [0, K];
  - the residual failure probability after all 1 + K attempts is bler^(1 + K),
    a function of bler and the fixed limit K only.

L -> num_prb is the edge that creates the confounding (paper Sec. II-B) and
the edge severed by do(num_prb = t).
"""
import numpy as np


def scheduler_num_prb(load, eps_T, cfg):
    """Eq. (2), observational mode. Under do(num_prb = t) the orchestrator
    replaces this equation by the constant t."""
    base = int(cfg["mac"]["base_prb"])
    alpha = float(cfg["mac"]["alpha_L"])
    raw = np.floor(base * (1.0 - alpha * np.asarray(load))) + np.asarray(eps_T)
    return np.clip(raw, 1, base)


def harq_expected_retx(bler, cfg):
    """Eq. (3): expected HARQ retransmissions sum_{k=1..K} bler^k, in [0, K]."""
    K = int(cfg["mac"]["harq_max_rounds"])
    bler = np.asarray(bler, dtype=float)
    return sum(bler ** k for k in range(1, K + 1))


def residual_bler(bler, cfg):
    """Residual decoding failure after the initial transmission and K
    retransmissions that each fail independently with probability bler:
    bler^(1 + K), K = mac.harq_max_rounds."""
    K = int(cfg["mac"]["harq_max_rounds"])
    bler = np.asarray(bler, dtype=float)
    return np.clip(bler, 1e-12, 1.0) ** (1.0 + K)


def se_eff(se_phy, xi_hw, cfg):
    """Device-impairment-modified spectral efficiency (effect modifier xi_hw)."""
    k = float(cfg["mac"]["kappa_hw"])
    return np.asarray(se_phy) * np.maximum(1.0 + k * np.asarray(xi_hw), 0.05)


def mac_throughput(num_prb, se_phy, xi_hw, harq_retx, cfg):
    """MAC service rate mu, Mbps: the rate the MAC can serve after
    retransmission overhead. A retransmission consumes airtime whether or not
    it succeeds, hence the 1 / (1 + kappa_harq * harq_retx) term; whether a
    served block decodes is settled once, in network.packet_loss.
    """
    C0 = float(cfg["mac"]["tput_scale_mbps_per_prb_se"])
    kh = float(cfg["mac"]["kappa_harq"])
    tput = (C0 * np.asarray(num_prb) * se_eff(se_phy, xi_hw, cfg)
            / (1.0 + kh * np.asarray(harq_retx)))
    floor = float(cfg["mac"].get("tput_floor_mbps", 0.0))
    return np.maximum(tput, floor)



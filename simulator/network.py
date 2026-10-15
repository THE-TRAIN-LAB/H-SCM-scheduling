"""L3 network layer (paper Sec. II-A). All functions vectorized over units.

Structural equations owned by this module:

    rho       = min(lambda / mac_tput, rho_max)                                 (eq. (4))
    queue_dep = [ rho^2 * xi_tb / (2*(1 - rho)) + eps3 ]_+                     (eq. (4))
    pkt_loss  = 1 - (1 - bler_res) * exp(-max(0, queue_dep - q_thr) / q_scale) (eq. (5))
    goodput   = mac_tput * (1 - pkt_loss)          <- outcome Y

lambda (network.arrival_mbps) is a fixed constant, not an SCM node: it is the
loading parameter of the queueing model only, setting the utilization rho and
hence backlog and loss, and it is not the offered rate of the flow whose
goodput is reported. Goodput is a served rate: the MAC service rate reduced by the losses that apply to what
is served. pkt_loss therefore has no mac_tput parent, since
Y = mac_tput * (1 - pkt_loss) already carries the service limit once, and
rho_max is a validity cap on the queue term only.
"""
import numpy as np

from .mac import residual_bler


def raw_traffic_intensity(mac_tput, cfg):
    """Uncapped utilization lambda / mac_tput."""
    arr = float(cfg["network"]["arrival_mbps"])
    return arr / np.asarray(mac_tput)


def traffic_intensity(mac_tput, cfg):
    """rho for eq. (4): lambda / mac_tput capped at rho_max."""
    rho_max = float(cfg["network"]["rho_max"])
    return np.clip(raw_traffic_intensity(mac_tput, cfg), 0.0, rho_max)


def queue_depth(mac_tput, xi_tb, eps3, cfg):
    """Eq. (4): mean queue backlog Q.

        Q = [ rho^2 * C_a^2 / (2 (1 - rho)) + eps3 ]_+     with xi_tb = C_a^2

    This is the GI/G/1 mean-queue-length approximation with no service-time
    variability (C_s^2 = 0): once mac_tput is fixed for a unit, its service
    capability is fixed at that operating point. The term is the backlog
    (units waiting), not the total occupancy, since it feeds the overflow
    loss in packet_loss. Arrival variability enters once, as C_a^2 = xi_tb
    in the numerator. The approximation requires rho < 1, so rho is capped
    at rho_max; overload reaches packet_loss through the capped backlog.
    """
    rho = traffic_intensity(mac_tput, cfg)
    qd = rho ** 2 * np.asarray(xi_tb) / (2.0 * (1.0 - rho)) + np.asarray(eps3)
    return np.maximum(qd, 0.0)


def packet_loss(bler, queue_dep, cfg):
    """Eq. (5): residual link loss x queue-overflow loss.

    The residual decoding failure after the initial transmission and K
    retransmissions is bler^(1 + K) (mac.residual_bler), so harq_retx is not
    a parent. Both factors are fractions of what is served, so mac_tput is
    not a parent either (see the module note).
    """
    q_thr = float(cfg["network"]["q_thr"])
    q_scale = float(cfg["network"]["q_scale"])
    b_res = residual_bler(bler, cfg)
    overflow = np.exp(-np.maximum(0.0, np.asarray(queue_dep) - q_thr) / q_scale)
    return np.clip(1.0 - (1.0 - b_res) * overflow, 0.0, 1.0)


def goodput(mac_tput, pkt_loss):
    """Outcome Y: goodput = tput * (1 - loss), Mbps."""
    return np.asarray(mac_tput) * (1.0 - np.asarray(pkt_loss))

"""Light-cone covering compiler and stratified shot allocation for LC-QGNN deployment.

cover_graph packs the exact light cones of many roots into shared circuits of at most
Q_hw qubits.  Because every gate outside a root's cone commutes with that root's
Heisenberg-evolved observable, each root read from a covering circuit has exactly the
expectation of its own cone circuit (verified numerically in exp_covering.py).

Shot allocation: a stratum is (circuit, channel, measurement basis).  One shot of a
stratum returns +/-1 outcomes for every root assigned to that circuit.  With the
read-out linearised around the exact expectations, the graph margin estimator is a sum
of stratum means, and the Neyman rule n_s ~ sigma_s minimises its variance.
"""
import math
import numpy as np
from data_utils import ball, cone


def cover_graph(g, L, Qhw, Qcone=None):
    """Qcone: cone cap used in training (cones larger than Qcone were truncated there and
    therefore get their own truncated circuit here so that deployment matches training)."""
    Qcone = Qhw if Qcone is None else Qcone
    cones = {}
    own = []          # roots whose exact cone exceeds Qhw: own (truncated) circuit
    for r in range(g.n):
        nodes, dist = ball(g, [r], L)
        if len(nodes) <= min(Qhw, Qcone):
            cones[r] = set(nodes)
        else:
            own.append(r)
    unc = set(cones)
    circuits = []
    while unc:
        seed = max(unc, key=lambda r: (len(cones[r]), -r))
        S = set(cones[seed]); A = [seed]; unc.discard(seed)
        while True:
            # roots already contained come for free
            free = [r for r in unc if cones[r] <= S]
            for r in free:
                A.append(r); unc.discard(r)
            best, bcost = None, None
            for r in unc:
                new = len(cones[r] - S)
                if len(S) + new <= Qhw:
                    key = (new, -len(cones[r] & S), r)
                    if bcost is None or key < bcost:
                        best, bcost = r, key
            if best is None:
                break
            S |= cones[best]; A.append(best); unc.discard(best)
        circuits.append((sorted(S), sorted(A)))
    for r in own:
        nodes, edges, _ = cone(g, r, L, Qcone)
        circuits.append((nodes, [r], edges))   # truncated cone circuit (edges given explicitly)
    return circuits


# ---------------- numpy state-vector simulation of one circuit ----------------

def _zbits(m):
    s = np.arange(2 ** m)
    return 1 - 2 * ((s[:, None] >> (m - 1 - np.arange(m))[None, :]) & 1)   # (2^m, m) +-1


def _rx_all(psi, beta, m):
    for q in range(m):
        p = psi.reshape(2 ** q, 2, 2 ** (m - q - 1))
        c, s = math.cos(beta[q]), math.sin(beta[q])
        a0, a1 = p[:, 0, :].copy(), p[:, 1, :].copy()
        p[:, 0, :] = c * a0 - 1j * s * a1
        p[:, 1, :] = -1j * s * a0 + c * a1
        psi = p.reshape(-1)
    return psi


def simulate(nodes, edges_local, feats, cp, Zb=None):
    """cp: one channel's parameters as numpy arrays. Returns final state (2^m,)."""
    m = len(nodes)
    Zb = _zbits(m) if Zb is None else Zb
    E = np.zeros(2 ** m)
    for a, b in edges_local:
        E += Zb[:, a] * Zb[:, b]
    psi = np.full(2 ** m, 1 / math.sqrt(2 ** m), np.complex128)
    for l in range(cp["gamma"].shape[0]):
        h = feats @ cp["wh"][l] + cp["bh"][l]
        psi = psi * np.exp(-1j * (cp["gamma"][l] * E + Zb @ h))
        beta = feats @ cp["wb"][l] + cp["bb"][l]
        psi = _rx_all(psi, beta, m)
    return psi


def basis_probs(psi, m, basis):
    """Probability vector of the computational outcome after rotating every qubit so that
    `basis` (X, Y or Z) is measured."""
    if basis == "Z":
        return np.abs(psi) ** 2
    H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    U = H if basis == "X" else H @ np.diag([1, -1j])     # H S^dagger maps Y -> Z
    for q in range(m):
        p = psi.reshape(2 ** q, 2, 2 ** (m - q - 1))
        psi = np.einsum("ij,ajb->aib", U, p).reshape(-1)
    pr = np.abs(psi) ** 2
    return pr / pr.sum()


def circuit_edges(g, nodes):
    idx = {v: i for i, v in enumerate(nodes)}
    return [(idx[a], idx[b]) for a in nodes for b in g.adj[a] if b in idx and idx[a] < idx[b]]

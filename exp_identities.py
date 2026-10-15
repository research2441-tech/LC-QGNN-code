"""Numerical checks of the closed-form identities used in Section 3.3.

(1) One-layer root read-out  phi_r = cos^{d_r}(2g) (cos2h, cos2b sin2h, sin2b sin2h)
(2) Two-body correlator without field and mixer
        <X_r X_j> = 1/2 cos^{e_rj}(2g) (1 + cos^{c_rj}(4g))
(3) The 1-WL-equivalent pair C6 vs 2 x C3: root read-outs coincide at L = 1 and
    differ at L = 2 (median maximum difference over random parameter draws).
Every check runs the full-register state-vector simulator of covering.py.
Output: results/identities.json
"""
import json, os
import numpy as np
import networkx as nx
from covering import simulate, basis_probs, _zbits


def expect(psi, m, Zb):
    pr = {b: basis_probs(psi, m, b) for b in "XYZ"}
    return pr, np.stack([pr[b] @ Zb for b in "XYZ"], 1)          # (m, 3) one-body read-outs


def run_graph(G, cp, feats):
    n = G.number_of_nodes(); Zb = _zbits(n)
    psi = simulate(list(range(n)), list(G.edges()), feats, cp, Zb)
    return psi, n, Zb


def params(rng, L, F=1, zero_fields=False):
    cp = {"gamma": rng.uniform(0, np.pi, L), "wh": rng.normal(0, 0.5, (L, F)), "bh": rng.uniform(0, np.pi, L),
          "wb": rng.normal(0, 0.5, (L, F)), "bb": rng.uniform(0, np.pi, L)}
    if zero_fields:
        cp.update(wh=np.zeros((L, F)), bh=np.zeros(L), wb=np.zeros((L, F)), bb=np.zeros(L))
    return cp


def main(draws=50, seed=0):
    rng = np.random.default_rng(seed)
    err1 = err2 = 0.0
    for t in range(draws):
        G = nx.gnp_random_graph(int(rng.integers(5, 10)), 0.45, seed=int(rng.integers(1 << 30)))
        n = G.number_of_nodes(); feats = rng.normal(size=(n, 1))
        cp = params(rng, 1)
        psi, n, Zb = run_graph(G, cp, feats)
        _, phi = expect(psi, n, Zb)
        g = cp["gamma"][0]; h = feats @ cp["wh"][0] + cp["bh"][0]; b = feats @ cp["wb"][0] + cp["bb"][0]
        deg = np.array([G.degree(v) for v in range(n)])
        pred = np.cos(2 * g) ** deg[:, None] * np.stack([np.cos(2 * h), np.cos(2 * b) * np.sin(2 * h),
                                                          np.sin(2 * b) * np.sin(2 * h)], 1)
        err1 = max(err1, float(np.abs(phi - pred).max()))
        # two-body correlator, no field and mixer
        cp0 = params(rng, 1, zero_fields=True)
        psi, n, Zb = run_graph(G, cp0, feats)
        px = basis_probs(psi, n, "X"); g = cp0["gamma"][0]
        for r in range(n):
            for j in range(r + 1, n):
                xx = float(px @ (Zb[:, r] * Zb[:, j]))
                Nr, Nj = set(G[r]) - {j}, set(G[j]) - {r}
                c = len(Nr & Nj); e = len(Nr ^ Nj)
                pred2 = 0.5 * np.cos(2 * g) ** e * (1 + np.cos(4 * g) ** c)
                err2 = max(err2, abs(xx - pred2))
    # 1-WL-equivalent pair
    C6 = nx.cycle_graph(6); T2 = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))
    out = {}
    for L in (1, 2):
        diffs = []
        for t in range(200):
            cp = params(rng, L); feats = np.ones((6, 1))
            a = expect(*run_graph(C6, cp, feats)[0:1], 6, _zbits(6))[1][0]
            b = expect(*run_graph(T2, cp, feats)[0:1], 6, _zbits(6))[1][0]
            diffs.append(float(np.abs(a - b).max()))
        out[f"L{L}_median_diff"] = float(np.median(diffs)); out[f"L{L}_max_diff"] = float(np.max(diffs))
    R = {"onelayer_max_abs_err": err1, "twobody_max_abs_err": err2, "draws": draws, **out}
    os.makedirs("results", exist_ok=True)
    json.dump(R, open("results/identities.json", "w"), indent=1)
    print(R)


if __name__ == "__main__":
    main()

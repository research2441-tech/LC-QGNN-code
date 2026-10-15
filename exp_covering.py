"""Covering-compiler statistics and exactness audit.
Outputs results/covering.json with, per dataset and hardware width Q_hw:
  circuits per graph (covering vs one circuit per node), mean qubits per circuit,
  qubit utilisation, share of graphs fitting in one circuit, and the maximum deviation
between covering-circuit expectations and cone expectations under trained parameters.
"""
import json, os, sys, time
import numpy as np
from data_utils import load_tu, DATASETS, ball
from covering import cover_graph, simulate, basis_probs, circuit_edges, _zbits


def stats(name, L=2, Qcone=10, widths=(10, 12, 16, 20, 24)):
    gs = load_tu(name)
    out = {}
    for Qhw in widths:
        nc, nn, qsum, one, own = 0, 0, 0, 0, 0
        for g in gs:
            c = cover_graph(g, L, Qhw, Qcone)
            nc += len(c); nn += g.n; qsum += sum(len(x[0]) for x in c)
            one += len(c) == 1; own += sum(len(x) == 3 for x in c)
        out[Qhw] = {"circuits_per_graph": nc / len(gs), "nodes_per_graph": nn / len(gs),
                    "reduction": nn / nc, "mean_qubits": qsum / nc, "single_circuit_share": one / len(gs),
                    "truncated_circuits_per_graph": own / len(gs)}
    return out


def exactness(name, L=2, Qcone=10, Qhw=16, n_graphs=40, res="results"):
    import lcqgnn as Q
    from cones import ConeTable
    gs = load_tu(name)
    P = np.load(os.path.join(res, f"params_{name}_lcqgnn_f1.npy"), allow_pickle=True).item()["params"]
    C = P["circ"]["gamma"].shape[0]
    cps = [{k: np.asarray(v[c], np.float64) for k, v in P["circ"].items()} for c in range(C)]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(gs), size=min(n_graphs, len(gs)), replace=False)
    sub = [gs[i] for i in sel]
    tab = ConeTable(sub, L, Qcone); sim = Q.ConeSim(tab); phi = sim.phi(P["circ"])
    err = 0.0; nroots = 0
    for gi, g in enumerate(sub):
        for cir in cover_graph(g, L, Qhw, Qcone):
            nodes, roots = cir[0], cir[1]
            edges = cir[2] if len(cir) == 3 else circuit_edges(g, nodes)
            m = len(nodes); Zb = _zbits(m)
            for c in range(C):
                psi = simulate(nodes, edges, g.feat[nodes].astype(np.float64), cps[c], Zb)
                pr = {b: basis_probs(psi, m, b) for b in "XYZ"}
                for r in roots:
                    i = nodes.index(r)
                    ex = np.array([pr[b] @ Zb[:, i] for b in "XYZ"])
                    err = max(err, float(np.abs(ex - phi[tab.node_cone[gi][r], 3 * c:3 * c + 3]).max()))
                    nroots += 1
    return {"max_abs_dev": err, "roots_checked": nroots, "graphs": len(sub)}


def full_register(name, L=2, Qcone=10, nmax=16, n_graphs=40, res="results"):
    """Theorem 1 check: whole-graph register (one qubit per node, all edges) vs cone
    simulator, for graphs with at most nmax nodes and no truncated cone."""
    import lcqgnn as Q
    from cones import ConeTable
    gs = load_tu(name)
    P = np.load(os.path.join(res, f"params_{name}_lcqgnn_f1.npy"), allow_pickle=True).item()["params"]
    C = P["circ"]["gamma"].shape[0]
    cps = [{k: np.asarray(v[c], np.float64) for k, v in P["circ"].items()} for c in range(C)]
    cand = [i for i, g in enumerate(gs) if g.n <= nmax and all(len(ball(g, [r], L)[0]) <= Qcone for r in range(g.n))]
    rng = np.random.default_rng(1)
    sel = rng.choice(cand, size=min(n_graphs, len(cand)), replace=False)
    sub = [gs[i] for i in sel]
    tab = ConeTable(sub, L, Qcone); sim = Q.ConeSim(tab); phi = sim.phi(P["circ"])
    err = 0.0; nroots = 0
    for gi, g in enumerate(sub):
        nodes = list(range(g.n)); edges = circuit_edges(g, nodes); Zb = _zbits(g.n)
        for c in range(C):
            psi = simulate(nodes, edges, g.feat.astype(np.float64), cps[c], Zb)
            pr = {b: basis_probs(psi, g.n, b) for b in "XYZ"}
            for r in nodes:
                ex = np.array([pr[b] @ Zb[:, r] for b in "XYZ"])
                err = max(err, float(np.abs(ex - phi[tab.node_cone[gi][r], 3 * c:3 * c + 3]).max()))
                nroots += 1
    return {"max_abs_dev": err, "roots_checked": nroots, "graphs": len(sub), "max_n": int(max(g.n for g in sub))}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "stats"
    os.makedirs("results", exist_ok=True)
    if which == "stats":
        R = {d: stats(d) for d in DATASETS + ["REG3TRI"]}
        json.dump(R, open("results/covering.json", "w"), indent=1)
        for d, v in R.items():
            print(d, {k: (round(x["circuits_per_graph"], 2), round(x["reduction"], 2), round(x["mean_qubits"], 1)) for k, x in v.items()})
    elif which == "full":
        R = {d: full_register(d) for d in sys.argv[2:]}
        json.dump(R, open("results/full_register.json", "w"), indent=1)
        print(R)
    else:
        R = {d: exactness(d) for d in sys.argv[2:]}
        json.dump(R, open("results/exactness.json", "w"), indent=1)
        print(R)

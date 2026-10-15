"""Finite-shot deployment study.

For every test graph (folds given on the command line) the trained fold model is
compiled into covering circuits (Q_hw qubits, cones capped at Q_cone as in training).
Exact outcome distributions are obtained by state-vector simulation and shots are drawn
from them, so correlations between roots measured in the same circuit are preserved.

Strategies at equal total shot budget M per graph:
  PER-NODE  : one circuit per node (cone), uniform split   [ego-decomposition practice]
  COV-UNI   : covering circuits, uniform split over strata
  COV-NEY   : covering circuits, Neyman split n_s ~ sigma_s with sigma_s from a pilot of
              25 % of the budget (no oracle information)
  COV-ORA   : Neyman split with exact sigma_s (oracle, reference only)
plus SEQ    : shot-certified sequential rule (pilot, then doubling until |m| >= z*SE).
Readout error can be added with --flip q (symmetric bit-flip on every measured qubit,
followed by the standard 1/(1-2q) rescaling).
"""
import argparse, json, os, time
import numpy as np
from data_utils import load_tu, load_folds, cone
from covering import cover_graph, simulate, basis_probs, circuit_edges, _zbits


def head_np(hp, z):
    h = np.maximum(z @ hp["W1"] + hp["b1"], 0)
    return h @ hp["W2"] + hp["b2"]


def margin_grad(hp, z):
    pre = z @ hp["W1"] + hp["b1"]
    d = (hp["W2"][:, 1] - hp["W2"][:, 0]) * (pre > 0)
    return hp["W1"] @ d


class GraphJob:
    """All strata of one graph: probability vectors and per-shot statistics."""

    def __init__(self, g, P, kappa, L, Qhw, Qcone, per_node=False):
        C = P["circ"]["gamma"].shape[0]
        cps = [{k: np.asarray(v[c], np.float64) for k, v in P["circ"].items()} for c in range(C)]
        if per_node:
            circs = []
            for r in range(g.n):
                nodes, edges, _ = cone(g, r, L, Qcone)
                circs.append((nodes, [r], edges))
        else:
            circs = cover_graph(g, L, Qhw, Qcone)
        self.n, self.C = g.n, C
        phi = np.zeros((g.n, 3 * C))
        strata = []          # (probs, signs matrix (2^m, |A|), roots, channel, basis)
        for cir in circs:
            nodes, roots = cir[0], cir[1]
            edges = cir[2] if len(cir) == 3 else circuit_edges(g, nodes)
            m = len(nodes); Zb = _zbits(m)
            pos = [nodes.index(r) for r in roots]
            for c in range(C):
                psi = simulate(nodes, edges, g.feat[nodes].astype(np.float64), cps[c], Zb)
                for bi, b in enumerate("XYZ"):
                    pr = basis_probs(psi, m, b)
                    S = Zb[:, pos].astype(np.float64)
                    phi[roots, 3 * c + bi] = pr @ S
                    strata.append((pr, S, np.array(roots), c, bi))
        self.phi_exact = phi
        z = self.pool(phi, kappa)
        self.kappa = kappa
        self.exact_logits = head_np(P["head"], z)
        gz = margin_grad(P["head"], z)
        F = 3 * C
        a = gz[:F] / g.n + gz[F:] / kappa           # d margin / d phi_{r,f}, identical for all r
        self.strata = []
        for pr, S, roots, c, bi in strata:
            y = S.sum(1) * a[3 * c + bi]            # per-shot stratum variable
            mu = pr @ y; sd = np.sqrt(max(pr @ (y * y) - mu * mu, 0.0))
            self.strata.append((pr, S, roots, c, bi, y, sd))
        self.P = P

    def pool(self, phi, kappa):
        return np.concatenate([phi.mean(0), phi.sum(0) / kappa])

    def estimate(self, alloc, rng, flip=0.0):
        """alloc: shots per stratum.  Returns the estimated root expectations."""
        phi = np.zeros_like(self.phi_exact)
        for (pr, S, roots, c, bi, y, sd), ns in zip(self.strata, alloc):
            ns = int(ns)
            if ns <= 0:
                continue
            cnt = rng.multinomial(ns, pr)
            nz = np.nonzero(cnt)[0]
            if flip > 0:
                rep = np.repeat(nz, cnt[nz])
                Sm = S[rep] * np.where(rng.random((len(rep), S.shape[1])) < flip, -1.0, 1.0)
                est = Sm.mean(0) / (1 - 2 * flip)
            else:
                est = (cnt[nz] @ S[nz]) / ns
            phi[roots, 3 * c + bi] = est
        return phi

    def predict(self, phi):
        return head_np(self.P["head"], self.pool(phi, self.kappa))


def neyman(sd, M, floor=1):
    k = len(sd)
    if sd.sum() <= 0:
        return np.full(k, M // k)
    w = sd / sd.sum()
    n = np.floor(w * (M - floor * k)).astype(int) + floor
    return n


def run(dataset, folds, budgets, reps, Qhw, Qcone, L, res, flip, seed=0):
    gs = load_tu(dataset); y = np.array([g.label for g in gs]); F = load_folds(dataset)
    rng = np.random.default_rng(seed)
    out = {s: {M: {"acc": [], "agree": [], "mse": []} for M in budgets} for s in ("PER-NODE", "COV-UNI", "COV-NEY", "COV-ORA")}
    seq = {"shots": [], "correct": [], "agree": []}
    exact_correct = []
    meta = {"strata_cov": [], "strata_pn": []}
    for k in folds:
        D = np.load(os.path.join(res, f"params_{dataset}_lcqgnn_f{k}.npy"), allow_pickle=True).item()
        P = {"circ": {kk: np.asarray(v) for kk, v in D["params"]["circ"].items()},
             "head": {kk: np.asarray(v) for kk, v in D["params"]["head"].items()}}
        kappa = D["kappa"]
        for gi in F[k - 1][1]:
            g = gs[gi]
            jc = GraphJob(g, P, kappa, L, Qhw, Qcone)
            jp = GraphJob(g, P, kappa, L, Qhw, Qcone, per_node=True)
            meta["strata_cov"].append(len(jc.strata)); meta["strata_pn"].append(len(jp.strata))
            ex = int(np.argmax(jc.exact_logits)); exact_correct.append(ex == y[gi])
            sd = np.array([s[-1] for s in jc.strata])
            for M in budgets:
                for strat in out:
                    accs, agr, se = [], [], []
                    m_ex = jc.exact_logits[1] - jc.exact_logits[0]
                    for r in range(reps):
                        if strat == "PER-NODE":
                            ks = len(jp.strata); al = np.full(ks, max(M // ks, 1))
                            phi = jp.estimate(al, rng, flip)
                            pr = jp.predict(phi)
                        elif strat == "COV-UNI":
                            ks = len(jc.strata); al = np.full(ks, max(M // ks, 1))
                            pr = jc.predict(jc.estimate(al, rng, flip))
                        elif strat == "COV-ORA":
                            al = neyman(sd, M); pr = jc.predict(jc.estimate(al, rng, flip))
                        else:   # pilot 25 %, then Neyman on estimated sigma
                            ks = len(jc.strata); n0 = max((M // 4) // ks, 2)
                            sdh = pilot_sd(jc, n0, rng)
                            al = neyman(sdh, max(M - n0 * ks, ks)) + n0
                            pr = jc.predict(jc.estimate(al, rng, flip))
                        p = int(np.argmax(pr)); accs.append(p == y[gi]); agr.append(p == ex)
                        se.append(((pr[1] - pr[0]) - m_ex) ** 2)
                    out[strat][M]["acc"].append(float(np.mean(accs))); out[strat][M]["agree"].append(float(np.mean(agr)))
                    out[strat][M]["mse"].append(float(np.mean(se)))
            # sequential shot-certified decision (z = 2)
            for r in range(reps):
                used, pred = sequential(jc, rng, budgets[0], budgets[-1])
                seq["shots"].append(used); seq["correct"].append(pred == y[gi]); seq["agree"].append(pred == ex)
    summ = {"exact_acc": float(np.mean(exact_correct)), "n_graphs": len(exact_correct),
            "mean_strata_cov": float(np.mean(meta["strata_cov"])), "mean_strata_pernode": float(np.mean(meta["strata_pn"]))}
    for s in out:
        summ[s] = {M: {"acc": float(np.mean(v["acc"])), "agree": float(np.mean(v["agree"])),
                       "margin_mse": float(np.mean(v["mse"]))} for M, v in out[s].items()}
    summ["SEQ"] = {"mean_shots": float(np.mean(seq["shots"])), "median_shots": float(np.median(seq["shots"])),
                   "acc": float(np.mean(seq["correct"])), "agree": float(np.mean(seq["agree"]))}
    return summ


def pilot_sd(job, n0, rng):
    sds = []
    for (pr, S, roots, c, bi, y, sd) in job.strata:
        cnt = rng.multinomial(n0, pr)
        nz = np.nonzero(cnt)[0]
        m = cnt[nz] @ y[nz] / n0
        v = cnt[nz] @ (y[nz] - m) ** 2 / max(n0 - 1, 1)
        sds.append(np.sqrt(v) + 1e-6)
    return np.array(sds)


def sequential(job, rng, M0, Mmax, z=2.0):
    """Doubling budget; Neyman allocation from accumulated counts; stop when the
    linearised margin is at least z standard errors from the decision boundary."""
    ks = len(job.strata)
    counts = [np.zeros(len(s[0])) for s in job.strata]
    def draw(al):
        for i, ((pr, *_), n) in enumerate(zip(job.strata, al)):
            if n > 0:
                counts[i] += rng.multinomial(int(n), pr)
    draw(np.full(ks, max(M0 // ks, 2)))
    while True:
        phi = np.zeros_like(job.phi_exact); var = 0.0; sds = []
        for (pr, S, roots, c, bi, y, sd), cnt in zip(job.strata, counts):
            n = cnt.sum()
            phi[roots, 3 * c + bi] = (cnt @ S) / n
            m = cnt @ y / n; v = cnt @ (y - m) ** 2 / max(n - 1, 1)
            var += v / n; sds.append(np.sqrt(v) + 1e-6)
        lg = job.predict(phi); marg = lg[1] - lg[0]
        used = int(sum(c.sum() for c in counts))
        if abs(marg) >= z * np.sqrt(var) or used >= Mmax:
            return used, int(np.argmax(lg))
        draw(neyman(np.array(sds), used))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("--folds", default="1-10")
    ap.add_argument("--reps", type=int, default=10); ap.add_argument("--Qhw", type=int, default=16)
    ap.add_argument("--flip", type=float, default=0.0); ap.add_argument("--res", default="results")
    ap.add_argument("--tag", default="")
    ap.add_argument("--budgets", default="64,128,256,512,1024,2048,4096,8192,16384")
    a = ap.parse_args()
    f0, f1 = map(int, a.folds.split("-"))
    budgets = [int(x) for x in a.budgets.split(",")]
    t = time.time()
    S = run(a.dataset, list(range(f0, f1 + 1)), budgets, a.reps, a.Qhw, 10, 2, a.res, a.flip)
    S["time"] = time.time() - t
    json.dump(S, open(os.path.join(a.res, f"shots_{a.dataset}{a.tag}.json"), "w"), indent=1)
    print(json.dumps(S, indent=1))

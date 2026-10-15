"""Statistical analysis of the 10-fold benchmark.
* per data set: mean, SD, 95% t-interval of fold accuracy
* paired Wilcoxon signed-rank test LC-QGNN vs every baseline over the 10 folds,
  Holm-corrected within each data set; matched-pairs rank-biserial effect size
* Friedman test over data sets with average ranks (Demsar 2006)
Output: results/stats.json
"""
import json, glob, os
import numpy as np
from scipy import stats
from summarize import load

MODELS = ["lcqgnn", "wl", "mlp", "gcn", "sage", "gin", "gat", "conegin"]
DATA = ["MUTAG", "PTC", "NCI1", "PROTEINS", "IMDBBINARY", "REG3TRI"]


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0
    for i, k in enumerate(o):
        run = max(run, (m - i) * p[k]); adj[k] = min(run, 1.0)
    return adj


def rank_biserial(a, b):
    d = np.asarray(a) - np.asarray(b); d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def main():
    out = {"per_dataset": {}, "friedman": None}
    acc = {}
    for ds in DATA:
        p = f"results/{ds}.jsonl"
        if not os.path.exists(p):
            continue
        R = load(p)
        acc[ds] = {}
        for m in MODELS:
            if m in R and len(R[m]) == 10:
                acc[ds][m] = np.array([R[m][k]["test_acc"] for k in range(1, 11)]) * 100
        info = {}
        for m, a in acc[ds].items():
            h = stats.t.ppf(0.975, 9) * a.std(ddof=1) / np.sqrt(10)
            info[m] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "ci": [float(a.mean() - h), float(a.mean() + h)]}
        if "lcqgnn" in acc[ds]:
            q = acc[ds]["lcqgnn"]; names, ps, effs = [], [], []
            for m, a in acc[ds].items():
                if m == "lcqgnn":
                    continue
                if np.allclose(q, a):
                    pv = 1.0
                else:
                    pv = float(stats.wilcoxon(q, a, zero_method="pratt").pvalue)
                names.append(m); ps.append(pv); effs.append(rank_biserial(q, a))
            adj = holm(ps)
            for m, pv, pa, e in zip(names, ps, adj, effs):
                info[m].update(p=pv, p_holm=float(pa), rbc=e)
        out["per_dataset"][ds] = info
    full = [ds for ds in acc if all(m in acc[ds] for m in MODELS)]
    if len(full) >= 3:
        M = np.array([[acc[ds][m].mean() for m in MODELS] for ds in full])
        ranks = np.array([stats.rankdata(-row) for row in M])
        chi = stats.friedmanchisquare(*M.T)
        out["friedman"] = {"datasets": full, "models": MODELS, "avg_rank": ranks.mean(0).tolist(),
                           "chi2": float(chi.statistic), "p": float(chi.pvalue)}
    json.dump(out, open("results/stats.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    o = main()
    for ds, v in o["per_dataset"].items():
        print(ds, {m: (round(x["mean"], 2), round(x.get("p_holm", -1), 3)) for m, x in v.items()})
    print(o["friedman"])

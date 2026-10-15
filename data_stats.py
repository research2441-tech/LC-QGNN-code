"""Dataset and light-cone statistics (Table: data).  Output results/data_stats.json"""
import json
import numpy as np
from data_utils import load_tu, DATASETS, ball
from cones import ConeTable
R = {}
for name in DATASETS + ["REG3TRI"]:
    gs = load_tu(name)
    n = np.array([g.n for g in gs]); m = np.array([len(g.edges()) for g in gs])
    y = np.array([g.label for g in gs])
    cs = np.array([len(ball(g, [r], 2)[0]) for g in gs for r in range(g.n)])
    tab = ConeTable(gs, 2, 10)
    R[name] = {"graphs": len(gs), "classes": int(y.max() + 1), "nodes_mean": float(n.mean()), "nodes_max": int(n.max()),
               "edges_mean": float(m.mean()), "F": int(gs[0].feat.shape[1]), "cone_median": float(np.median(cs)),
               "cone_p90": float(np.percentile(cs, 90)), "trunc_frac": float((cs > 10).mean()),
               "unique": tab.n_unique, "total": tab.n_total, "dedup": tab.n_total / tab.n_unique,
               "class_balance": [int(c) for c in np.bincount(y)]}
    print(name, R[name])
json.dump(R, open("results/data_stats.json", "w"), indent=1)

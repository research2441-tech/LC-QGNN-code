"""Collects every number reported in the manuscript into results_summary.json.

Per-fold test accuracies of all models and ablations, significance tests, data and
cone statistics, exactness checks, closed-form checks, covering statistics, shot
studies and gradient variances.  A re-run can be compared with this file directly.
"""
import json, os, glob
from summarize import load


def jl(path):
    R = load(path) if os.path.exists(path) else {}
    return {m: {str(k): round(f[k]["test_acc"] * 100, 4) for k in sorted(f)} for m, f in R.items()}


def j(name):
    p = os.path.join("results", name)
    return json.load(open(p)) if os.path.exists(p) else None


if __name__ == "__main__":
    S = {"accuracy_per_fold": {d: jl(f"results/{d}.jsonl") for d in ("MUTAG", "PTC", "NCI1", "REG3TRI")},
         "ablation_per_fold": {d: jl(f"results_abl/{d}.jsonl") for d in ("MUTAG", "REG3TRI")}}
    for name in ("stats", "data_stats", "full_register", "exactness", "identities", "covering", "trainability"):
        S[name] = j(f"{name}.json")
    S["shots"] = {os.path.basename(p)[6:-5]: json.load(open(p)) for p in sorted(glob.glob("results/shots_*.json"))}
    json.dump(S, open("results_summary.json", "w"), indent=1)
    print("written results_summary.json")

"""Summaries of results/*.jsonl: mean (SD) test accuracy per model per dataset."""
import json, sys, glob, os
import numpy as np
from collections import defaultdict
def load(path):
    R = defaultdict(dict)
    for ln in open(path):
        r = json.loads(ln); R[r["model"]][r["fold"]] = r
    return R
if __name__ == "__main__":
    for p in sorted(glob.glob("results/*.jsonl")):
        R = load(p); print(os.path.basename(p))
        for m, f in R.items():
            a = np.array([f[k]["test_acc"] for k in sorted(f)]) * 100
            t = np.mean([f[k]["time"] for k in f])
            print(f"  {m:14s} n={len(a):2d}  {a.mean():6.2f} ({a.std(ddof=1) if len(a)>1 else 0:5.2f})  time/fold {t:7.1f}s  params {f[min(f)].get('params','-')}")

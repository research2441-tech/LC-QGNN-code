"""10-fold benchmark driver.  Usage:
    python run_benchmark.py DATASET MODEL [--folds 1-10] [--L 2] [--Q 10] [--C 4] [--tag name] [--ablate X]
MODEL in {lcqgnn, mlp, gcn, sage, gin, gat, conegin, wl}.
Every model sees the identical outer folds (official GIN indices) and the identical inner
90/10 stratified split (seeded by fold index) used for epoch / hyper-parameter selection.
Results are appended as JSON lines to results/<DATASET>.jsonl.
"""
import argparse, json, os, time
import numpy as np
import jax
from sklearn.model_selection import train_test_split

from data_utils import load_tu, load_folds
from cones import ConeTable
import lcqgnn as Q
import baselines as B

CFG = {  # epochs, batch
    "MUTAG": (100, 32), "PTC": (100, 32), "NCI1": (60, 64), "PROTEINS": (40, 64), "IMDBBINARY": (60, 64), "REG3TRI": (100, 32),
}


def n_params(tree):
    return int(sum(np.size(l) for l in jax.tree_util.tree_leaves(tree)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("model")
    ap.add_argument("--folds", default="1-10")
    ap.add_argument("--L", type=int, default=2); ap.add_argument("--Q", type=int, default=10)
    ap.add_argument("--C", type=int, default=4); ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--ablate", default="none")   # none|nozz|sharedmix|zonly|nofield
    ap.add_argument("--tag", default=None); ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    graphs = load_tu(a.dataset); y = np.array([g.label for g in graphs])
    folds = load_folds(a.dataset)
    f0, f1 = map(int, a.folds.split("-"))
    epochs, batch = CFG[a.dataset]
    if a.epochs: epochs = a.epochs
    tag = a.tag or a.model
    os.makedirs(a.out, exist_ok=True)
    table = sim = bank = None
    if a.model in ("lcqgnn", "conegin"):
        t = time.time(); table = ConeTable(graphs, a.L, a.Q)
        if a.model == "lcqgnn":
            sim = Q.ConeSim(table)
        else:
            bank = B.ConeBank(table, a.Q)
        prep = time.time() - t
    for k in range(f0, f1 + 1):
        tr, te = folds[k - 1]
        tri, vai = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=k)
        t = time.time(); rec = {"dataset": a.dataset, "model": tag, "fold": k, "seed": a.seed}
        if a.model == "lcqgnn":
            p = Q.init_params(jax.random.PRNGKey(1000 * a.seed + k), a.C, a.L, graphs[0].feat.shape[1], int(y.max()) + 1)
            if a.ablate != "none":
                Q.ABLATE = a.ablate
            best, kappa = Q.train(p, sim, tri, vai, y, epochs=epochs, batch=batch, lr=a.lr or 0.02, seed=k)
            pr = Q.predict(best[2], sim, Q.GraphIndex(table, te), kappa)
            rec.update(params=n_params(best[2]), unique_cones=table.n_unique, total_cones=table.n_total, prep=prep,
                       L=a.L, Q=a.Q, C=a.C, ablate=a.ablate)
            np.save(os.path.join(a.out, f"params_{a.dataset}_{tag}_f{k}.npy"),
                    {"params": jax.tree_util.tree_map(np.asarray, best[2]), "kappa": kappa}, allow_pickle=True)
        elif a.model == "conegin":
            best, kappa = B.train_cone_gin(table, bank, y, tri, vai, seed=k, epochs=epochs, batch=batch, lr=a.lr or 0.01)
            pr = B.predict_cone_gin(best[2], table, bank, te, kappa)
            rec.update(params=n_params(best[2]), L=a.L, Q=a.Q)
        elif a.model == "wl":
            pred, sel = B.run_wl(graphs, y, tri, vai, te)
            pr = np.eye(int(y.max()) + 1)[pred]
            best = (sel[0], 0, None, sel[1])
        else:
            best, kappa, gg, sl = B.train_gnn(a.model, graphs, y, tri, vai, seed=k, epochs=epochs, batch=batch, lr=a.lr or 0.01)
            pr = B.predict_gnn(a.model, best[2], gg, te, kappa, sl)
            rec.update(params=n_params(best[2]))
        rec.update(test_acc=float(np.mean(pr.argmax(1) == y[te])), val_acc=float(best[0]),
                   best_epoch=str(best[3]), time=time.time() - t)
        with open(os.path.join(a.out, f"{a.dataset}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()

"""REG3-TRI: a 1-WL-indistinguishable benchmark.
Random connected 3-regular graphs with n in {10,12,14,16,18,20} nodes and a constant node
feature.  All graphs with the same n receive identical 1-WL colourings, so any
message-passing GNN bounded by 1-WL is at chance level.  The label is 1 when the number
of triangles exceeds the median triangle count of graphs with the same n (ties removed),
so the label is independent of n.  Written in the GIN text format with 10 stratified folds.
"""
import os
import numpy as np
import networkx as nx
from sklearn.model_selection import StratifiedKFold

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "REG3TRI")


def main(per_n=120, seed=7):
    rng = np.random.default_rng(seed)
    graphs, labels = [], []
    for n in (10, 12, 14, 16, 18, 20):
        pool = []
        while len(pool) < 4 * per_n:
            G = nx.random_regular_graph(3, n, seed=int(rng.integers(1 << 30)))
            if nx.is_connected(G):
                pool.append((G, sum(nx.triangles(G).values()) // 3))
        tri = np.array([t for _, t in pool]); med = np.median(tri)
        lo = [G for G, t in pool if t < med]; hi = [G for G, t in pool if t > med]
        k = min(per_n // 2, len(lo), len(hi))
        for G in lo[:k]:
            graphs.append(G); labels.append(0)
        for G in hi[:k]:
            graphs.append(G); labels.append(1)
    os.makedirs(os.path.join(OUT, "10fold_idx"), exist_ok=True)
    with open(os.path.join(OUT, "REG3TRI.txt"), "w") as f:
        f.write(f"{len(graphs)}\n")
        for G, y in zip(graphs, labels):
            f.write(f"{G.number_of_nodes()} {y}\n")
            for v in range(G.number_of_nodes()):
                nb = sorted(G.neighbors(v))
                f.write("0 %d %s\n" % (len(nb), " ".join(map(str, nb))))
    y = np.array(labels)
    skf = StratifiedKFold(10, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        np.savetxt(os.path.join(OUT, "10fold_idx", f"train_idx-{k}.txt"), tr, fmt="%d")
        np.savetxt(os.path.join(OUT, "10fold_idx", f"test_idx-{k}.txt"), te, fmt="%d")
    print(len(graphs), np.bincount(y))


if __name__ == "__main__":
    main()

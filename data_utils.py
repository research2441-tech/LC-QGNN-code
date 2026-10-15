"""Loading of TU benchmark graphs (GIN text format) and light-cone extraction.

Every graph is stored as an adjacency list plus integer node tags.  The official
10-fold test indices distributed with the GIN benchmark are used so that results
can be compared with prior work on identical partitions.
"""
import os
from collections import deque
import numpy as np

DATA_ROOT = os.environ.get("LCQGNN_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
DATASETS = ["MUTAG", "PTC", "PROTEINS", "NCI1", "IMDBBINARY"]


class Graph:
    __slots__ = ("n", "adj", "tags", "label", "feat")

    def __init__(self, n, adj, tags, label):
        self.n, self.adj, self.tags, self.label = n, adj, tags, label
        self.feat = None

    def edges(self):
        return [(i, j) for i in range(self.n) for j in self.adj[i] if i < j]


def load_tu(name, root=DATA_ROOT):
    path = os.path.join(root, name, f"{name}.txt")
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    ng = int(lines[0]); p = 1
    graphs, raw_labels, tagset = [], [], {}
    for _ in range(ng):
        n, lab = map(int, lines[p].split()[:2]); p += 1
        adj = [set() for _ in range(n)]; tags = []
        for i in range(n):
            row = lines[p].split(); p += 1
            t, k = int(row[0]), int(row[1])
            tags.append(t)
            for j in row[2:2 + k]:
                j = int(j)
                if j != i:
                    adj[i].add(j); adj[j].add(i)
        graphs.append(Graph(n, [sorted(a) for a in adj], tags, lab))
        raw_labels.append(lab)
        for t in tags:
            tagset.setdefault(t, len(tagset))
    lab_map = {l: i for i, l in enumerate(sorted(set(raw_labels)))}
    unlabeled = len(tagset) <= 1          # IMDB-B: no node labels
    if unlabeled:
        # structure-only datasets: use one-hot degree (capped) as the node tag, the
        # convention of the GIN benchmark
        maxd = max(max((len(a) for a in g.adj), default=0) for g in graphs)
        cap = min(maxd, 64)
        for g in graphs:
            g.feat = np.zeros((g.n, cap + 1), np.float32)
            for i in range(g.n):
                g.feat[i, min(len(g.adj[i]), cap)] = 1.0
    else:
        for g in graphs:
            g.feat = np.zeros((g.n, len(tagset)), np.float32)
            for i, t in enumerate(g.tags):
                g.feat[i, tagset[t]] = 1.0
    for g in graphs:
        g.label = lab_map[g.label]
    return graphs


def load_folds(name, root=DATA_ROOT):
    d = os.path.join(root, name, "10fold_idx")
    folds = []
    for k in range(1, 11):
        tr = np.loadtxt(os.path.join(d, f"train_idx-{k}.txt"), dtype=int)
        te = np.loadtxt(os.path.join(d, f"test_idx-{k}.txt"), dtype=int)
        folds.append((tr, te))
    return folds


def ball(g, roots, radius):
    """Nodes within graph distance <= radius of any root (BFS order, roots first)."""
    dist = {r: 0 for r in roots}
    q = deque(roots); order = list(roots)
    while q:
        u = q.popleft()
        if dist[u] == radius:
            continue
        for v in g.adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1; order.append(v); q.append(v)
    return order, dist


def cone(g, r, L, qmax=None):
    """Backward light cone of a one-body observable on root r for an L-layer circuit.

    Returns (nodes, edges_local, truncated) where nodes[0] == r.  Edges whose two
    endpoints both lie on the outer shell (distance L) are dropped: they commute with
    the Heisenberg-evolved observable at every layer, so removing them is exact.
    If qmax is given and the cone is larger, the outermost nodes are removed in BFS
    order (largest distance first, then lowest degree) and truncated=True.
    """
    order, dist = ball(g, [r], L)
    truncated = False
    if qmax is not None and len(order) > qmax:
        inner = [v for v in order if dist[v] < L]
        shell = [v for v in order if dist[v] == L]
        shell.sort(key=lambda v: -len(g.adj[v]))
        if len(inner) >= qmax:
            # even the (L-1)-ball overflows: keep BFS prefix
            order = order[:qmax]
        else:
            order = inner + shell[: qmax - len(inner)]
        truncated = True
    idx = {v: i for i, v in enumerate(order)}
    edges = []
    for v in order:
        for u in g.adj[v]:
            if u in idx and idx[v] < idx[u]:
                if dist[v] == L and dist[u] == L:
                    continue
                edges.append((idx[v], idx[u]))
    return order, edges, truncated

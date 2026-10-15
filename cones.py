"""Light-cone extraction, exact de-duplication and bucketing for the LC-QGNN simulator.

A cone is the induced sub-circuit that determines the one-body observables of a
root qubit after L layers.  Two cones with identical labelled structure (same node
features in the same canonical order and the same edge set) produce identical
expectation values, so they are simulated once.  The canonical order is a
refinement order; it is never used to merge cones that differ, therefore the
de-duplication is exact (it may only miss some isomorphic duplicates).
"""
import numpy as np
from data_utils import cone


def _canonical(nodes, edges, feats, L_dist=None):
    k = len(nodes)
    nb = [[] for _ in range(k)]
    for a, b in edges:
        nb[a].append(b); nb[b].append(a)
    tag = [int(np.argmax(feats[i])) if feats[i].sum() > 0 else -1 for i in range(k)]
    # colour refinement seeded by (is_root, tag, degree in cone)
    col = [(1 if i == 0 else 0, tag[i], len(nb[i])) for i in range(k)]
    for _ in range(3):
        col = [(col[i], tuple(sorted(col[j] for j in nb[i]))) for i in range(k)]
        uniq = {c: t for t, c in enumerate(sorted(set(col), key=repr))}
        col = [uniq[c] for c in col]
    order = sorted(range(k), key=lambda i: (i != 0, col[i], i))
    pos = {v: t for t, v in enumerate(order)}
    e2 = tuple(sorted(tuple(sorted((pos[a], pos[b]))) for a, b in edges))
    f2 = tuple(feats[v].tobytes() for v in order)
    return order, (k, e2, f2)


class ConeTable:
    """All cones of a list of graphs, de-duplicated and grouped by qubit count."""

    def __init__(self, graphs, L, qmax):
        self.L, self.qmax = L, qmax
        keys = {}
        self.node_cone = []            # per graph: array of unique-cone ids
        self.trunc = []                # per graph: fraction of truncated cones
        self.sizes = []                # per graph: cone sizes
        recs = []
        for g in graphs:
            ids, tr, sz = [], 0, []
            for r in range(g.n):
                nodes, edges, t = cone(g, r, L, qmax)
                feats = g.feat[nodes]
                order, key = _canonical(nodes, edges, feats)
                if key not in keys:
                    keys[key] = len(recs)
                    recs.append((len(nodes), key[1], feats[order]))
                ids.append(keys[key]); tr += t; sz.append(len(nodes))
            self.node_cone.append(np.array(ids, np.int32))
            self.trunc.append(tr / max(g.n, 1)); self.sizes.append(sz)
        self.n_unique = len(recs)
        self.n_total = sum(g.n for g in graphs)
        # buckets by padded size: a cone with k nodes is simulated on the smallest group
        # size g >= k; the g-k padding qubits have no edges and are never measured, so they
        # do not change the root read-out (exact), but they reduce the number of shapes.
        groups = [g for g in (5, 8, 10, 12, 14, 16, 18, 20) if g < qmax] + [qmax] if qmax else None
        self.buckets = {}
        for uid, (k, e, f) in enumerate(recs):
            gk = next(g for g in groups if g >= k) if groups else k
            self.buckets.setdefault(gk, []).append((uid, e, f))
        F = graphs[0].feat.shape[1]
        self.bucket_data = {}
        for k, items in sorted(self.buckets.items()):
            ids = np.array([u for u, _, _ in items], np.int32)
            A = np.zeros((len(items), k, k), np.int8)
            X = np.zeros((len(items), k, F), np.float32)
            for t, (_, e, f) in enumerate(items):
                for a, b in e:
                    A[t, a, b] = A[t, b, a] = 1
                X[t, :len(f)] = f
            self.bucket_data[k] = (ids, A, X)


def zz_energy(A):
    """sum_{i<j} A_ij z_i z_j over all 2^k computational basis states (big-endian)."""
    B, k, _ = A.shape
    s = np.arange(2 ** k)
    Z = 1 - 2 * ((s[:, None] >> (k - 1 - np.arange(k))[None, :]) & 1)   # (2^k, k)
    Z = Z.astype(np.float32)
    E = np.empty((B, 2 ** k), np.int8)
    for c in range(0, B, 512):
        Ac = A[c:c + 512].astype(np.float32)
        E[c:c + 512] = (np.einsum('bsl,sl->bs', np.einsum('sk,bkl->bsl', Z, Ac), Z) / 2).astype(np.int8)
    return E, Z

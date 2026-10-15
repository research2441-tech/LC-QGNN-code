"""Classical baselines trained under exactly the same protocol as LC-QGNN.

MLP-Sum   : structure-agnostic (node features pooled, no edges)
GCN, SAGE, GIN, GAT : message-passing GNNs (3 layers, width 64)
Cone-GIN  : GIN applied to the identical light cones used by LC-QGNN, root readout
WL-SVM    : Weisfeiler-Lehman subtree kernel with an SVM (h and C chosen on inner validation)
"""
import math
from collections import Counter
import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.svm import SVC

from lcqgnn import pool, head_apply


def _dense(key, i, o):
    return {"W": jax.random.normal(key, (i, o)) * math.sqrt(2.0 / i), "b": jnp.zeros(o)}


def _lin(p, x):
    return x @ p["W"] + p["b"]


def init_gnn(key, kind, F, n_cls, hidden=64, layers=3, head_hidden=32):
    keys = jax.random.split(key, 4 * layers + 4)
    P = {"layers": []}
    d = F
    for l in range(layers):
        k = keys[4 * l: 4 * l + 4]
        if kind == "gin":
            P["layers"].append({"m1": _dense(k[0], d, hidden), "m2": _dense(k[1], hidden, hidden), "eps": jnp.zeros(())})
        elif kind == "gcn":
            P["layers"].append({"w": _dense(k[0], d, hidden)})
        elif kind == "sage":
            P["layers"].append({"self": _dense(k[0], d, hidden), "nb": _dense(k[1], d, hidden)})
        elif kind == "gat":
            P["layers"].append({"w": _dense(k[0], d, hidden), "a_s": 0.1 * jax.random.normal(k[1], (4, hidden // 4)),
                                "a_d": 0.1 * jax.random.normal(k[2], (4, hidden // 4))})
        elif kind == "mlp":
            P["layers"].append({"w": _dense(k[0], d, hidden)})
        d = hidden
    d_in = 2 * hidden
    P["head"] = {"W1": jax.random.normal(keys[-2], (d_in, head_hidden)) * math.sqrt(2.0 / d_in), "b1": jnp.zeros(head_hidden),
                 "W2": jax.random.normal(keys[-1], (head_hidden, n_cls)) * math.sqrt(1.0 / head_hidden), "b2": jnp.zeros(n_cls)}
    return P


def gnn_nodes(P, kind, x, snd, rcv, n_nodes):
    h = x
    deg = jax.ops.segment_sum(jnp.ones_like(rcv, jnp.float32), rcv, n_nodes)
    for lp in P["layers"]:
        if kind == "gin":
            agg = jax.ops.segment_sum(h[snd], rcv, n_nodes)
            h = jax.nn.relu(_lin(lp["m2"], jax.nn.relu(_lin(lp["m1"], (1 + lp["eps"]) * h + agg))))
        elif kind == "gcn":
            dn = 1.0 / jnp.sqrt(deg + 1.0)
            hw = _lin(lp["w"], h)
            agg = jax.ops.segment_sum(hw[snd] * dn[snd][:, None], rcv, n_nodes) * dn[:, None] + hw * (dn ** 2)[:, None]
            h = jax.nn.relu(agg)
        elif kind == "sage":
            agg = jax.ops.segment_sum(h[snd], rcv, n_nodes) / jnp.maximum(deg, 1.0)[:, None]
            h = jax.nn.relu(_lin(lp["self"], h) + _lin(lp["nb"], agg))
        elif kind == "gat":
            hw = _lin(lp["w"], h).reshape(n_nodes, 4, -1)
            es = jnp.sum(hw * lp["a_s"], -1); ed = jnp.sum(hw * lp["a_d"], -1)
            # self loops are added as explicit edges by the batcher
            e = jax.nn.leaky_relu(es[snd] + ed[rcv], 0.2)
            emax = jax.ops.segment_max(e, rcv, n_nodes)
            w = jnp.exp(e - emax[rcv])
            z = jax.ops.segment_sum(w, rcv, n_nodes)
            msg = hw[snd] * (w / jnp.maximum(z[rcv], 1e-9))[:, :, None]
            h = jax.nn.elu(jax.ops.segment_sum(msg, rcv, n_nodes).reshape(n_nodes, -1))
        elif kind == "mlp":
            h = jax.nn.relu(_lin(lp["w"], h))
    return h


def make_batch(graphs, gids, n_slots, self_loops=False):
    xs, s, r, ng = [], [], [], []
    off = 0
    for t, gi in enumerate(gids):
        g = graphs[gi]
        xs.append(g.feat); ng.append(np.full(g.n, t))
        for i in range(g.n):
            for j in g.adj[i]:
                s.append(off + j); r.append(off + i)
            if self_loops:
                s.append(off + i); r.append(off + i)
        off += g.n
    x = np.concatenate(xs); ng = np.concatenate(ng)
    N = max(256, 1 << (off - 1).bit_length()); Em = max(len(s), 1)
    Ep = max(512, 1 << (Em - 1).bit_length())
    x = np.concatenate([x, np.zeros((N - off, x.shape[1]), np.float32)])
    ng = np.concatenate([ng, np.full(N - off, n_slots)]).astype(np.int32)
    s = np.array(s + [N - 1] * (Ep - len(s)), np.int32); r = np.array(r + [N - 1] * (Ep - len(r)), np.int32)
    return x, s, r, ng


def _gnn_loss(P, kind, x, s, r, ng, n, y, mask, kappa, wd):
    h = gnn_nodes(P, kind, x, s, r, x.shape[0])
    z = pool(h, ng, n + 1, kappa)[:n]
    logits = head_apply(P["head"], z)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    ce = jnp.sum(ce * mask) / jnp.sum(mask)
    reg = wd * sum(jnp.sum(l ** 2) for l in jax.tree_util.tree_leaves(P) if l.ndim == 2)
    return ce + reg, logits


_gnn_grad = jax.jit(jax.value_and_grad(_gnn_loss, has_aux=True), static_argnums=(1, 6))
_gnn_eval = jax.jit(_gnn_loss, static_argnums=(1, 6))


def train_gnn(kind, graphs, y, tr, va, seed=0, epochs=100, batch=32, lr=0.01, wd=1e-4):
    F = graphs[0].feat.shape[1]; ncls = int(y.max()) + 1
    P = init_gnn(jax.random.PRNGKey(seed), kind, F, ncls)
    kappa = float(np.mean([graphs[g].n for g in tr]))
    opt = optax.adam(lr); st = opt.init(P); rng = np.random.default_rng(seed)
    sl = kind == "gat"
    if kind == "mlp":
        G = graphs
        graphs = [type(g)(g.n, [[] for _ in range(g.n)], g.tags, g.label) for g in G]
        for g, g0 in zip(graphs, G):
            g.feat = g0.feat
    vb = make_batch(graphs, va, len(va), sl)
    best = (-1, 1e9, P, 0)
    for ep in range(epochs + 1):
        if ep > 0:
            perm = rng.permutation(tr)
            for s0 in range(0, len(perm), batch):
                bg = perm[s0:s0 + batch]
                x, s, r, ng = make_batch(graphs, bg, batch, sl)
                yb = np.zeros(batch, np.int32); yb[:len(bg)] = y[bg]
                mask = (np.arange(batch) < len(bg)).astype(np.float32)
                (_, _), g = _gnn_grad(P, kind, x, s, r, ng, batch, yb, mask, kappa, wd)
                up, st = opt.update(g, st, P); P = optax.apply_updates(P, up)
        vl, lg = _gnn_eval(P, kind, *vb, len(va), y[va].astype(np.int32), np.ones(len(va), np.float32), kappa, 0.0)
        acc = float(np.mean(np.argmax(np.asarray(lg), 1) == y[va]))
        if acc > best[0] or (acc == best[0] and float(vl) < best[1]):
            best = (acc, float(vl), P, ep)
    return best, kappa, graphs, sl


def predict_gnn(kind, P, graphs, ids, kappa, sl):
    out = []
    for s0 in range(0, len(ids), 256):
        b = ids[s0:s0 + 256]
        x, s, r, ng = make_batch(graphs, b, len(b), sl)
        _, lg = _gnn_eval(P, kind, x, s, r, ng, len(b), np.zeros(len(b), np.int32), np.ones(len(b), np.float32), kappa, 0.0)
        out.append(np.asarray(lg))
    return np.concatenate(out)


# ---------------- Cone-GIN: classical processing of the identical light cones ----------------

def init_cone_gin(key, F, n_cls, hidden=32, layers=3):
    ks = jax.random.split(key, 2 * layers + 2)
    P = {"layers": []}; d = F
    for l in range(layers):
        P["layers"].append({"m1": _dense(ks[2 * l], d, hidden), "m2": _dense(ks[2 * l + 1], hidden, hidden), "eps": jnp.zeros(())})
        d = hidden
    d_in = 2 * hidden
    P["head"] = {"W1": jax.random.normal(ks[-2], (d_in, 32)) * math.sqrt(2.0 / d_in), "b1": jnp.zeros(32),
                 "W2": jax.random.normal(ks[-1], (32, n_cls)) * math.sqrt(1.0 / 32), "b2": jnp.zeros(n_cls)}
    return P


def cone_gin_root(P, A, X):
    h = X; A = A.astype(jnp.float32)
    for lp in P["layers"]:
        h = jax.nn.relu(_lin(lp["m2"], jax.nn.relu(_lin(lp["m1"], (1 + lp["eps"]) * h + A @ h))))
    return h[:, 0, :]


class ConeBank:
    """Per-bucket dense cone tensors padded to one shape so a single jit suffices."""

    def __init__(self, table, kmax):
        U = table.n_unique; F = next(iter(table.bucket_data.values()))[2].shape[2]
        self.A = np.zeros((U, kmax, kmax), np.int8); self.X = np.zeros((U, kmax, F), np.float32)
        for k, (ids, A, X) in table.bucket_data.items():
            self.A[ids, :k, :k] = A; self.X[ids, :k] = X


def _cg_loss(P, A, X, uid_map, idx_uid, idx_graph, n, y, mask, kappa, wd):
    r = cone_gin_root(P, A, X)
    z = pool(r[uid_map[idx_uid]], idx_graph, n + 1, kappa)[:n]
    logits = head_apply(P["head"], z)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    ce = jnp.sum(ce * mask) / jnp.sum(mask)
    reg = wd * sum(jnp.sum(l ** 2) for l in jax.tree_util.tree_leaves(P) if l.ndim == 2)
    return ce + reg, logits


_cg_grad = jax.jit(jax.value_and_grad(_cg_loss, has_aux=True), static_argnums=(6,))
_cg_eval = jax.jit(_cg_loss, static_argnums=(6,))


def _cg_inputs(bank, gidx):
    uids = np.unique(gidx.node_uid)
    U = max(64, 1 << (len(uids) - 1).bit_length())
    pad = np.concatenate([uids, np.full(U - len(uids), uids[0])])
    m = np.zeros(bank.A.shape[0], np.int32); m[uids] = np.arange(len(uids))
    return bank.A[pad], bank.X[pad], m


def train_cone_gin(table, bank, y, tr, va, seed=0, epochs=100, batch=32, lr=0.01, wd=1e-4):
    from lcqgnn import GraphIndex
    F = bank.X.shape[2]; ncls = int(y.max()) + 1
    P = init_cone_gin(jax.random.PRNGKey(seed), F, ncls)
    kappa = float(np.mean([len(table.node_cone[g]) for g in tr]))
    opt = optax.adam(lr); st = opt.init(P); rng = np.random.default_rng(seed)
    vi = GraphIndex(table, va); vA, vX, vm = _cg_inputs(bank, vi)
    best = (-1, 1e9, P, 0)
    for ep in range(epochs + 1):
        if ep > 0:
            perm = rng.permutation(tr)
            for s0 in range(0, len(perm), batch):
                bg = perm[s0:s0 + batch]
                bi = GraphIndex(table, bg, n_pad=batch)
                A, X, m = _cg_inputs(bank, bi)
                yb = np.zeros(batch, np.int32); yb[:len(bg)] = y[bg]
                (_, _), g = _cg_grad(P, A, X, m, bi.node_uid, bi.node_graph, batch, yb, bi.mask, kappa, wd)
                up, st = opt.update(g, st, P); P = optax.apply_updates(P, up)
        vl, lg = _cg_eval(P, vA, vX, vm, vi.node_uid, vi.node_graph, vi.n, y[va].astype(np.int32),
                          np.ones(vi.n, np.float32), kappa, 0.0)
        acc = float(np.mean(np.argmax(np.asarray(lg)[:vi.n_real], 1) == y[va]))
        if acc > best[0] or (acc == best[0] and float(vl) < best[1]):
            best = (acc, float(vl), P, ep)
    return best, kappa


def predict_cone_gin(P, table, bank, ids, kappa):
    from lcqgnn import GraphIndex
    gi = GraphIndex(table, ids); A, X, m = _cg_inputs(bank, gi)
    _, lg = _cg_eval(P, A, X, m, gi.node_uid, gi.node_graph, gi.n, np.zeros(gi.n, np.int32),
                     np.ones(gi.n, np.float32), kappa, 0.0)
    return np.asarray(lg)[:gi.n_real]


# ---------------- WL subtree kernel ----------------

def wl_features(graphs, h):
    labels = [[int(np.argmax(g.feat[i])) for i in range(g.n)] for g in graphs]
    feats = [Counter(("0", l) for l in lab) for lab in labels]
    comp = {}
    for it in range(1, h + 1):
        new = []
        for g, lab in zip(graphs, labels):
            nl = []
            for i in range(g.n):
                sig = (lab[i], tuple(sorted(lab[j] for j in g.adj[i])))
                if sig not in comp:
                    comp[sig] = len(comp)
                nl.append(comp[sig])
            new.append(nl)
        labels = new
        for f, lab in zip(feats, labels):
            f.update((str(it), l) for l in lab)
    return feats


def wl_kernel_matrix(feats):
    vocab = {}
    for f in feats:
        for k in f:
            vocab.setdefault(k, len(vocab))
    M = np.zeros((len(feats), len(vocab)), np.float64)
    for t, f in enumerate(feats):
        for k, v in f.items():
            M[t, vocab[k]] = v
    K = M @ M.T
    d = np.sqrt(np.diag(K))
    return K / np.outer(d, d)


def run_wl(graphs, y, tr, va, te):
    best = (-1, None)
    Ks = {h: wl_kernel_matrix(wl_features(graphs, h)) for h in range(1, 6)}
    for h, K in Ks.items():
        for C in (0.01, 0.1, 1, 10, 100):
            clf = SVC(kernel="precomputed", C=C).fit(K[np.ix_(tr, tr)], y[tr])
            acc = np.mean(clf.predict(K[np.ix_(va, tr)]) == y[va])
            if acc > best[0]:
                best = (acc, (h, C))
    h, C = best[1]
    K = Ks[h]
    clf = SVC(kernel="precomputed", C=C).fit(K[np.ix_(tr, tr)], y[tr])
    return clf.predict(K[np.ix_(te, tr)]), best

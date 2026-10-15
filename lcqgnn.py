"""LC-QGNN: light-cone-exact quantum graph neural network (state-vector simulation in JAX).

Circuit on a cone with k qubits (qubit 0 is the root), for each channel c and layer l:
    U_l = prod_q exp(-i beta_{l,q} X_q) * exp(-i sum_q h_{l,q} Z_q) * exp(-i gamma_l sum_{(a,b)} Z_a Z_b)
with node-feature-dependent fields h_{l,q} = w_h^T x_q + b_h and mixing angles
beta_{l,q} = w_b^T x_q + b_b (parameters shared by all qubits -> permutation equivariance).
The root readout is (<X_0>, <Y_0>, <Z_0>) for every channel.
"""
import math
import numpy as np
import jax
import jax.numpy as jnp
import optax
from cones import zz_energy

jax.config.update("jax_default_matmul_precision", "highest")

ABLATE = "none"   # none | nozz | sharedmix | nofield | zonly  (read at trace time)


def init_params(key, C, L, F, n_cls, hidden=32):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    circ = {
        "gamma": jax.random.uniform(k1, (C, L), minval=0.1, maxval=0.8),
        "wh": 0.5 * jax.random.normal(k2, (C, L, F)),
        "bh": jnp.zeros((C, L)),
        "wb": 0.5 * jax.random.normal(k3, (C, L, F)),
        "bb": jnp.full((C, L), 0.4),
    }
    d_in = 6 * C
    head = {
        "W1": jax.random.normal(k4, (d_in, hidden)) * math.sqrt(2.0 / d_in),
        "b1": jnp.zeros(hidden),
        "W2": jax.random.normal(k5, (hidden, n_cls)) * math.sqrt(1.0 / hidden),
        "b2": jnp.zeros(n_cls),
    }
    return {"circ": circ, "head": head}


def _apply_rx(psi, beta, k):
    # psi: (B, 2^k) ; beta: (B, k)  -> exp(-i beta_q X_q) on every qubit q
    B = psi.shape[0]
    for q in range(k):
        p = psi.reshape(B, 2 ** q, 2, 2 ** (k - q - 1))
        c = jnp.cos(beta[:, q])[:, None, None]
        s = jnp.sin(beta[:, q])[:, None, None]
        a0, a1 = p[:, :, 0, :], p[:, :, 1, :]
        n0 = c * a0 - 1j * s * a1
        n1 = -1j * s * a0 + c * a1
        psi = jnp.stack([n0, n1], axis=2).reshape(B, 2 ** k)
    return psi


def cone_forward_channel(cp, E, X, Z, k):
    """cp: params of one channel {gamma:(L,), wh:(L,F), bh:(L,), wb:(L,F), bb:(L,)}."""
    B = E.shape[0]
    L = cp["gamma"].shape[0]
    psi = jnp.full((B, 2 ** k), 1.0 / math.sqrt(2 ** k), dtype=jnp.complex64)
    Ef = E.astype(jnp.float32)
    for l in range(L):
        h = X @ cp["wh"][l] + cp["bh"][l]                     # (B,k)
        if ABLATE == "nofield":
            h = h * 0.0
        g = cp["gamma"][l] * (0.0 if ABLATE == "nozz" else 1.0)
        phase = g * Ef + h @ Z.T                               # (B,2^k)
        psi = psi * jnp.exp(-1j * phase.astype(jnp.complex64))
        beta = (X @ cp["wb"][l]) * (0.0 if ABLATE == "sharedmix" else 1.0) + cp["bb"][l]
        psi = _apply_rx(psi, beta, k)
    p = psi.reshape(B, 2, 2 ** (k - 1))
    a0, a1 = p[:, 0, :], p[:, 1, :]
    z = jnp.sum(jnp.abs(a0) ** 2 - jnp.abs(a1) ** 2, axis=1)
    cross = jnp.sum(jnp.conj(a0) * a1, axis=1)
    if ABLATE == "zonly":
        return jnp.stack([0.0 * z, 0.0 * z, z], axis=1)
    return jnp.stack([2 * cross.real, 2 * cross.imag, z], axis=1)   # (B,3)


def _hmat(n):
    H = np.array([[1.0]], np.float32)
    h1 = np.array([[1, 1], [1, -1]], np.float32) / np.sqrt(2)
    for _ in range(n):
        H = np.kron(H, h1)
    return jnp.asarray(H)


def _hadamard_all(pr, pi, k):
    """H^{(x)k} applied to real and imaginary parts via a two-factor Kronecker split."""
    a = k // 2; b = k - a; B = pr.shape[0]
    Ha, Hb = _hmat(a), _hmat(b)
    def t(x):
        x = x.reshape(B, 2 ** a, 2 ** b)
        x = jnp.einsum("ij,bjk,lk->bil", Ha, x, Hb)
        return x.reshape(B, 2 ** k)
    return t(pr), t(pi)


def cone_forward_channel_h(cp, E, X, Z, k):
    """Same circuit as cone_forward_channel; the mixer prod_q exp(-i beta_q X_q) is applied as
    H^{(x)k} diag(exp(-i sum_q beta_q z_q)) H^{(x)k} on separate real/imaginary parts."""
    B = E.shape[0]
    L = cp["gamma"].shape[0]
    pr = jnp.full((B, 2 ** k), 1.0 / math.sqrt(2 ** k), dtype=jnp.float32)
    pi = jnp.zeros((B, 2 ** k), jnp.float32)
    Ef = E.astype(jnp.float32)
    for l in range(L):
        h = X @ cp["wh"][l] + cp["bh"][l]
        if ABLATE == "nofield":
            h = h * 0.0
        g = cp["gamma"][l] * (0.0 if ABLATE == "nozz" else 1.0)
        ph = g * Ef + h @ Z.T
        c, s = jnp.cos(ph), jnp.sin(ph)
        pr, pi = pr * c + pi * s, pi * c - pr * s
        beta = (X @ cp["wb"][l]) * (0.0 if ABLATE == "sharedmix" else 1.0) + cp["bb"][l]
        pb = beta @ Z.T
        pr, pi = _hadamard_all(pr, pi, k)
        c, s = jnp.cos(pb), jnp.sin(pb)
        pr, pi = pr * c + pi * s, pi * c - pr * s
        pr, pi = _hadamard_all(pr, pi, k)
    half = 2 ** (k - 1)
    r0, r1 = pr[:, :half], pr[:, half:]
    i0, i1 = pi[:, :half], pi[:, half:]
    z = jnp.sum(r0 ** 2 + i0 ** 2 - r1 ** 2 - i1 ** 2, axis=1)
    cre = jnp.sum(r0 * r1 + i0 * i1, axis=1)          # Re(conj(a0) a1)
    cim = jnp.sum(r0 * i1 - i0 * r1, axis=1)          # Im(conj(a0) a1)
    if ABLATE == "zonly":
        return jnp.stack([0.0 * z, 0.0 * z, z], axis=1)
    return jnp.stack([2 * cre, 2 * cim, z], axis=1)


USE_HADAMARD = True


def cone_forward(circ, E, X, Z, k):
    f = cone_forward_channel_h if USE_HADAMARD else cone_forward_channel
    out = jax.vmap(lambda cp: f(cp, E, X, Z, k))(circ)   # (C,B,3)
    return jnp.transpose(out, (1, 0, 2)).reshape(E.shape[0], -1)


_fwd_cache, _vjp_cache = {}, {}


def _fwd(k):
    if k not in _fwd_cache:
        _fwd_cache[k] = jax.jit(lambda circ, E, X, Z: cone_forward(circ, E, X, Z, k))
    return _fwd_cache[k]


def _vjp(k):
    if k not in _vjp_cache:
        def f(circ, E, X, Z, g):
            _, vjp = jax.vjp(lambda c: cone_forward(c, E, X, Z, k), circ)
            return vjp(g)[0]
        _vjp_cache[k] = jax.jit(f)
    return _vjp_cache[k]


class ConeSim:
    """Holds precomputed ZZ spectra for every bucket of a ConeTable and evaluates any
    subset of unique cones in fixed-shape (power-of-two padded) chunks."""

    def __init__(self, table, amp_budget=2 ** 19):
        self.table = table
        self.buckets = []
        self.where = np.zeros((table.n_unique, 2), np.int64)   # uid -> (bucket, row)
        for bi, (k, (ids, A, X)) in enumerate(sorted(table.bucket_data.items())):
            E, Z = zz_energy(A)
            cmax = max(1, amp_budget // (2 ** k))
            cmax = min(cmax, 1 << (len(ids) - 1).bit_length())   # one fixed chunk shape per k
            self.buckets.append((k, ids, E, X, Z.astype(np.float32), cmax))
            self.where[ids, 0] = bi
            self.where[ids, 1] = np.arange(len(ids))

    def _plan(self, uids):
        """Group requested uids by bucket and cut into padded chunks."""
        uids = np.asarray(uids)
        b_of = self.where[uids, 0]
        for bi in np.unique(b_of):
            k, ids, E, X, Z, cmax = self.buckets[bi]
            sel = uids[b_of == bi]
            rows = self.where[sel, 1]
            for s in range(0, len(sel), cmax):
                r = rows[s:s + cmax]; u = sel[s:s + cmax]
                m = len(r)
                P = next(c for c in (max(1, cmax // 16), max(1, cmax // 8), max(1, cmax // 4), max(1, cmax // 2), cmax) if c >= m)
                Ec = E[r]; Xc = X[r]
                if P > m:
                    Ec = np.concatenate([Ec, np.zeros((P - m,) + Ec.shape[1:], Ec.dtype)])
                    Xc = np.concatenate([Xc, np.zeros((P - m,) + Xc.shape[1:], Xc.dtype)])
                yield k, Z, u, m, Ec, Xc

    def phi(self, circ, uids=None):
        """Returns array (n_unique, 3C); rows not requested are zero."""
        if uids is None:
            uids = np.arange(self.table.n_unique)
        C = circ["gamma"].shape[0]
        out = np.zeros((self.table.n_unique, 3 * C), np.float32)
        for k, Z, u, m, Ec, Xc in self._plan(uids):
            out[u] = np.asarray(_fwd(k)(circ, Ec, Xc, Z))[:m]
        return out

    def grad(self, circ, gphi, uids=None):
        if uids is None:
            uids = np.arange(self.table.n_unique)
        tot = jax.tree_util.tree_map(jnp.zeros_like, circ)
        for k, Z, u, m, Ec, Xc in self._plan(uids):
            g = np.zeros((Ec.shape[0], gphi.shape[1]), np.float32)
            g[:m] = gphi[u]
            tot = jax.tree_util.tree_map(jnp.add, tot, _vjp(k)(circ, Ec, Xc, Z, g))
        return tot


# ---------------- graph-level head ----------------

def pool(phi_nodes, node_graph, n_graphs, kappa):
    s = jax.ops.segment_sum(phi_nodes, node_graph, n_graphs)
    cnt = jax.ops.segment_sum(jnp.ones(phi_nodes.shape[0]), node_graph, n_graphs)
    return jnp.concatenate([s / jnp.maximum(cnt, 1)[:, None], s / kappa], axis=1)


def head_apply(hp, z):
    h = jax.nn.relu(z @ hp["W1"] + hp["b1"])
    return h @ hp["W2"] + hp["b2"]


class GraphIndex:
    """Node->cone and node->graph maps for a set of graphs, padded to fixed shapes so that
    the jitted loss is compiled only a handful of times (padding nodes point to a dummy
    graph slot that is discarded)."""

    def __init__(self, table, graph_ids, n_pad=None):
        self.graph_ids = np.asarray(graph_ids)
        nc = [table.node_cone[g] for g in self.graph_ids]
        uid = np.concatenate(nc)
        gr = np.concatenate([np.full(len(a), t, np.int32) for t, a in enumerate(nc)])
        self.n_real = len(self.graph_ids)
        self.n = n_pad if n_pad is not None else self.n_real
        m = len(uid)
        M = max(256, 1 << (m - 1).bit_length())
        self.node_uid = np.concatenate([uid, np.full(M - m, uid[0])]).astype(np.int32)
        self.node_graph = np.concatenate([gr, np.full(M - m, self.n)]).astype(np.int32)
        self.mask = (np.arange(self.n) < self.n_real).astype(np.float32)


def _loss_fn(hp, phi_u, idx_uid, idx_graph, n, y, mask, kappa, wd):
    z = pool(phi_u[idx_uid], idx_graph, n + 1, kappa)[:n]
    logits = head_apply(hp, z)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    ce = jnp.sum(ce * mask) / jnp.sum(mask)
    reg = wd * sum(jnp.sum(v ** 2) for kk, v in hp.items() if kk.startswith("W"))
    return ce + reg, logits


_loss_grad = jax.jit(jax.value_and_grad(_loss_fn, argnums=(0, 1), has_aux=True), static_argnums=(4,))


@jax.jit
def _logits(hp, phi_nodes, idx_graph, kappa, n1):
    return head_apply(hp, pool(phi_nodes, idx_graph, n1.shape[0], kappa))


def predict(params, sim, gidx, kappa):
    phi_u = jnp.asarray(sim.phi(params["circ"], np.unique(gidx.node_uid)))
    out = _logits(params["head"], phi_u[gidx.node_uid], gidx.node_graph, kappa, jnp.zeros(gidx.n + 1))
    return np.asarray(out)[: gidx.n_real]


def train(params, sim, tr_graphs, va_graphs, y_all, epochs=100, batch=64, lr=0.02, wd=1e-4,
          seed=0, log=None):
    """Mini-batch Adam; the epoch with the best inner-validation accuracy (ties broken by
    validation loss) is retained, following the hold-out model-selection protocol."""
    table = sim.table
    kappa = float(np.mean([len(table.node_cone[g]) for g in tr_graphs]))
    opt = optax.adam(lr)
    state = opt.init(params)
    rng = np.random.default_rng(seed)
    va_idx = GraphIndex(table, va_graphs); yva = y_all[va_graphs]
    best = (-1.0, 1e9, params, 0)
    tr_graphs = np.asarray(tr_graphs)
    for ep in range(epochs + 1):
        if ep > 0:
            perm = rng.permutation(tr_graphs)
            for s in range(0, len(perm), batch):
                bg = perm[s:s + batch]
                bi = GraphIndex(table, bg, n_pad=batch)
                uids = np.unique(bi.node_uid)
                phi_u = jnp.asarray(sim.phi(params["circ"], uids))
                yb = np.zeros(batch, np.int32); yb[: len(bg)] = y_all[bg]
                (loss, _), (g_head, g_phi) = _loss_grad(params["head"], phi_u, bi.node_uid, bi.node_graph,
                                                         batch, jnp.asarray(yb), jnp.asarray(bi.mask), kappa, wd)
                g_circ = sim.grad(params["circ"], np.asarray(g_phi), uids)
                upd, state = opt.update({"circ": g_circ, "head": g_head}, state, params)
                params = optax.apply_updates(params, upd)
        lv = predict(params, sim, va_idx, kappa)
        acc = float(np.mean(np.argmax(lv, 1) == yva))
        vloss = float(optax.softmax_cross_entropy_with_integer_labels(jnp.asarray(lv), jnp.asarray(yva)).mean())
        if acc > best[0] or (acc == best[0] and vloss < best[1]):
            best = (acc, vloss, jax.tree_util.tree_map(lambda a: a, params), ep)
        if log is not None:
            log.append((ep, acc, vloss))
    return best, kappa

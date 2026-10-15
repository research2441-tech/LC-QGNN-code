"""Gradient-variance scaling: root-anchored (light-cone) cost vs a global cost.

For random connected 3-regular graphs, the variance (over uniformly random circuit
parameters) of d<X_r>/d gamma_1 is computed through the light cone of r, so graphs far
larger than any state vector can be handled.  For comparison, the global cost
<prod_i Z_i>, which has no light-cone reduction, is evaluated on the full graph for
n <= 14.  Output: results/trainability.json
"""
import json, math
import numpy as np
import networkx as nx
import jax, jax.numpy as jnp
from data_utils import Graph, cone
from cones import zz_energy
from lcqgnn import cone_forward_channel

jax.config.update("jax_enable_x64", False)


def nxg(n, seed):
    while True:
        G = nx.random_regular_graph(3, n, seed=seed)
        if nx.is_connected(G):
            break
        seed += 1000
    g = Graph(n, [sorted(G.neighbors(v)) for v in range(n)], [0] * n, 0)
    g.feat = np.ones((n, 1), np.float32)
    return g


def rand_params(key, L):
    k = jax.random.split(key, 3)
    return {"gamma": jax.random.uniform(k[0], (L,), maxval=2 * math.pi),
            "wh": jnp.zeros((L, 1)), "bh": jax.random.uniform(k[1], (L,), maxval=2 * math.pi),
            "wb": jnp.zeros((L, 1)), "bb": jax.random.uniform(k[2], (L,), maxval=2 * math.pi)}


def local_var(n, L, draws=400, seed=0):
    g = nxg(n, seed)
    nodes, edges, _ = cone(g, 0, L)
    k = len(nodes)
    A = np.zeros((1, k, k), np.int8)
    for a, b in edges:
        A[0, a, b] = A[0, b, a] = 1
    E, Z = zz_energy(A)
    X = jnp.ones((1, k, 1)); Zj = jnp.asarray(Z, jnp.float32)
    f = lambda cp: cone_forward_channel(cp, E, X, Zj, k)[0, 0]
    g1 = jax.jit(jax.grad(f))
    ks = jax.random.split(jax.random.PRNGKey(seed), draws)
    gs = np.array([float(g1(rand_params(kk, L))["gamma"][0]) for kk in ks])
    return float(np.var(gs)), k


def global_var(n, L, draws=200, seed=0):
    g = nxg(n, seed)
    A = np.zeros((1, n, n), np.int8)
    for a in range(n):
        for b in g.adj[a]:
            A[0, a, b] = 1
    E, Z = zz_energy(A)
    Zj = jnp.asarray(Z, jnp.float32); X = jnp.ones((1, n, 1))
    from lcqgnn import _apply_rx

    def cost(cp):
        psi = jnp.full((1, 2 ** n), 1 / math.sqrt(2 ** n), dtype=jnp.complex64)
        for l in range(L):
            psi = psi * jnp.exp(-1j * (cp["gamma"][l] * E.astype(jnp.float32) + cp["bh"][l] * jnp.sum(Zj, 1)[None]).astype(jnp.complex64))
            psi = _apply_rx(psi, jnp.full((1, n), cp["bb"][l]), n)
        # global parity observable prod_i Z_i (no light-cone reduction possible)
        par = jnp.prod(Zj, 1)
        return jnp.sum(jnp.abs(psi[0]) ** 2 * par)
    g1 = jax.jit(jax.grad(cost))
    ks = jax.random.split(jax.random.PRNGKey(seed), draws)
    gs = np.array([float(g1(rand_params(kk, L))["gamma"][0]) for kk in ks])
    return float(np.var(gs))


if __name__ == "__main__":
    R = {"local": {}, "global": {}}
    for L in (1, 2, 3):
        R["local"][L] = []
        for n in ((8, 16, 32, 64, 128, 256, 512) if L < 3 else (8, 16, 32)):
            v, k = local_var(n, L, draws=400 if L < 3 else 60)
            R["local"][L].append((n, v, k)); print("local", L, n, v, k, flush=True)
        R["global"][L] = []
        for n in (4, 6, 8, 10, 12, 14):
            v = global_var(n, L)
            R["global"][L].append((n, v)); print("global", L, n, v, flush=True)
        json.dump(R, open("results/trainability.json", "w"), indent=1)
    json.dump(R, open("results/trainability.json", "w"), indent=1)

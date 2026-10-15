"""Grayscale schematic figures: Fig. 1 (framework) and Fig. 2 (light cone, cone circuit, covering)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif", "font.size": 7.5})


def box(ax, x, y, w, h, title, body=None, fill="white", lw=0.9, ls="-", tfs=6.6, bfs=6.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.08",
                                fc=fill, ec="black", lw=lw, ls=ls))
    if body:
        ax.text(x + w / 2, y + h * 0.71, title, ha="center", va="center", fontsize=tfs, fontweight="bold", linespacing=1.15)
        ax.text(x + w / 2, y + h * 0.27, body, ha="center", va="center", fontsize=bfs, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tfs, fontweight="bold")
    return (x, y, w, h)


def arrow(ax, p, q, dashed=False, lw=0.9):
    a = FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw, color="black",
                        ls=(0, (3, 2)) if dashed else "-", shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def seg(ax, xs, ys, dashed=False, lw=0.9):
    ax.plot(xs, ys, color="black", lw=lw, ls=(0, (3, 2)) if dashed else "-", solid_capstyle="butt")


def fig1(path):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.set_xlim(0, 16.1); ax.set_ylim(0, 9.4); ax.axis("off")
    W, H, step = 2.3, 1.6, 2.7
    xs = [0.15 + i * step for i in range(6)]
    yT, yD = 6.55, 1.45
    cy = lambda y: y + H / 2
    ax.text(0.15, 9.05, "(a) Training and exact classical simulation", fontsize=7.6, fontweight="bold", ha="left")
    top = [("Input\ngraph", "$G=(V,E)$, $\\mathbf{x}_v$\nqubit per node", "white"),
           ("Light-cone\nextraction", "$\\mathcal{B}_L(r)$, cap $Q$\nexact support", "white"),
           ("Cone\nde-duplication", "canonical key\n$U\\ll|V|$ circuits", "white"),
           ("Equivariant\ncone circuit", "$[R_X R_Z\\,ZZ]^{L}$\n$C$ channels", "#e6e6e6"),
           ("Root\nread-out", "$\\langle X\\rangle,\\langle Y\\rangle,\\langle Z\\rangle$\n$\\phi_r\\in\\mathbb{R}^{3C}$", "white"),
           ("Pooling\nand head", "mean, sum$/\\kappa$\nMLP $\\to\\hat{y}_G$", "white")]
    for x, (t, b, f) in zip(xs, top):
        box(ax, x, yT, W, H, t, b, fill=f)
    for i in range(5):
        arrow(ax, (xs[i] + W, cy(yT)), (xs[i + 1], cy(yT)))
    # loss and reverse-mode gradient
    yL = 4.35; HL = 0.95
    box(ax, xs[5], yL, W, HL, "Loss $\\mathcal{L}$")
    arrow(ax, (xs[5] + W / 2, yT), (xs[5] + W / 2, yL + HL))
    yg = yL + HL / 2
    xc = xs[3] + W / 2
    arrow(ax, (xs[5], yg), (xs[4] + W / 2 + 0.02, yg), dashed=True)
    seg(ax, [xs[4] + W / 2, xc], [yg, yg], dashed=True)
    arrow(ax, (xc, yg), (xc, yT), dashed=True)
    ax.text((xs[5] + xs[4] + W / 2) / 2, yg + 0.25, "$\\partial\\mathcal{L}/\\partial\\phi_u$", ha="center", fontsize=6.6)
    ax.text(xc + 0.15, yg + 0.85, "VJP per\nunique cone", ha="left", va="center", fontsize=6.3)
    ax.text((xs[4] + W / 2 + xc) / 2 + 0.6, yg - 0.32, "reverse-mode gradient (dashed)", ha="center", fontsize=6.2, style="italic")
    # separator
    ax.plot([0.1, 16.0], [3.75, 3.75], color="black", lw=0.5, ls=(0, (1, 2)))
    ax.text(0.15, 3.42, "(b) Deployment", fontsize=7.6, fontweight="bold", ha="left")
    bot = [("Test\ngraph", "unseen $G$\nnode features", "white"),
           ("Covering\ncompiler", "pack cones into\n$J$ circuits $S_j$", "#e6e6e6"),
           ("Shared\ncircuits", "$J\\times C\\times 3$ strata\n(X, Y, Z bases)", "white"),
           ("Stratified\nshots", "pilot + Neyman\n$N_s\\propto\\hat{\\sigma}_s$", "#e6e6e6"),
           ("Root\nestimates", "$\\hat{\\phi}_r$ from\nshared records", "white"),
           ("Certified\ndecision", "stop: $|\\hat{m}|\\geq z\\,\\widehat{\\mathrm{SE}}$\nelse add shots", "white")]
    for x, (t, b, f) in zip(xs, bot):
        box(ax, x, yD, W, H, t, b, fill=f)
    for i in range(5):
        arrow(ax, (xs[i] + W, cy(yD)), (xs[i + 1], cy(yD)))
    # trained parameters theta* flow from the circuit to the shared circuits (solid elbow)
    xt = xs[3] + 0.35
    ye = 3.98
    seg(ax, [xt, xt], [yT, ye])
    seg(ax, [xt, xs[2] + W / 2], [ye, ye])
    arrow(ax, (xs[2] + W / 2, ye), (xs[2] + W / 2, yD + H))
    ax.text(xt - 0.12, (yT + ye) / 2 + 0.35, "trained\n$\\theta^{\\star}$", ha="right", va="center", fontsize=6.4)
    # sequential feedback (dashed) below the deployment lane
    yb = 0.75
    seg(ax, [xs[5] + W / 2, xs[5] + W / 2], [yD, yb], dashed=True)
    seg(ax, [xs[5] + W / 2, xs[3] + W / 2], [yb, yb], dashed=True)
    arrow(ax, (xs[3] + W / 2, yb), (xs[3] + W / 2, yD), dashed=True)
    ax.text((xs[5] + xs[3]) / 2 + W / 2, yb - 0.32, "more shots while the margin is not certified", ha="center", fontsize=6.3, style="italic")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def fig2(path):
    fig = plt.figure(figsize=(7.4, 2.75))
    # (a) light cone
    ax = fig.add_axes([0.0, 0.0, 0.30, 0.92]); ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-3.3, 3.3); ax.set_ylim(-3.3, 3.5)
    for rr, ls in ((1.15, (0, (2, 2))), (2.3, (0, (2, 2)))):
        ax.add_patch(Circle((0, 0), rr, fill=False, ec="0.55", lw=0.6, ls=ls))
    P = {"r": (0, 0), "a": (1.15, 0), "b": (-0.575, 1.0), "c": (-0.575, -1.0),
         "d": (2.3, 0.45), "e": (2.0, -1.1), "f": (-1.0, 2.07), "g": (-2.25, 0.4), "h": (-1.4, -1.83),
         "o1": (3.05, 1.6), "o2": (-2.95, 2.0)}
    edges = [("r", "a"), ("r", "b"), ("r", "c"), ("b", "c"), ("a", "d"), ("a", "e"), ("b", "f"), ("b", "g"), ("c", "h")]
    drop = [("d", "e"), ("g", "f")]
    out = [("d", "o1"), ("f", "o2")]
    for u, v in edges:
        ax.plot(*zip(P[u], P[v]), color="black", lw=0.9, zorder=1)
    for u, v in drop:
        ax.plot(*zip(P[u], P[v]), color="black", lw=0.9, ls=(0, (1, 1.6)), zorder=1)
    for u, v in out:
        ax.plot(*zip(P[u], P[v]), color="0.6", lw=0.7, zorder=1)
    for k, (x, y) in P.items():
        if k == "r":
            ax.add_patch(Circle((x, y), 0.26, fc="black", ec="black", zorder=3))
            ax.text(x, y, "$r$", color="white", ha="center", va="center", fontsize=7.5, zorder=4)
        elif k.startswith("o"):
            ax.add_patch(Circle((x, y), 0.2, fc="white", ec="0.6", lw=0.7, ls=(0, (2, 1)), zorder=3))
        else:
            fc = "0.75" if k in "abc" else "white"
            ax.add_patch(Circle((x, y), 0.2, fc=fc, ec="black", lw=0.8, zorder=3))
            if k in "abc":
                off = {"a": (0.05, -0.5), "b": (-0.5, 0.05), "c": (-0.5, -0.1)}[k]
                ax.text(x + off[0], y + off[1], f"${k}$", ha="center", va="center", fontsize=7.0)
    ax.text(0, -3.15, "(a) cone $\\mathcal{B}_2(r)$: solid edges kept,\ndotted shell–shell edges dropped", ha="center", va="center", fontsize=6.8)
    ax.text(1.3, 0.75, "$d=1$", fontsize=6.3); ax.text(1.45, 2.45, "$d=2$", fontsize=6.3)
    # (b) one layer of the cone circuit for 4 qubits
    bx = fig.add_axes([0.33, 0.04, 0.42, 0.88]); bx.axis("off"); bx.set_xlim(0, 10.6); bx.set_ylim(-0.95, 4.6)
    labels = ["$r$", "$a$", "$b$", "$c$"]
    ys = [3.6, 2.5, 1.4, 0.3]
    for q, y in enumerate(ys):
        bx.plot([0.95, 9.6], [y, y], color="black", lw=0.7, zorder=0)
        bx.text(0.15, y, "$|+\\rangle$", va="center", fontsize=7.2)
        bx.text(10.1, y, labels[q], va="center", ha="center", fontsize=7.4)
    # ZZ gates on edges r-a, r-b, r-c, b-c (commuting)
    for x0, (i, j) in zip((1.35, 2.05, 2.75, 3.45), ((0, 1), (0, 2), (0, 3), (2, 3))):
        bx.plot([x0, x0], [ys[i], ys[j]], color="black", lw=0.8, zorder=1)
        bx.plot([x0, x0], [ys[i], ys[j]], ls="none", marker="o", ms=3.6, color="black", zorder=2)
    bx.text(2.4, 4.2, "$e^{-i\\gamma_\\ell Z_uZ_v}$ on cone edges", ha="center", fontsize=6.6)
    for q, y in enumerate(ys):
        bx.add_patch(Rectangle((4.15, y - 0.3), 1.45, 0.6, fc="white", ec="black", lw=0.8, zorder=2))
        bx.text(4.875, y, "$R_Z(2h_{\\ell,v})$", ha="center", va="center", fontsize=6.0, zorder=3)
        bx.add_patch(Rectangle((6.0, y - 0.3), 1.45, 0.6, fc="0.85", ec="black", lw=0.8, zorder=2))
        bx.text(6.725, y, "$R_X(2\\beta_{\\ell,v})$", ha="center", va="center", fontsize=6.0, zorder=3)
    bx.add_patch(Rectangle((1.05, -0.22 - 0.1), 6.6, 4.25, fill=False, ec="0.4", lw=0.6, ls=(0, (3, 2))))
    bx.text(4.35, -0.62, "layer $\\ell$, repeated $L$ times", ha="center", fontsize=6.6)
    bx.add_patch(Rectangle((8.15, ys[0] - 0.32), 0.95, 0.64, fc="white", ec="black", lw=0.8, zorder=2))
    bx.text(8.625, ys[0], "X/Y/Z", ha="center", va="center", fontsize=5.8, zorder=3)
    bx.text(5.3, -0.95, "(b) one layer on $r$ and its neighbours; only $r$ is read out", ha="center", va="center", fontsize=6.8)
    # (c) covering
    cx = fig.add_axes([0.77, 0.0, 0.23, 0.92]); cx.set_aspect("equal"); cx.axis("off"); cx.set_xlim(-0.4, 6.2); cx.set_ylim(-1.9, 5.0)
    pts = [(0.4, 2.0), (1.3, 3.1), (1.3, 0.9), (2.4, 2.0), (3.5, 3.1), (3.5, 0.9), (4.5, 2.0), (5.5, 3.1), (5.5, 0.9)]
    E2 = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 6), (5, 6), (6, 7), (6, 8)]
    for u, v in E2:
        cx.plot(*zip(pts[u], pts[v]), color="black", lw=0.8, zorder=1)
    for i, (x, y) in enumerate(pts):
        fc = "black" if i in (3, 4) else "white"
        cx.add_patch(Circle((x, y), 0.22, fc=fc, ec="black", lw=0.8, zorder=3))
    cx.add_patch(FancyBboxPatch((-0.05, 0.45), 5.12, 3.1, boxstyle="round,pad=0.05,rounding_size=0.3", fill=False, ec="black", lw=0.8, ls=(0, (3, 2))))
    cx.text(2.45, 4.15, "$S_j$: one circuit, $|S_j|\\leq Q_{\\mathrm{hw}}$", ha="center", fontsize=6.4)
    cx.text(2.9, -0.15, "filled roots share $S_j$ ($L=1$ shown)", ha="center", fontsize=6.4)
    cx.text(2.9, -1.35, "(c) light-cone covering", ha="center", va="center", fontsize=6.8)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs("figs", exist_ok=True)
    fig1("figs/fig1_framework.png"); fig2("figs/fig2_lightcone.png")

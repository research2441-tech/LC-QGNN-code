"""Grayscale result figures (Figs. 3-5 of the manuscript).

Usage: python fig_results.py   -> figs/fig3_folds.png, figs/fig4_shots.png, figs/fig5_train.png
Identity is carried by marker shape and line style, never by colour alone, so every
figure survives black-and-white printing.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from summarize import load

plt.rcParams.update({"font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif", "font.size": 7.5,
                     "axes.linewidth": 0.7, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})
NAMES = {"lcqgnn": "LC-QGNN", "wl": "WL-SVM", "mlp": "MLP-Sum", "gcn": "GCN", "sage": "SAGE", "gin": "GIN",
         "gat": "GAT", "conegin": "Cone-GIN"}
ORDER = ["lcqgnn", "conegin", "gin", "gcn", "sage", "gat", "mlp", "wl"]
DSN = {"MUTAG": "MUTAG", "PTC": "PTC-MR", "NCI1": "NCI1", "PROTEINS": "PROTEINS", "IMDBBINARY": "IMDB-B", "REG3TRI": "REG3-TRI"}
GRID = dict(color="0.88", lw=0.5)


def fig_folds(path, datasets=("MUTAG", "PTC", "NCI1", "REG3TRI")):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.3))
    for ax, ds in zip(axes.ravel(), datasets):
        R = load(f"results/{ds}.jsonl")
        ms = [m for m in ORDER if m in R and len(R[m]) == 10]
        data = [np.array([R[m][k]["test_acc"] for k in sorted(R[m])]) * 100 for m in ms]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=3.2),
                        medianprops=dict(color="black", lw=1.0), whiskerprops=dict(lw=0.7), capprops=dict(lw=0.7),
                        flierprops=dict(marker="o", markersize=2.2, markerfacecolor="0.4", markeredgecolor="0.4"))
        for i, b in enumerate(bp["boxes"]):
            b.set_facecolor("0.45" if ms[i] == "lcqgnn" else ("0.8" if ms[i] == "conegin" else "white"))
            b.set_edgecolor("black"); b.set_linewidth(0.7)
        ax.set_xticks(range(1, len(ms) + 1)); ax.set_xticklabels([NAMES[m] for m in ms], rotation=40, ha="right", fontsize=6.6)
        ax.set_title(DSN[ds], fontsize=8, pad=4)
        ax.grid(axis="y", **GRID); ax.set_axisbelow(True)
        ax.set_ylabel("Test accuracy (%)", fontsize=7)
        ax.tick_params(axis="y", labelsize=6.6)
        if ds == "REG3TRI":
            ax.axhline(50, color="0.35", lw=0.7, ls=":")
            ax.text(0.55, 51.5, "chance", fontsize=6, color="0.25", ha="left", va="bottom")
    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04); plt.close(fig)


STY = {"PER-NODE": ("o", "-", "white", "one circuit per node, uniform"),
       "COV-UNI": ("s", "--", "white", "covering, uniform"),
       "COV-NEY": ("^", "-", "black", "covering, Neyman (pilot)"),
       "COV-ORA": ("x", ":", "black", "covering, Neyman (oracle)")}


def fig_shots(path, datasets=("MUTAG", "PTC", "REG3TRI")):
    ds_list = [d for d in datasets if os.path.exists(f"results/shots_{d}.json")]
    fig, axes = plt.subplots(1, len(ds_list), figsize=(7.2, 2.5), squeeze=False)
    for j, ds in enumerate(ds_list):
        S = json.load(open(f"results/shots_{ds}.json")); ax = axes[0, j]
        for k, (mk, ls, fc, lb) in STY.items():
            Ms = sorted(int(x) for x in S[k])
            ax.plot(Ms, [100 * S[k][str(M)]["agree"] for M in Ms], marker=mk, ls=ls, color="black", lw=0.8, ms=3.4, mfc=fc, label=lb)
        seq = S["SEQ"]
        ax.plot([seq["mean_shots"]], [100 * seq["agree"]], marker="*", ms=7, color="black", ls="none", label="sequential rule ($z=2$)")
        ax.set_xscale("log", base=2); ax.set_xlabel("Shots per graph $M$"); ax.set_title(DSN[ds], fontsize=8, pad=4)
        if j == 0:
            ax.set_ylabel("Agreement with exact decision (%)", fontsize=7)
        ax.grid(**GRID)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=6.2, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.6)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04); plt.close(fig)


def fig_train(path):
    T = json.load(open("results/trainability.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    mk = {"1": "o", "2": "s", "3": "^"}
    ax = axes[0]
    for L, v in T["local"].items():
        ax.plot([x[0] for x in v], [x[1] for x in v], marker=mk[L], color="black", lw=0.8, ms=3.5, mfc="white",
                label=f"root read-out, $L={L}$")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Graph size $n$ (3-regular)"); ax.set_ylabel("Var$[\\partial_{\\gamma_1}\\langle X_r\\rangle]$")
    ax.grid(**GRID); ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.set_title("(a) light-cone read-out", fontsize=7.8, pad=4)
    ax = axes[1]
    for L, v in T["global"].items():
        ax.plot([x[0] for x in v], [x[1] for x in v], marker=mk[L], color="black", lw=0.8, ls="--", ms=3.5,
                mfc="black", label=f"global parity, $L={L}$")
    ax.set_yscale("log")
    ax.set_xlabel("Graph size $n$ (full register)"); ax.set_ylabel("Var$[\\partial_{\\gamma_1}\\langle Z^{\\otimes n}\\rangle]$")
    ax.grid(**GRID); ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.set_title("(b) global cost", fontsize=7.8, pad=4)
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator
    for ax in axes:
        ax.set_yscale("log"); ax.set_ylim(0.08, 3.0)
        ax.yaxis.set_major_locator(FixedLocator([0.1, 0.2, 0.5, 1, 2]))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%g")); ax.yaxis.set_minor_locator(NullLocator())
    axes[0].xaxis.set_major_locator(FixedLocator([8, 16, 32, 64, 128, 256, 512]))
    axes[0].xaxis.set_major_formatter(FormatStrFormatter("%d")); axes[0].xaxis.set_minor_locator(NullLocator())
    fig.tight_layout(w_pad=2.4)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04); plt.close(fig)


if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    fig_folds("figs/fig3_folds.png")
    fig_shots("figs/fig4_shots.png")
    fig_train("figs/fig5_train.png")

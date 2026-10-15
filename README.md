# LC-QGNN: Light-Cone-Exact Quantum Graph Neural Networks

Reference implementation of the manuscript *"Light-Cone-Exact Quantum Graph Neural Networks with Shared-Circuit Covering and Certified Shot Allocation for Graph Classification"*.

All code lives in this single flat folder. `reproduce.sh` regenerates every table and figure of the paper from the raw data.

## What the code does

| File | Purpose |
|---|---|
| `data_utils.py` | Loads TU benchmarks (GIN text format) and their official 10-fold indices; balls and light cones |
| `cones.py` | Cone extraction, exact de-duplication by canonical labelled key, size-group padding, ZZ spectra |
| `lcqgnn.py` | LC-QGNN model: JAX state-vector simulator of equivariant ZZ-field-mixer cone circuits, root read-out, pooling, MLP head, two-pass (VJP) gradients, mini-batch training |
| `covering.py` | Light-cone covering compiler (Algorithm 1) and a NumPy state-vector simulator with X/Y/Z measurement |
| `baselines.py` | MLP-Sum, GCN, GraphSAGE, GIN, GAT, Cone-GIN (GIN on the identical cones) and the WL subtree kernel SVM |
| `run_benchmark.py` | 10-fold driver: identical outer folds, inner 90/10 split and selection rule for every model |
| `make_synthetic.py` | REG3-TRI: 1-WL-indistinguishable 3-regular graphs labelled by triangle count (seed 7, 10 stratified folds) |
| `exp_covering.py` | Covering statistics (`stats`), full-register exactness check (`full`), covering-circuit exactness check (`exact`) |
| `exp_identities.py` | Numerical check of the closed-form one-layer read-out, the two-body correlator and the C6 / 2xC3 separation |
| `exp_shots.py` | Finite-shot study: per-node, covering-uniform, covering-Neyman (pilot), oracle; sequential certified rule (Algorithm 2); optional read-out bit flips |
| `exp_trainability.py` | Gradient variance of light-cone read-outs (n up to 512) vs a global parity cost |
| `stats.py` | Mean, SD, 95% t-interval; Wilcoxon signed-rank with Holm correction; rank-biserial effect size; Friedman test |
| `data_stats.py`, `summarize.py` | Data-set and cone statistics; quick result summaries |
| `fig_arch.py`, `fig_results.py` | Figures 1-5 (grayscale) |
| `results_summary.json` | Every per-fold accuracy and every summary number reported in the paper, for comparison with a re-run |

## Installation

```
python -m pip install -r requirements.txt
```

Tested with Python 3.13, JAX 0.11.2, Optax 0.2.8, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, NetworkX 3.6.1 and Matplotlib 3.11.2.

The code is CPU-only. The full pipeline takes roughly one to two days on a two-core machine. The NCI1 quantum model and the shot study account for most of that time.

## Data

The TU data sets (MUTAG, PTC-MR, NCI1, PROTEINS, IMDB-B) are the public versions in the GIN repository, together with its official ten-fold indices: https://github.com/weihua916/powerful-gnns (`dataset.zip`).

1. Unzip the archive.
2. Place the data as `./data/<NAME>/<NAME>.txt` and `./data/<NAME>/10fold_idx/`, or set `LCQGNN_DATA` to the folder that contains them.

`python make_synthetic.py` writes REG3-TRI to `./data/REG3TRI`. No new data were collected, and no personal data are involved.

## Reproducing the paper

```
bash reproduce.sh                                    # everything
python run_benchmark.py MUTAG lcqgnn --folds 1-10    # one model on one data set
python exp_shots.py MUTAG --folds 1-10 --reps 10     # shot study (needs the trained parameters)
python stats.py                                      # significance tests
```

- **Defaults.** LC-QGNN runs with `--L 2 --Q 10 --C 4`. Ablations use `--L 1`, `--C 1` and `--ablate {nozz,nofield,sharedmix,zonly}`, with `--out results_abl`.
- **Outputs.**
  - Per-fold records are appended to `results/<DATASET>.jsonl`.
  - Trained parameters go to `results/params_<DATASET>_<tag>_f<fold>.npy`.
  - Analysis outputs are written as JSON files in `results/`.
- **Seeds.** All seeds derive from the fold index (initialisation, mini-batch order, inner split). Bit-identical results across machines are not guaranteed, because floating-point reduction order differs between BLAS builds.

## Built-in exactness checks

- `exp_covering.py full ...`: whole-graph register (graphs up to 16 nodes) against the cone simulator.
- `exp_covering.py exact ...`: covering circuits (16 qubits) against the cone simulator.
- `exp_identities.py`: closed forms of Section 3.3 against full state-vector simulation.

Size-group padding uses inert, uncoupled, unmeasured qubits, so it leaves every root read-out unchanged.

## Licence

MIT (see `LICENSE`).

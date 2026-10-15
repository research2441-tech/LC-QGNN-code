#!/bin/bash
# End-to-end reproduction of every table and figure of the LC-QGNN manuscript.
# Expects the TU data in ./data (see README) or LCQGNN_DATA pointing to it.
set -e
cd "$(dirname "$0")"
python make_synthetic.py                     # REG3-TRI (seed 7)
python data_stats.py                         # Table 2
python exp_covering.py stats                 # Table 6
for d in MUTAG PTC NCI1 REG3TRI; do          # Tables 3-4, Fig. 3
  for m in lcqgnn conegin gin gcn sage gat mlp wl; do
    python run_benchmark.py $d $m
  done
done
for d in REG3TRI MUTAG; do                   # Table 5 (ablations)
  python run_benchmark.py $d lcqgnn --L 1 --tag abl_L1 --out results_abl
  python run_benchmark.py $d lcqgnn --C 1 --tag abl_C1 --out results_abl
  for a in nozz nofield sharedmix zonly; do
    python run_benchmark.py $d lcqgnn --ablate $a --tag abl_$a --out results_abl
  done
done
python stats.py                              # Wilcoxon/Holm, Friedman
python exp_covering.py full MUTAG PTC NCI1 REG3TRI    # Section 5.1 (full register)
python exp_covering.py exact MUTAG PTC NCI1 REG3TRI   # Section 5.1 (covering circuits)
python exp_identities.py                     # Section 3.3 closed forms
python exp_trainability.py                   # Fig. 5
python exp_shots.py MUTAG --folds 1-10 --reps 10                    # Table 7, Fig. 4
B=64,256,1024,4096,16384
python exp_shots.py PTC --folds 1-10 --reps 3 --budgets $B
python exp_shots.py REG3TRI --folds 1-10 --reps 2 --budgets $B
python exp_shots.py MUTAG --folds 1-10 --reps 3 --flip 0.02 --tag _flip --budgets $B
python fig_arch.py                           # Figs. 1-2
python fig_results.py                        # Figs. 3-5

"""Backfill reps 20-39 for Laguerre d=25 in the joint L-sweep experiment (see
conversation) -- Joint_LSweep_40Rep_Experiment.py ran that basis/dimension at
only 20 reps to start, matching everything else (RNN all dims, Laguerre
d=1/d=10) which is already at the full 40. Reuses that script's own
LAGUERRE_LAMBDA table, seeds, and run_cell function directly, just with
rep_start=20 so it appends reps 20-39 to the SAME CSV without touching
reps 0-19.

Run: python -m scripts.Joint_LSweep_Laguerre_D25_Backfill
"""
import time

from scripts.Joint_LSweep_40Rep_Experiment import L_VALUES, LAGUERRE_LAMBDA, N_REPS, run_cell

if __name__ == "__main__":
    print(f"Laguerre d=25 backfill: reps 20-{N_REPS-1}, L in {L_VALUES}\n")
    t_start = time.time()
    for L in L_VALUES:
        run_cell("laguerre", 25, L, LAGUERRE_LAMBDA[(25, L)], n_reps=N_REPS, rep_start=20)
    print(f"\ndone, total {time.time()-t_start:.1f}s")

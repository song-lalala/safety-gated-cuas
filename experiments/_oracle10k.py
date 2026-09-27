import sys, os
sys.path.insert(0, 'experiments')
from concurrent.futures import ProcessPoolExecutor
from run_review_sweeps import heuristic_cell, run_cell, cell_key, load_cell, W1_SIGMAS
from cuas_sim.statistics import make_rws_aggregator

BIG = list(range(1, 10001))
# W1 found the per-sigma optimum at (theta_abort, theta_track) = (0, 0) for every
# sigma, with theta_risk 0.1 at sigma=0 and 0.7 elsewhere. Only those six cells
# need the larger sample; the grid search that located them stays at N=1000.
SPECS = [heuristic_cell(BIG, s, 0.0, 0.0, 0.1 if s == 0.0 else 0.7) for s in W1_SIGMAS]

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=6) as pool:
        list(pool.map(run_cell, SPECS))
    rws = make_rws_aggregator()
    print("sigma,oracle_rws_N10000")
    for s, spec in zip(W1_SIGMAS, SPECS):
        print(f"{s},{rws(list(load_cell(cell_key(spec)).values())):.4f}")

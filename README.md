# Safety-gated engagement decisions for contact-based C-UAS — simulation code

Simulation framework and experiment driver for

> M. Song, "Statistically Calibrated Safety Gating for Noise-Robust
> Contact-Based Counter-UAS Engagement," *IEEE Access* (under review,
> Access-2026-37078).

The version used in the article is tagged `v1.0.1`.

## Setup

Python 3.10 or later (developed and tested on CPython 3.14.4). The simulator
itself has no third-party dependencies; `pytest` is the only test requirement.

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[test]"
pytest                          # 181 tests, a few seconds
```

`pip install -e .` is required: the driver imports `cuas_sim`, so running it in
a fresh clone without installing the package fails with
`ModuleNotFoundError: No module named 'cuas_sim'`. Setting `PYTHONPATH=src`
works as an alternative.

## Reproducing the article

Run from the repository root. The order matters: the first command writes every
N = 1000 table, including the velocity-noise check of Section VI-D, which has no
N = 10,000 cells; the second raises the headline tables to N = 10,000.

```
python experiments/run_review_sweeps.py all --seeds 1000
for t in w5 w13 w14 w15 w16 w17; do
    python experiments/run_review_sweeps.py $t --seeds 10000
done
```

On Windows PowerShell the second command is

```
foreach ($t in 'w5','w13','w14','w15','w16','w17') {
    python experiments/run_review_sweeps.py $t --seeds 10000
}
```

Results are written to `outputs/review/`, where the 25 result tables of the
article's supplementary material (Tables S1–S25) are committed, so a run can be
compared against them directly. The driver caches each configuration, so a
repeat run recomputes only what changed; from an empty cache the full set takes
several hours, most of it in the N = 10,000 passes.

One thing differs legitimately between runs: the three runtime rows of Table S22
(`kalman_us`, `quad_simpson_us`, `quad_trig_us`) are timing measurements and
vary by machine. Everything else is deterministic given the seed streams.

The tables were produced on Windows and carry CRLF line endings, as does the
supplementary material; Git stores them with LF, and a run on Linux or macOS
writes LF. Compare them ignoring line endings, for example with
`diff --strip-trailing-cr`.

## Layout

- `src/cuas_sim/` — the simulator: scenarios, observation model, estimation
  (α–β and Kalman), trajectory prediction, decision policies, evaluators, and
  the Monte Carlo runner.
- `experiments/run_review_sweeps.py` — the driver that produces every table and
  figure in the article. Work items `w1`–`w6` and `w8`–`w17` (there is no
  `w7`). `--analyze-only` skips the cell-building pass, but it is not a pure
  re-read of the cache: the analyses of `w1`, `w6`, `w12` and `w17` run
  simulations of their own.
- `experiments/` (other scripts) — earlier analysis tools from the conference
  version of this work. Not needed to reproduce the article, and kept because
  some tests import them. Two of them draw figures and import `numpy` and
  `matplotlib` inside those functions; neither package is needed for the tests
  or for the driver.
- `tests/` — unit tests.
- `outputs/review/*.csv` — the 25 result tables the article ships as
  supplementary material.


# How close to a saddle-node can a noisy system be observed?

Snapshot information, escape, and the observability horizon.

This repository contains the LaTeX source and the code used to derive and
verify every quantitative claim in the paper (submitted to Chaos).

## Contents

- `main.tex`, `appendix_proofs.tex`, `paper1_refs.bib` — the manuscript source.
- `figures/` — the compiled figures (also regenerable from the scripts below).
- Verification and analysis scripts:
  - `verify_all.py` — analytic checks and the core eigenproblem solver.
  - `monte_carlo.py` — stochastic validation.
  - `adjoint.py` — the left eigenfunction and Hellmann–Feynman check.
  - `path_information.py` — the continuous-record information-rate bound.
  - `census_channel.py` — the binary-census design rule and its efficiency.
  - `kibble_zurek.py` — Kibble–Zurek freeze-out vs. the observability horizon.
  - `hazard_generality.py` — hazard-shape independence of the census rule.
  - `fold_location_information.py` — Fisher information for the fold location.
  - `containment_simulation.py` — direct simulation fixing the containment
    threshold.
  - `fold_information_scaling.py` — asymptotic scaling of the fold-location
    information.
  - `stommel_horizon.py` — the two-box thermohaline application.
  - `observability_horizon.py` — standalone reference implementation of the
    observability horizon, with a `--selftest` mode; see below.
  - `generate_pre_figures.py`, `generate_stommel_figure.py` — figure
    generation.

## Requirements

Python 3.9+, with `numpy`, `scipy`, and `matplotlib` (for the figure
generators). No other dependencies.

## Using the reference implementation

```
python3 observability_horizon.py --D 0.09 --eps 1e-3
python3 observability_horizon.py --D 0.09 --eps 1e-3 --lead-time 300
python3 observability_horizon.py --selftest
```

## Building the manuscript

Compile with `tectonic main.tex` or any standard LaTeX toolchain (REVTeX 4.2,
`pre` class).

## Citation

If you use this code, please cite the paper (citation details to be added on
acceptance).

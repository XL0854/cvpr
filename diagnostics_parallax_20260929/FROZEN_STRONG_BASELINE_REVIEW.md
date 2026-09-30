# Frozen unified strong-baseline review

## Research positioning

The observed problem is residual local misalignment in the RGB output initialized by frozen RopStitch at alpha 0.5. Candidate correspondences and residual-grid optimization are post-processing diagnostics. Their filtering or solver failures are not attributed to RopStitch internals.

No predictor, learned module, new data, grid change or alpha experiment is permitted in this review.

## Data and development status

Reuse the existing 15 diagnostic pairs: five each from courtyard, delivery_area and electro. These pairs have already been used for diagnosis and are development evidence, not an untouched final test set.

## Groups

- A: cached frozen RopStitch alpha=0.5 initial result.
- C-Adam: cached conventional rule (forward confidence >=0.5, reverse confidence >=0.5, cycle <=2 px), weight `clip(min confidence, 0.05, 1)`, Adam lr 0.2 for 150 steps.
- C-LBFGS: exactly the same C candidates and weights, initialization, objective, 13x13 target residual grid, fixed reference and feasibility rule; L-BFGS with lr 0.8, up to 15 accepted outer calls of 20 internal iterations with strong-Wolfe line search.
- D-LBFGS: the same fixed candidates restricted to truth-evaluable `e<=3` native pixels, unit weights; otherwise identical to C-LBFGS. It is diagnostic only.

## Shared objective and validity

SmoothL1 beta 2 px at 512 scale, anchor 0.002, first-difference smoothness 0.02, triangle barrier 10, sampled TPS-Jacobian barrier 100 and per-component residual cap 128 px. The reference mesh is fixed. A candidate state is feasible only when all mesh triangles and sampled TPS determinants preserve the baseline orientation. Invalid final states return the last accepted feasible state.

Held cells (`source_cell_id % 4 == 0`) are scoring-only and excluded from C and D fitting.

## Accounting

Record optimizer wall time, every L-BFGS closure call (one objective plus gradient evaluation), outer calls, accepted feasible states, maximum component displacement and termination reason. Strong-Wolfe line-search calls count in both wall time and evaluation count.

For equal-time comparison, retain the latest accepted feasible L-BFGS checkpoint whose cumulative optimizer time does not exceed that pair's cached C-Adam fit time. If none exists, use the initial state. Full L-BFGS results are also reported.

Total diagnostic latency is component-wise: cached RopStitch CPU export + cached RoMa GPU matching + optimizer CPU time. It is a mixed-device diagnostic record, not a deployable latency benchmark.

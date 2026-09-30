# Diagnostic correction log

The first exploratory execution of `diagnose_residual_bottleneck.py` returned the raw final iterate for every configuration. That did not reproduce the established A/B/C/D behavior, which returns the last structurally feasible iterate when the raw final grid is invalid.

Before interpreting results, the script was corrected as follows:

- baseline, longer-solver, cap and fine-grid comparisons now use the same last-feasible return rule as the established experiment;
- intentionally relaxed regularization scans retain their raw endpoint and explicitly mark structural invalidity, so folded states can be visualized but cannot count as improvements;
- all three pairs and every configuration were rerun uniformly after the correction.

The preliminary inconsistent CSV was overwritten. The final Adam-150 held errors reproduce the established D results: 84.2141, 16.8099 and 42.9171 px.

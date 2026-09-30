# Frozen residual-error diagnostic

This selection and protocol were fixed before running any new solver, constraint or grid-resolution scan.

## Representative pairs

1. `courtyard/DSC_0317_DSC_0318`: truth-selected D starts with training geometry loss 68.38 and still has 46.47 after 150 Adam steps; held error remains 84.21 px. Selected as the training-correspondence fitting failure.
2. `electro/DSC_9302_DSC_9303`: D training geometry falls from 15.26 to 1.84, but held error is 16.81 px and initially aligned held points rise to 22.33 px. D uses 76 training points in only 12 source cells. Selected as the train/held generalization and distortion failure.
3. `courtyard/DSC_0315_DSC_0317`: D lowers held error from A 72.24 to 42.92 px while moving a control point up to 53.0 px and retaining only 89.4% of A's overlap. Selected as the error-versus-distortion tradeoff case.

Conclusions are limited to these three diagnostic samples.

## Fixed data

- The candidate set, ETH3D `e<=3` correct-match rule, unit weights, initial RopStitch alpha=0.5 meshes, fixed reference side and scoring-only cell rule (`cell_id % 4 == 0`) remain unchanged.
- Scoring points never enter optimization. No parameters are selected from held error.
- Baseline model: 13x13 target residual grid, Adam lr 0.2, 150 steps, SmoothL1 beta 2 px, anchor 0.002, smoothness 0.02, triangle barrier 10, sampled TPS-Jacobian barrier 100, displacement cap 128 px.

## Diagnostic sequence

1. Verify forward-map/inverse-sampling scale and gradients with identity, translation, finite-difference and inversion checks. Record raw Euclidean and SmoothL1 training errors, held errors, individual losses and gradient norms.
2. Same 13x13 model: compare baseline Adam-150, Adam-600 and LBFGS only as convergence checks.
3. Constraint scans use the same Adam-300 budget. Structural multipliers are 0, 0.1, 1 and 10 at cap 128. Displacement caps are 16, 32, 64 and 128 px at structural multiplier 1. Invalid/folded results remain reported as invalid and cannot count as improvement.
4. Only after steps 1--3, compare one 25x25 grid against 13x13 with the same loss coefficients, cap and Adam-600 budget. The initial 25x25 mesh is a bilinear interpolation of the same initial 13x13 target mesh; reference geometry and all points remain fixed.

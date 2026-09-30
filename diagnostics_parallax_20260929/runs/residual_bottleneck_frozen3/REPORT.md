# Given-correct-correspondence residual-error diagnosis

## Boundary and samples

This is a three-pair diagnosis, not a dataset-level claim. The samples and reasons were frozen in `FROZEN_REMAINDER_DIAGNOSTIC.md` before the new scans:

- courtyard DSC_0317--0318: baseline truth-selected training loss remains high.
- electro DSC_9302--9303: training improves, while independent and initially aligned points regress.
- courtyard DSC_0315--0317: error reduction accompanies large displacement and overlap loss.

Every comparison uses the same fixed `e<=3` truth-selected candidate set, unit weights, alpha=0.5 initialization, fixed reference mesh and scoring-only cells. The held cells never enter fitting or configuration choice.

## Implementation checks

The TPS class maps normalized common-canvas coordinates to normalized source-image coordinates. Point alignment first inverts the fixed reference TPS at source pixels and evaluates the target TPS at those canvas coordinates. Rendering evaluates the same inverse canvas-to-image TPS and passes the result to `grid_sample`.

Numerical checks passed:

| Check | Result |
|---|---:|
| Identity forward-map maximum error | 5.55e-16 |
| TPS forward/inverse round-trip maximum error | 3.21e-30 |
| Known 17/-9 px translation inverse error | 6.66e-16 |
| Identity inverse-sampling ramp maximum error | 8.54e-6 px |
| Autograd vs finite-difference directional-gradient relative error | 2.22e-8 |
| Finite gradients and valid inversions | yes |

The coordinate convention is therefore internally consistent for these tests: correspondence coordinates use the 512 network scale, TPS normalized coordinates use `pixel/256-1`, and rendering converts with `pixel=(normalized+1)*256`, then `grid=2*pixel/511-1` under `align_corners=True`.

## Original errors and optimization traces

`Train` below is the symmetric common-canvas distance on the fixed correct training correspondences. `Held` is the independent ETH3D truth-point canvas distance. It is different from the SmoothL1 optimization loss.

| Pair | Train n | Held n | Initial train | Adam-150 train | Initial held | Adam-150 held |
|---|---:|---:|---:|---:|---:|---:|
| 0317--0318 | 341 | 127 | 126.69 | 108.75 | 104.18 | 84.21 |
| 9302--9303 | 76 | 28 | 24.66 | 6.27 | 17.31 | 16.81 |
| 0315--0317 | 381 | 157 | 42.65 | 9.80 | 72.24 | 42.92 |

For Adam-150, the SmoothL1 geometry term and gradient norm change as follows:

| Pair | Geometry term | Delta gradient norm | Returned state |
|---|---:|---:|---|
| 0317--0318 | 68.38 -> 46.47 | 0.137 -> 0.082 | raw final invalid; last feasible returned |
| 9302--9303 | 15.26 -> 1.84 | 0.136 -> 0.019 | raw final feasible |
| 0315--0317 | 22.59 -> 4.65 | 0.118 -> 0.090 | raw final feasible |

The complete histories record geometry, raw Euclidean error, anchor, smoothness, triangle barrier, TPS-Jacobian barrier and gradient norm every 25 steps. They are stored as `<pair>/<configuration>_history.json`.

## Solver diagnosis with the same 13x13 model

| Pair | Adam-150 train/held | Adam-600 train/held | L-BFGS train/held | L-BFGS fold | L-BFGS anisotropy p95 | L-BFGS overlap retention |
|---|---:|---:|---:|---:|---:|---:|
| 0317--0318 | 108.75 / 84.21 | 108.75 / 84.21 | 16.26 / 18.83 | 0 | 2.33 | 1.110 |
| 9302--9303 | 6.27 / 16.81 | 0.64 / 6.68 | 0.82 / 6.05 | 0 | 2.47 | 0.862 |
| 0315--0317 | 9.80 / 42.92 | 1.60 / 25.09 | 1.60 / 25.01 | 0 | 2.21 | 0.843 |

Longer Adam works on two pairs. On 0317--0318, its raw trajectory lowers the geometry term to 11.46 but crosses the structural-validity boundary, so the required last-feasible return remains identical to Adam-150. L-BFGS reaches a later feasible state and lowers held error on all three pairs without sampled folds. This is direct evidence that solver path/convergence is a bottleneck in the baseline diagnosis.

L-BFGS is not a complete fix. On 9302--9303, initially aligned held points improve from Adam-150's 22.33 to 4.53 px but remain worse than the original A value of 1.76 px. On 0315--0317, overlap retention falls from Adam-150's 0.894 to 0.843 while held error improves. The lower error therefore retains a content/geometry cost.

## Constraint and amplitude scans

All scan points use Adam-300. Red points in `constraint_tradeoff.png` are structurally invalid and are not counted as improvements.

- Removing or reducing structural regularization produces folds in all three samples: sampled fold fractions reach 33.7%, 21.1% and 11.7% for regularization scale zero. Low optimization loss from those states is invalid evidence.
- Increasing regularization to 10 gives valid, less distorted states. Held error becomes 82.79, 11.15 and 41.99 px for the three pairs, with anisotropy p95 1.28, 1.37 and 1.78. It protects geometry but leaves much of the alignment residual.
- Component caps of 16 and 32 px protect coverage/shape but leave higher alignment error. A 64 px cap is sufficient for the two easier solver cases; the hard 0317--0318 case still crosses the feasibility boundary.

This establishes a real alignment-versus-structure conflict. Simply relaxing constraints does not yield valid improvement.

## Grid capacity and spatial support

The single 25x25 comparison uses the same initialization shape, correspondences, weights, coefficients, cap and Adam-600 budget. All three raw fine-grid endpoints become structurally invalid and are returned to early feasible states:

| Pair | 13x13 Adam-600 held | 25x25 Adam-600 held | Fine-grid returned displacement | Conclusion |
|---|---:|---:|---:|---|
| 0317--0318 | 84.21 | 99.88 | 8.9 px | worse; early feasibility failure |
| 9302--9303 | 6.68 | 17.61 | 9.8 px | worse; early feasibility failure |
| 0315--0317 | 25.09 | 62.62 | 13.9 px | worse; early feasibility failure |

The finer grid is ineffective under this solver and constraint setup. Because optimization fails before exploiting its extra degrees of freedom, this experiment does not prove that 13x13 capacity is sufficient or insufficient. Increasing grid density again would be unjustified.

All held points lie in source cells deliberately excluded from training, so the cell-present subset is empty by construction. Distance-to-training-support still shows a limited extrapolation signal after L-BFGS: on 0317--0318, held points within 64 px average 16.19 px while the three farther points average 127.96 px; on 0315--0317 the corresponding values are 22.69 px and 41.83 px (19 far points). Electro has only one farther held point. These counts are too small for a general claim, but sparse/distant support remains a concrete failure condition to retain.

## Conclusions

The strongest evidenced bottleneck in these samples is insufficient/constrained optimization of the existing residual grid. L-BFGS with the same model, data and loss coefficients is the simplest effective control, reducing independent error on all three samples. A second evidenced bottleneck is the alignment-versus-structure/coverage tradeoff; relaxing structure creates invalid folds, while valid lower-error solutions can reduce overlap or degrade initially aligned points.

Warp expressiveness cannot yet be excluded. The fine grid did not help because its optimization hit feasibility limits, so it is not a clean capacity test. Spatial-support extrapolation is also unresolved but supported by the small far-from-training subsets.

The stable condition worth a future literature check is optimization near the structural-validity boundary under sparse or uneven correspondence support. This report does not propose a new method or claim novelty, and no subsequent research direction is started automatically.

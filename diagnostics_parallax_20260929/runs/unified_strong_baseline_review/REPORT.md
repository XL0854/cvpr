# Unified strong-baseline review

## Corrected research position

This review studies RGB-only post-processing of residual local misalignment after frozen RopStitch output at alpha 0.5. It does not attribute post-hoc correspondence filtering failures or residual-optimizer failures to RopStitch internals. C-LBFGS is a conventional strong baseline; D-LBFGS uses ETH3D truth only for diagnosis. Neither is presented as a new method.

All 15 pairs were previously used for diagnosis. These results are development evidence, not an independent final test.

## Fairness and reuse

A and C-Adam meshes, candidate sets and timings are reused. Only the missing C-LBFGS/D-LBFGS fits were run. Every optimized group uses the same 13x13 target residual grid, fixed reference, alpha=0.5 initialization, SmoothL1 beta 2 px, anchor 0.002, smoothness 0.02, triangle barrier 10, TPS-Jacobian barrier 100, per-component 128 px cap and last-feasible return rule.

C uses forward/reverse confidence >=0.5, cycle <=2 px and clipped minimum-confidence weights. D selects `e<=3` native-pixel matches from the same candidates with unit weights. Held source cells never enter fitting.

L-BFGS uses lr 0.8, strong-Wolfe line search, at most 15 outer calls of 20 internal iterations. Every closure call is counted as one objective and gradient evaluation. Fourteen of 15 C runs and 14 of 15 D runs stop after two stagnant outer calls; courtyard 0317--0318 reaches the 15-outer limit. C uses 54--351 closure evaluations and D uses 65--362. Line-search evaluations are included in wall time.

## Geometry results

Values are point-count-weighted held common-canvas errors at the 512 input scale.

| Scene | Held n | A | C-Adam | C-LBFGS | D-LBFGS | Time-matched C-LBFGS | Time-matched D-LBFGS |
|---|---:|---:|---:|---:|---:|---:|---:|
| courtyard | 615 | 65.95 | 53.55 | 39.33 | 35.89 | 39.63 | 37.17 |
| delivery_area | 208 | 18.03 | 13.79 | 11.64 | 12.91 | 11.64 | 12.91 |
| electro | 531 | 15.30 | 11.87 | 11.31 | 11.11 | 11.31 | 11.11 |
| all development pairs | 1,354 | 38.73 | 31.10 | 24.09 | 22.64 | 24.22 | 23.22 |

C-LBFGS improves A on 13/15 pairs, equals A on courtyard 0309--0311 and mildly worsens electro 9257--9258. D improves A on 11/15, equals A on two and worsens it on two. At full solve, C obtains 91.0% of the pooled A-to-D reduction; under the cached C-Adam wall-time budget it obtains 93.6%. D has no stable scene-wide advantage: it is better on courtyard by 3.44 px, worse on delivery_area by 1.27 px and better on electro by 0.20 px.

The equal-time state is the latest accepted feasible L-BFGS outer state within each pair's measured C-Adam fit time. Strong-Wolfe calls count against this budget. On pairs where the full solve already terminates inside the budget, full and time-matched results coincide.

## Initially aligned regions, distortion and coverage

| Scene | A aligned | C-Adam aligned | C-LBFGS aligned | D-LBFGS aligned | C/D minimum overlap retention | C/D maximum anisotropy p95 |
|---|---:|---:|---:|---:|---:|---:|
| courtyard | 1.49 | 0.56 | 0.56 | 0.63 | 0.866 / 0.843 | 2.03 / 2.33 |
| delivery_area | 1.57 | 1.10 | 1.11 | 1.18 | 0.953 / 0.980 | 3.68 / 2.27 |
| electro | 1.78 | 1.94 | 2.10 | 2.45 | 0.847 / 0.862 | 1.82 / 2.47 |

All reported returned meshes have zero sampled folds and zero held inversion failures. Stronger solving does not eliminate the tradeoff: electro's initially aligned regions regress, and the worst C/D overlap retention is 84.7%/84.3%. Lower geometric error cannot be treated as an unconditional improvement on those pairs.

## Pair-level failures and spatial support

The complete distribution is in `all_pair_geometry.png` and `all_groups.csv`.

1. **Both L-BFGS groups fail through structural feasibility: courtyard 0309--0311.** C and D accept zero feasible outer states and return A at 120.61 px. Their raw geometry terms fall to 6.03 and 6.68, but every accepted outer candidate violates the structural-validity test. C-Adam had reached a feasible 105.63 px result. This is solver-path/constraint interaction, not evidence that a learned correspondence model is needed.

2. **C trails D in a spatially localized support gap: courtyard 0315--0317.** C/D are 35.66/25.01 px at essentially the same wall-time budget. Most of D's gain is in the lower-left source region: held cells centered near `(37,362)`, `(29,409)` and `(40,468)` improve by 11.1, 28.8 and 37.4 px. In the latter two cells, nearest selected-support distance changes from C 78.8/124.3 px to D 39.8/55.6 px. The per-point correlation between D's error gain and D's support-distance advantage is 0.974. This is evidence that the conventional rule misses useful spatial support on this pair.

3. **A second C<D pair is not explained by support distance: delivery_area 0695--0696.** C/D are 18.82/13.85 px, and D improves several right-side cells by 9--17 px, but D support is often farther away. Because D also changes selection size and weights and much of C lacks evaluable truth, this pair cannot isolate correspondence judgment from objective weighting or warp behavior.

4. **Truth selection is not reliably safer.** On delivery_area 0717--0718, C improves A from 40.52 to 30.24 px, while D accepts no feasible outer update and returns A. On delivery_area 0685--0686 and several electro pairs, D is also worse than C. The diagnostic truth selector therefore does not define a stable deployable target.

The source-space visualization `failure_region_maps.jpg` uses green for held points where D beats C and red where C beats D. Per-cell values and support distances are in `selected_region_cells.csv`.

## Runtime

Average optimizer-only CPU time and objective/gradient evaluations:

| Scene | C-Adam | C-LBFGS | D-LBFGS |
|---|---:|---:|---:|
| courtyard | 1.35 s / 150 | 1.52 s / 165 | 1.53 s / 173 |
| delivery_area | 1.61 s / 150 | 0.98 s / 115 | 1.11 s / 136 |
| electro | 1.54 s / 150 | 1.12 s / 111 | 1.20 s / 123 |

Including cached RopStitch CPU export, cached RoMa GPU matching, CPU optimization and measured CPU rendering, the mixed-device observed totals for C-Adam/C-LBFGS/D-LBFGS are 5.09/5.25/5.27 s on courtyard, 4.28/3.66/3.76 s on delivery_area and 4.32/3.89/3.99 s on electro. These numbers include actual line-search cost but are diagnostic component sums, not a same-device deployment benchmark.

## Judgment and recommendation

C-LBFGS is retained as the conventional strong baseline. It explains most of the attainable diagnostic gain, including at matched wall time, and D does not provide a stable comprehensive advantage. No reliability predictor or other learned selector is supported by this evidence.

Two mechanisms remain visible: structural-feasibility blocking where both C and D fail, and localized missing spatial support where D helps C on one clear pair. The second condition is not stable enough across the 15 development pairs to establish a method gap. The 13x13 expression limit remains unexcluded because this review intentionally does not change the grid, and the earlier fine-grid diagnostic was confounded by feasibility failure.

**在正确对应空间支撑被常规可靠性规则遗漏的局部大错位条件下，常规强基线仍出现独立评价误差偏高；证据是 courtyard 0315--0317 左下三个评价区域中 D-LBFGS 比 C-LBFGS 低 11--37 px，且 D 的最近支撑距离明显更小；当前尚不能排除结构约束与 13x13 形变表达能力限制。**

This condition is suitable for a focused literature check, but the current evidence does not yet support claiming novelty. Stop here; do not train, add data or start another experiment automatically.

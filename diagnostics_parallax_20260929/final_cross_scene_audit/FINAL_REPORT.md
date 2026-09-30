# Frozen cross-scene correspondence diagnosis

## Scope and audit

Courtyard is retained as the development diagnosis. The two added scenes, delivery_area (indoor) and electro (outdoor), were named and their adjacent-frame/coverage-based sampling rule was frozen before RoMa matching or A/B/C/D optimization. Each contributes five pairs. The new scenes received the courtyard thresholds, weights, constraints, learning rate and 150-step budget without tuning or stopping-point selection.

All methods start from the same frozen RopStitch alpha=0.5 meshes. A does not optimize. B uses initial canvas residual <=6 px and unit weights. C uses forward confidence >=0.5, reverse confidence >=0.5 and cycle error <=2 px, weighted by clipped minimum bidirectional confidence. D uses only ETH3D-evaluable, e<=3 native-pixel candidates from the same fixed candidate set with unit weights. D receives no extra truth points and is only a diagnostic selector.

The reference mesh is fixed. B/C/D update the same 13x13 target residual grid using Adam (lr 0.2), SmoothL1 beta 2 px, anchor 0.002, smoothness 0.02, triangle barrier 10, TPS-Jacobian barrier 100, displacement clamp [-128,128] px and exactly 150 steps. No group uses validation stopping.

Cells with `source_cell_id % 4 == 0` are scoring-only and are excluded from every optimization group. Courtyard scores are development evidence and are not claimed as untouched test results. The new-scene scoring points did not enter optimization, pair selection, parameter selection or stopping decisions.

The 65.95/53.55/52.07 px courtyard figures are point-count-weighted means over 615 held ETH3D truth points for A/C/D. For each point, the source pixel and ETH3D target truth pixel are inverted through their respective meshes into the shared RopStitch canvas; their Euclidean separation is measured in canvas pixels at the 512x512 network scale. All final held evaluations had zero inversion failures.

## Filtering audit

The false-acceptance denominator is all truth-evaluable fixed candidates classified incorrect (`e_native > 6 px`). Candidates in the 3--6 px uncertainty interval and candidates without valid truth are excluded. The post-filter precision denominator is kept classified points: correct plus incorrect.

| Scene | Correct large residual | Residual filter retention at nominal 5% FAR | Actual incorrect acceptance | Frozen C large retention | Frozen C precision | Frozen C kept classified |
|---|---:|---:|---:|---:|---:|---:|
| courtyard | 1,450 | 0/1,450 (0%) | 9/191 (4.71%) | 937/1,450 (64.62%) | 99.04% | 1,764 |
| delivery_area | 281 | 0/281 (0%) | 5/103 (4.85%) | 122/281 (43.42%) | 96.01% | 326 |
| electro | 555 | 0/555 (0%) | 7/141 (4.96%) | 390/555 (70.27%) | 98.60% | 783 |

Thus correct large-initial-residual correspondences occur in all three scenes, and residual-based selection rejects all of them at the matched operating point. Frozen C retains a substantial but scene-dependent fraction. Per-pair denominators and counts are in `../runs/roma_er_*/filter_audit_per_pair.csv`.

## Geometry, structure and coverage

All geometry numbers below are point-count-weighted held canvas error in 512-scale pixels. `Aligned` uses points whose A error is <=3 px. Coverage is the minimum per-pair overlap relative to A. Area change is the maximum absolute per-pair total triangle-area ratio change. Every group had zero sampled TPS folds and zero held inversion failures.

| Scene | Held n | A | B | C | D | Controlled C | Controlled D | C aligned | D aligned | C/D min coverage | C/D max area change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| courtyard | 615 | 65.95 | 65.65 | 53.55 | 52.07 | 54.48 | 52.70 | 0.56 | 0.63 | 0.902 / 0.894 | 0.085 / 0.059 |
| delivery_area | 208 | 18.03 | 17.14 | 13.79 | 12.68 | 13.74 | 13.49 | 1.10 | 1.18 | 0.963 / 0.980 | 0.051 / 0.052 |
| electro | 531 | 15.30 | 14.94 | 11.87 | 12.34 | 11.51 | 12.47 | 1.93 | 5.43 | 0.879 / 0.889 | 0.120 / 0.110 |

The control matches C and D point counts independently within every source 8x8 cell and uses unit weights in both. D's raw advantage over C is 2.77% on courtyard and 8.01% on delivery_area, while D is 3.91% worse on electro. After the control, D is 3.26% better on courtyard, 1.83% better on delivery_area and 8.42% worse on electro. The delivery_area raw gap therefore largely reflects point count, spatial distribution and weighting rather than a stable selection advantage.

## Pair-level distribution and examples

D beats C on 3/5 courtyard pairs, 3/5 delivery_area pairs and 2/5 electro pairs. The complete primary and controlled pair differences are plotted in `all_pair_D_minus_C.png` and tabulated in `all_pairs.csv`.

- Improvement: courtyard DSC_0317--0318 changes from C 92.28 px to D 84.21 px; the spatial/count control remains favorable (90.72 vs 84.60 px). This is a condition where the selected correspondence set matters, but both errors remain large and D's final training geometry loss is 46.47, so correct selection alone does not make the residual grid fit the geometry.
- Essentially unchanged: courtyard DSC_0286--0287 is 1.160 px for C and 1.160 px for D. It already has low error, leaving no meaningful selection gain.
- Degradation: electro DSC_9302--9303 changes from C 6.44 px to D 16.81 px; controlled C/D is 7.33/16.81 px. D has only 76 training points in 12 source cells, moves the grid by up to 53.2 px, and increases aligned-region error from C 1.70 px to D 22.33 px. There are no detected folds, but the visible warp distortion and held-point regression show sparse-support overfitting/poor extrapolation.

The montage is `improved_unchanged_degraded_examples.jpg`; the three-scene truth-error/initial-residual plot is `e_r_scatter_three_scenes.png`.

## Runtime and memory

RoMa GPU matching, including first-pair warm-up in each run, averaged 1.82/0.95/0.85 s per pair on courtyard/delivery_area/electro and peaked at 4.60 GB allocated. C's 150-step residual-grid fit averaged 1.35/1.61/1.54 s per pair on CPU. The frozen RopStitch export averaged 1.86/1.65/1.85 s on CPU. The corresponding mixed-device observed sums are 5.03/4.21/4.25 s per pair. These are component diagnostics, not a deployment benchmark: RopStitch was forced to CPU because this tool session could not initialize CUDA, while RoMa was manually run on GPU. RopStitch GPU peak memory and a same-device end-to-end latency are therefore unavailable and are not fabricated.

## Decision

Correct large-residual false rejection repeats in all three scenes, but it does not translate into a stable advantage for truth-selected D over conventional C. C captures the stable practical gain in these 15 pairs; D provides no stable comprehensive improvement and can seriously damage already aligned regions when its correct support is sparse. The remaining evidence points mainly to the residual-grid optimization, support distribution and structural/generalization limits. On some hard courtyard pairs, even D's training geometry loss remains large, which also leaves warp expressiveness/objective conflict unresolved. Correspondence judgment is not the dominant remaining bottleneck under this protocol.

Following the preregistered stopping rule, the standalone correspondence-screening direction is stopped and archived. No screening network should be trained from this result. The isolated condition worth retaining is sparse or uneven correct-support coverage; any future work should first test optimization/warp behavior under that condition rather than introduce a learned selector.

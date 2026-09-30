# Frozen A/B/C/D implementation audit

## Shared optimization and evaluation

- A is the frozen RopStitch result at alpha 0.5. B/C/D start from exactly A's reference and target 13x13 meshes.
- The reference mesh is fixed. B/C/D optimize only the target mesh residual with Adam, lr 0.2, exactly 150 steps; there is no early stopping or result-dependent checkpoint choice.
- Every optimized group uses SmoothL1 point loss (beta 2 px at the 512x512 network scale), anchor 0.002, first-difference smoothness 0.02, triangle barrier 10, sampled TPS-Jacobian barrier 100, and displacement clamp [-128,128] px. An invalid final mesh reverts to the last feasible iterate.
- B selects valid training candidates with current initial canvas residual <=6 px and unit weights.
- C selects valid training candidates with forward confidence >=0.5, reverse confidence >=0.5, and forward-backward cycle <=2 px. Its point weight is clip(min(forward confidence, reverse confidence), 0.05, 1).
- D selects from the same fixed candidates only truth-evaluable matches with native-image truth error <=3 px, with unit weights. D adds no truth point and is a diagnostic selection, not a theoretical upper bound.
- Source 8x8 cells whose `cell_id % 4 == 0` are scoring-only. They are excluded from all optimization selections. Parameters were fixed on courtyard and are unchanged on the two new scenes.

## Error scale

For a held point, both its source pixel and ETH3D target truth pixel are inverted through their respective meshes into the shared initial RopStitch output canvas. The score is their Euclidean separation in canvas pixels at the 512x512 network input scale. An inversion failure is assigned 512 px and separately counted. The reported 65.95/53.55/52.07 px values are point-count-weighted arithmetic means over 615 held courtyard truth points for A/C/D; the final courtyard run had zero held inversion failures.

## Filtering denominator

The incorrect-acceptance denominator is every truth-evaluable fixed candidate with native-image truth error >6 px. The 3--6 px uncertainty band and unavailable truth are excluded. At the pooled nominal 5% operating point, integer thresholding accepts 9/191 incorrect courtyard points (4.71%). Frozen C is a separate fixed rule: it accepts 17/191 incorrect points (8.90%), retains 937/1450 correct large-residual points (64.62%), and has 1747/(1747+17)=99.04% classified precision.

## Engineering correction and control

The original diagnostic scripts embedded courtyard output names. They were changed to accept scene-independent geometry/diagnosis roots; thresholds and optimization settings were not changed. Courtyard was rerun uniformly. A supplementary control matches C and D point counts independently in every 8x8 source cell and uses unit weights in both groups. This isolates gross point-count/spatial-coverage and confidence-weight differences without changing the primary A/B/C/D definitions.

## Frozen cross-scene choice

Before matching or optimization, delivery_area and electro were fixed as indoor/outdoor scenes. Only adjacent lexicographic frames with at least 200 stride-32 truth-evaluable samples qualified. Five positions nearest equally spaced index quantiles including endpoints were selected; ties go to the earlier index. See `configs/delivery_area_final5.json` and `configs/electro_final5.json`.

# Final diagnostic protocol freeze

Frozen before downloading `delivery_area` or `electro`: 2026-09-30 Asia/Shanghai.

## Scope and scene selection

- Existing development scene: `courtyard`.
- Independent replication scenes fixed in advance: `delivery_area` (indoor) and `electro` (outdoor).
- Selection reason: both have official distorted DSLR JPG, matching distorted depth and calibration; together they add indoor/outdoor variation. They were selected before inspecting any RopStitch, RoMa or A/B/C/D result on these scenes.
- Official archive sizes verified by HTTP headers:
  - `delivery_area_dslr_jpg.7z`: 372,958,192 bytes.
  - `delivery_area_dslr_depth.7z`: 388,144,128 bytes.
  - `electro_dslr_jpg.7z`: 422,635,933 bytes.
  - `electro_dslr_depth.7z`: 655,596,830 bytes.

## Pair selection frozen before scene results

For each new scene:

1. Sort DSLR image basenames lexicographically.
2. Consider adjacent frames only (`i -> i+1`).
3. At a 32-native-pixel source grid, retain pairs with at least 200 evaluable common-visible ETH3D points.
4. If more than five pairs qualify, choose five indices nearest the evenly spaced quantiles from the first to last qualifying pair. Ties choose the earlier pair.
5. This rule uses only public depth/calibration overlap. It cannot read matcher correctness, RopStitch residual, A/B/C/D error, image-quality scores or qualitative fusion results.

## Frozen candidate and truth definitions

- RoMa outdoor commit `77f8d68803526dcddfd9b7a46bc76125bdc25f15`.
- Symmetric inference, coarse resolution 560, upsample resolution 864, custom correlation disabled.
- Candidate locations: complete 8-pixel regular source grid on the 512×512 input, 4096 candidates per pair; no confidence, cycle, geometric or RopStitch-residual prefilter.
- Correct: ETH3D target correspondence error `e <= 3` native target pixels.
- Uncertain: `3 < e <= 6` native target pixels; excluded from correct/incorrect filter precision.
- Incorrect: `e > 6` native target pixels.
- Correct large residual: correct and initial common-canvas residual `r >= 10` pixels in the RopStitch 512-input coordinate system.

## Frozen A/B/C/D definitions

- A: fixed-alpha RopStitch initialization; no residual-grid optimization.
- B: valid candidates with current canvas residual `r <= 6` pixels; unit weights.
- C: valid candidates with forward confidence >=0.5, reverse confidence >=0.5 and cycle error <=2 pixels; point weight is `clip(min(forward confidence, reverse confidence), 0.05, 1)`.
- D: diagnostic truth selection from the same candidate set: evaluable and `e <= 3` native pixels; unit weights. D adds no truth correspondence and is not called an upper bound.

For B/C/D, the reference mesh is fixed and only the target 13×13 residual mesh is optimized. All start from the identical RopStitch target mesh and use Adam (`lr=0.2`) for exactly 150 steps, Smooth-L1 alignment (`beta=2` in 512-scale target pixels), anchor weight 0.002, first-order grid smoothness weight 0.02, triangle barrier weight 10, TPS-Jacobian barrier weight 100, and per-axis displacement clamp [-128,128] pixels. A final invalid iterate is replaced by the last feasible iterate. No validation metric selects a stopping point.

## Optimization/evaluation separation

- Source locations are assigned to fixed 8×8 spatial cells.
- Cells with `cell_id mod 4 == 0` are scoring-only; none of their candidates enter B/C/D optimization, filtering parameter choice or stopping.
- All other cells are optimization-only.
- Parameters above were fixed on courtyard. New scenes cannot change them.

## Primary geometry metric

Each public-geometry correspondence is forward-mapped through both inverse TPS models to the fixed initial common canvas. Error is their Euclidean separation in canvas pixels. Canvas units correspond to the network's 512×512 input scale; canvas size may exceed 512 because it is the fixed bounding box of both initial meshes. An inversion failure is assigned 512 pixels; courtyard final scoring had zero inversion failures.

`65.95`, `53.55`, and `52.07` pixels are point-count-weighted arithmetic means over 615 scoring-only truth points pooled across the five courtyard pairs for A, C and D respectively. Per-pair means are also required so no claim relies only on this pooled statistic.

## Supplementary controlled comparison

Because C and D retain different counts, spatial distributions and weights, an audit control will match their per-cell point counts using deterministic evenly spaced sampling within each cell and unit weights for both. This is supplementary and does not replace the frozen A/B/C/D comparison.

# Current research position

The candidate problem is residual RGB alignment after frozen RopStitch output: starting from alpha=0.5 RopStitch results, use geometric evidence from the two RGB images to reduce residual local misalignment while controlling new distortion, degradation of initially aligned regions and valid-content loss.

This is an observed output-level problem and has not yet become a defensible paper contribution.

- Large-residual rejection of correct matches belongs to an added screening strategy; it is not attributed to RopStitch internals.
- Adam-150 insufficiency belongs to the added residual-grid optimizer; it is not a demonstrated RopStitch defect.
- Standalone correspondence-screening research remains archived. No reliability predictor is planned.
- AlphaPredictor, adaptive-alpha search and spatial projection-surface work remain paused and archived.

The current strongest conventional baseline is C-LBFGS: standard bidirectional confidence plus cycle consistency, the unchanged 13x13 residual grid and fully accounted L-BFGS optimization. On the existing 15 development pairs it explains most of the truth-selected diagnostic gain, but retains structural-feasibility and localized spatial-support failures. These data are development evidence rather than an independent final test.

No further experiment, training or data expansion is active after the unified review. The next decision is whether the evidenced localized support/feasibility condition is strong enough for a focused literature search or whether to change the research angle.

"""Conservative few-evaluation alpha search for RopStitch.

The original ternary_search remains untouched. This module evaluates a fixed
coarse alpha set, refines an interior candidate locally, and falls back to the
original search when the coarse optimum is on the search boundary.
"""
import time

import numpy as np
import torch

from test import test_once, ternary_search


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def adaptive_alpha_search(
    net,
    coef_net,
    input1,
    input2,
    candidates=(-1.0, -0.25, 0.5, 1.25, 2.0),
    local_iters=3,
    fallback_iters=20,
    max_out_height=4000,
):
    """Return ``(alpha, ssim, psnr, image, stats)``.

    The risk rule is fixed and deliberately transparent: a coarse optimum at
    either search boundary triggers the original ternary search. Interior
    candidates receive three local ternary rounds. No test quality is used to
    change the rule at runtime.
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim != 1 or len(candidates) < 3 or not np.all(np.diff(candidates) > 0):
        raise ValueError("candidates must be a strictly increasing 1-D sequence")
    started = time.perf_counter()
    _sync(input1.device)
    values = []
    for alpha in candidates:
        ssim, psnr, image = test_once(
            net, coef_net, input1, input2, alpha=float(alpha),
            max_out_height=max_out_height,
        )
        values.append((float(alpha), float(ssim), float(psnr), image))
    best_index = int(np.argmax([item[1] for item in values]))
    best = values[best_index]
    calls = len(values)
    fallback = best_index in (0, len(candidates) - 1)
    if fallback:
        alpha, ssim, psnr, image = ternary_search(
            net, coef_net, input1, input2,
            low=float(candidates[0]), high=float(candidates[-1]),
            max_iter=fallback_iters, max_out_height=max_out_height,
        )
        calls += 2 * fallback_iters
    else:
        low, high = float(candidates[best_index - 1]), float(candidates[best_index + 1])
        alpha, ssim, psnr, image = best
        for _ in range(local_iters):
            mid1 = low + (high - low) / 3.0
            mid2 = high - (high - low) / 3.0
            result1 = test_once(net, coef_net, input1, input2, alpha=mid1,
                                max_out_height=max_out_height)
            result2 = test_once(net, coef_net, input1, input2, alpha=mid2,
                                max_out_height=max_out_height)
            calls += 2
            if result1[0] > ssim:
                alpha, ssim, psnr, image = float(mid1), *result1
            if result2[0] > ssim:
                alpha, ssim, psnr, image = float(mid2), *result2
            if result1[0] < result2[0]:
                low = mid1
            else:
                high = mid2
    _sync(input1.device)
    return alpha, float(ssim), float(psnr), image, {
        "calls": int(calls),
        "fallback": bool(fallback),
        "coarse_best": float(best[0]),
        "elapsed_seconds": float(time.perf_counter() - started),
    }

"""One-pair e-r diagnosis using fixed, residual-blind SIFT candidates."""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import bootstrap
import cv2
import numpy as np
import torch

from eth3d_geometry import (correspondences_at_pixels, read_cameras, read_images,
                            read_raw_depth, scale_pixels)

sys.path.insert(0, str(bootstrap.ROOT / "diagnostics_geometry_20260915/scripts"))
from geometry import Warp  # verified original inverse-TPS diagnostic helper


def mutual_sift(image1, image2, nfeatures=12000):
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    keys1, desc1 = sift.detectAndCompute(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), None)
    keys2, desc2 = sift.detectAndCompute(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), None)
    if desc1 is None or desc2 is None:
        return np.empty((0, 4)), np.empty(0), {}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward, reverse = matcher.knnMatch(desc1, desc2, k=2), matcher.knnMatch(desc2, desc1, k=2)
    reverse_best = {pair[0].queryIdx: pair[0].trainIdx for pair in reverse if pair}
    records = []
    for pair in forward:
        if not pair:
            continue
        best = pair[0]
        if reverse_best.get(best.trainIdx) != best.queryIdx:
            continue
        ratio = best.distance / max(pair[1].distance, 1e-12) if len(pair) > 1 else np.inf
        records.append((*keys1[best.queryIdx].pt, *keys2[best.trainIdx].pt, ratio))
    array = np.asarray(records, dtype=np.float64)
    return array[:, :4], array[:, 4], {
        "detected_source": len(keys1), "detected_target": len(keys2),
        "mutual_candidates": len(array), "descriptor": "OpenCV SIFT",
        "candidate_rule": "mutual nearest descriptor; no Lowe-ratio, geometry, or RopStitch-residual rejection",
    }


def crop_pair(image1, image2, p, q, q_star, size=192):
    panels = []
    for image, centre in ((image1, p), (image2, q_star)):
        patch = cv2.getRectSubPix(image, (256, 256), tuple(map(float, centre)))
        panels.append(cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA))
    cv2.drawMarker(panels[0], (size//2, size//2), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
    scale = size/256
    delta = (q-q_star)*scale
    predicted = tuple(np.rint(np.array([size/2, size/2])+delta).astype(int))
    cv2.drawMarker(panels[1], (size//2, size//2), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
    if 0 <= predicted[0] < size and 0 <= predicted[1] < size:
        cv2.drawMarker(panels[1], predicted, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
    return np.hstack(panels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--rop-geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    image_root = args.scene / "images/dslr_images"
    depth_root = args.scene / "ground_truth_depth/dslr_images"
    calibration = args.scene / "dslr_calibration_jpg"
    image1, image2 = cv2.imread(str(image_root/args.source)), cv2.imread(str(image_root/args.target))
    candidates, ratios, matcher_info = mutual_sift(image1, image2)

    cameras, poses = read_cameras(calibration/"cameras.txt"), read_images(calibration/"images.txt")
    pose1, pose2 = poses[f"dslr_images/{args.source}"], poses[f"dslr_images/{args.target}"]
    camera1, camera2 = cameras[pose1.camera_id], cameras[pose2.camera_id]
    depth1, depth2 = read_raw_depth(depth_root/args.source, camera1), read_raw_depth(depth_root/args.target, camera2)
    truth = correspondences_at_pixels(camera1, pose1, depth1, camera2, pose2, depth2,
                                      candidates[:, :2])
    e_native = np.linalg.norm(candidates[:, 2:] - truth["q_star"], axis=1)

    geometry = np.load(args.rop_geometry)
    meshes = [torch.from_numpy(geometry[name]).float() for name in ("mesh_ref", "mesh_tgt")]
    both = torch.cat([mesh.reshape(-1, 2) for mesh in meshes])
    origin, extent = both.min(0).values, both.max(0).values-both.min(0).values
    warps = [Warp(mesh, origin, extent) for mesh in meshes]
    p512 = scale_pixels(candidates[:, :2], (camera1.width, camera1.height), (512, 512))
    q512 = scale_pixels(candidates[:, 2:], (camera2.width, camera2.height), (512, 512))
    with torch.no_grad():
        z1, inverse1, valid1 = warps[0].invert(torch.from_numpy(p512).float()/256-1)
        z2, inverse2, valid2 = warps[1].invert(torch.from_numpy(q512).float()/256-1)
        canvas1 = (z1+1)*extent/2+origin
        canvas2 = (z2+1)*extent/2+origin
        r_canvas = (canvas1-canvas2).norm(dim=1).numpy()
        inverse_valid = (valid1 & valid2).numpy()
    r_canvas[~inverse_valid] = np.nan

    evaluable = truth["evaluable"] & np.isfinite(e_native) & np.isfinite(r_canvas)
    correctness = np.full(len(candidates), "unable", dtype=object)
    correctness[evaluable & (e_native <= 3)] = "correct"
    correctness[evaluable & (e_native > 3) & (e_native <= 6)] = "uncertain"
    correctness[evaluable & (e_native > 6)] = "incorrect"
    rows = []
    for index, match in enumerate(candidates):
        rows.append({
            "index": index, "p_x": match[0], "p_y": match[1],
            "q_x": match[2], "q_y": match[3],
            "q_star_x": truth["q_star"][index, 0], "q_star_y": truth["q_star"][index, 1],
            "gt_label": str(truth["label"][index]), "correctness": str(correctness[index]),
            "e_native_px": e_native[index], "r_canvas_px": r_canvas[index],
            "lowe_ratio": ratios[index], "inverse_valid": bool(inverse_valid[index]),
        })
    with (args.output/"matches.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(args.output/"matches.npz", candidates=candidates, ratios=ratios,
                        q_star=truth["q_star"], gt_label=truth["label"],
                        e_native=e_native, r_canvas=r_canvas, correctness=correctness)

    plot = np.full((900, 1260, 3), 255, np.uint8)
    left, right, top, bottom = 100, 1220, 70, 820
    cv2.rectangle(plot, (left, top), (right, bottom), (30, 30, 30), 2)
    palette = {"correct": (119, 158, 27), "uncertain": (2, 171, 230),
               "incorrect": (2, 95, 217)}
    display = correctness != "unable"
    transform_e = np.log1p(np.maximum(e_native, 0)/3)
    transform_r = np.log1p(np.maximum(r_canvas, 0)/3)
    max_e = np.nanpercentile(transform_e[display], 99.5) if display.any() else 1
    max_r = np.nanpercentile(transform_r[display], 99.5) if display.any() else 1
    for label in ("incorrect", "uncertain", "correct"):
        mask = correctness == label
        ids = np.flatnonzero(mask)
        for index in ids:
            x = int(left + np.clip(transform_e[index]/max(max_e, 1e-8), 0, 1)*(right-left))
            y = int(bottom - np.clip(transform_r[index]/max(max_r, 1e-8), 0, 1)*(bottom-top))
            cv2.circle(plot, (x, y), 3, palette[label], -1, cv2.LINE_AA)
    cv2.putText(plot, f"{args.source} -> {args.target}: fixed mutual-SIFT candidates",
                (left, 38), cv2.FONT_HERSHEY_SIMPLEX, .75, (25, 25, 25), 2)
    cv2.putText(plot, "match error e (native target px; log1p scale)",
                (420, 875), cv2.FONT_HERSHEY_SIMPLEX, .62, (25, 25, 25), 1)
    cv2.putText(plot, "initial residual r (canvas px; log1p scale)",
                (left+10, top+28), cv2.FONT_HERSHEY_SIMPLEX, .55, (25, 25, 25), 1)
    for item, label in enumerate(("correct", "uncertain", "incorrect")):
        x = 850 + item*120
        cv2.circle(plot, (x, 55), 6, palette[label], -1)
        cv2.putText(plot, f"{label} {int((correctness==label).sum())}", (x+10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (25, 25, 25), 1)
    cv2.imwrite(str(args.output/"e_r_scatter.png"), plot)

    correct = correctness == "correct"
    high_ids = np.flatnonzero(correct & np.isfinite(r_canvas))
    high_ids = high_ids[np.argsort(r_canvas[high_ids])[::-1]][:12]
    tiles = []
    for index in high_ids:
        tile = crop_pair(image1, image2, candidates[index, :2], candidates[index, 2:], truth["q_star"][index])
        cv2.putText(tile, f"#{index} e={e_native[index]:.1f} r={r_canvas[index]:.1f}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(tile, f"#{index} e={e_native[index]:.1f} r={r_canvas[index]:.1f}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(tile)
    if tiles:
        cv2.imwrite(str(args.output/"correct_high_residual_examples.jpg"), np.vstack(tiles))

    sensitivity = {}
    for threshold in (5, 10, 20):
        sensitivity[str(threshold)] = {
            "correct_large_residual": int((correct & (r_canvas >= threshold)).sum()),
            "correct_total": int(correct.sum()),
            "fraction": float((correct & (r_canvas >= threshold)).sum()/max(correct.sum(), 1)),
        }
    reliability = {}
    for threshold in (.6, .7, .8):
        accepted = ratios <= threshold
        reliability[str(threshold)] = {
            "correct_retention": float((accepted & correct).sum()/max(correct.sum(), 1)),
            "incorrect_acceptance": float((accepted & (correctness == "incorrect")).sum()
                                          / max((correctness == "incorrect").sum(), 1)),
        }
    report = {
        "status": "one_pair_diagnostic_complete", "source": args.source, "target": args.target,
        "matcher": matcher_info, "candidate_gt_labels": dict(Counter(map(str, truth["label"]))),
        "correctness_counts": dict(Counter(map(str, correctness))),
        "threshold_protocol": {"correct": "e<=3 native px", "uncertain": "3<e<=6 native px",
                               "incorrect": "e>6 native px", "large_r": "not locked; sensitivity only"},
        "large_residual_sensitivity": sensitivity, "lowe_ratio_diagnostic": reliability,
        "r_definition": "distance between the two forward-mapped match locations in the fixed initial RopStitch canvas",
        "limits": ["single scene and pair", "SIFT is a diagnostic candidate generator, not the proposed method",
                   "CPU RopStitch export is functional only and excluded from timing conclusions"],
    }
    (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

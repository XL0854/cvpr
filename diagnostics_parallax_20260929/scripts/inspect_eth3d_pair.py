"""Build and visualize ETH3D common-visible geometry for one ordered pair."""
import argparse
import json
from collections import Counter
from pathlib import Path

import bootstrap
import cv2
import numpy as np

from eth3d_geometry import (common_visible_correspondences, read_cameras,
                            read_images, read_raw_depth)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--source", required=True, help="basename, e.g. DSC_0286.JPG")
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--max-lines", type=int, default=500)
    return parser.parse_args()


def depth_preview(depth):
    valid = np.isfinite(depth) & (depth > 0)
    gray = np.zeros(depth.shape, np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        gray[valid] = np.clip((depth[valid] - lo) / max(hi-lo, 1e-8) * 255, 0, 255)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def resized(image, scale=0.25):
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def main():
    args = arguments()
    calibration = args.scene / "dslr_calibration_jpg"
    image_root = args.scene / "images" / "dslr_images"
    depth_root = args.scene / "ground_truth_depth" / "dslr_images"
    cameras = read_cameras(calibration / "cameras.txt")
    poses = read_images(calibration / "images.txt")
    source_key, target_key = f"dslr_images/{args.source}", f"dslr_images/{args.target}"
    source_pose, target_pose = poses[source_key], poses[target_key]
    source_camera, target_camera = cameras[source_pose.camera_id], cameras[target_pose.camera_id]
    source_depth = read_raw_depth(depth_root / args.source, source_camera)
    target_depth = read_raw_depth(depth_root / args.target, target_camera)
    result = common_visible_correspondences(
        source_camera, source_pose, source_depth,
        target_camera, target_pose, target_depth, stride=args.stride)

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "geometry.npz", **result)
    source_image = cv2.imread(str(image_root / args.source), cv2.IMREAD_COLOR)
    target_image = cv2.imread(str(image_root / args.target), cv2.IMREAD_COLOR)
    if source_image is None or target_image is None:
        raise FileNotFoundError("Could not read one of the RGB images")
    cv2.imwrite(str(args.output / "source_depth.png"), resized(depth_preview(source_depth)))
    cv2.imwrite(str(args.output / "target_depth.png"), resized(depth_preview(target_depth)))

    source_overlay, target_overlay = source_image.copy(), target_image.copy()
    colors = {
        "evaluable": (40, 220, 40), "target_depth_missing": (0, 180, 255),
        "occluded_in_target": (20, 20, 230), "depth_inconsistent": (200, 80, 220),
        "uncertain_source_boundary": (255, 180, 30),
        "outside_or_behind_target": (120, 120, 120),
        "invalid_source_depth": (0, 0, 0),
    }
    for point, label in zip(result["p"], result["label"]):
        if label != "invalid_source_depth":
            cv2.circle(source_overlay, tuple(np.rint(point).astype(int)), 5, colors[str(label)], -1)
    for point, label in zip(result["q_star"], result["label"]):
        if np.isfinite(point).all() and label != "invalid_source_depth":
            cv2.circle(target_overlay, tuple(np.rint(point).astype(int)), 5, colors[str(label)], -1)
    cv2.imwrite(str(args.output / "source_labels.png"), resized(source_overlay))
    cv2.imwrite(str(args.output / "target_projections.png"), resized(target_overlay))

    scale = 0.18
    left, right = resized(source_image, scale), resized(target_image, scale)
    canvas = np.concatenate((left, right), axis=1)
    evaluable_ids = np.flatnonzero(result["evaluable"])
    if len(evaluable_ids) > args.max_lines:
        evaluable_ids = evaluable_ids[np.linspace(0, len(evaluable_ids)-1,
                                                  args.max_lines).astype(int)]
    offset = np.array([left.shape[1], 0])
    for index in evaluable_ids:
        p = np.rint(result["p"][index] * scale).astype(int)
        q = np.rint(result["q_star"][index] * scale + offset).astype(int)
        cv2.line(canvas, tuple(p), tuple(q), (60, 220, 60), 1, cv2.LINE_AA)
    cv2.imwrite(str(args.output / "common_visible_correspondences.jpg"), canvas)

    counts = Counter(map(str, result["label"]))
    sampled = len(result["p"])
    valid_source = sampled - counts.get("invalid_source_depth", 0)
    evaluable = counts.get("evaluable", 0)
    valid_depth_residual = (np.isfinite(result["target_observed_depth"])
                            & np.isfinite(result["target_predicted_depth"]))
    residual = np.abs(result["target_observed_depth"][valid_depth_residual]
                      - result["target_predicted_depth"][valid_depth_residual])
    report = {
        "status": "geometry_built_needs_visual_review",
        "scene": args.scene.name, "source": args.source, "target": args.target,
        "stride_px": args.stride, "camera_models": [source_camera.model, target_camera.model],
        "sampled_points": sampled, "label_counts": dict(sorted(counts.items())),
        "valid_source_depth_fraction": valid_source / sampled if sampled else 0.0,
        "evaluable_fraction_of_grid": evaluable / sampled if sampled else 0.0,
        "evaluable_fraction_of_valid_source": evaluable / valid_source if valid_source else 0.0,
        "target_depth_residual_median_m": float(np.median(residual)) if residual.size else None,
        "target_depth_residual_p95_m": float(np.percentile(residual, 95)) if residual.size else None,
        "pixel_convention": "array pixel centres; ETH3D cx/cy shifted by -0.5",
        "depth_use": "evaluation only",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

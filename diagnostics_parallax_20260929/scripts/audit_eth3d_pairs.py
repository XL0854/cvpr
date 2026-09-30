"""Audit GT coverage for nearby ETH3D frames without loading depth into RAM."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import bootstrap
import numpy as np

from eth3d_geometry import (camera_to_world, common_visible_correspondences,
                            read_cameras, read_images)


def camera_center(pose):
    return camera_to_world(pose, np.zeros((1, 3), dtype=np.float64))[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--offsets", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    calibration = args.scene / "dslr_calibration_jpg"
    depth_root = args.scene / "ground_truth_depth" / "dslr_images"
    cameras = read_cameras(calibration / "cameras.txt")
    poses_by_key = read_images(calibration / "images.txt")
    poses = sorted(poses_by_key.values(), key=lambda item: Path(item.name).name)
    rows = []
    for offset in args.offsets:
        for source_index in range(len(poses) - offset):
            source_pose, target_pose = poses[source_index], poses[source_index + offset]
            source_camera = cameras[source_pose.camera_id]
            target_camera = cameras[target_pose.camera_id]
            source_name, target_name = Path(source_pose.name).name, Path(target_pose.name).name
            source_depth = np.memmap(depth_root / source_name, dtype="<f4", mode="r",
                                     shape=(source_camera.height, source_camera.width))
            target_depth = np.memmap(depth_root / target_name, dtype="<f4", mode="r",
                                     shape=(target_camera.height, target_camera.width))
            result = common_visible_correspondences(
                source_camera, source_pose, source_depth,
                target_camera, target_pose, target_depth, stride=args.stride)
            counts = Counter(map(str, result["label"]))
            total = len(result["label"])
            valid_source = total - counts.get("invalid_source_depth", 0)
            evaluable = counts.get("evaluable", 0)
            rows.append({
                "source": source_name, "target": target_name, "frame_offset": offset,
                "baseline_m": float(np.linalg.norm(camera_center(source_pose)-camera_center(target_pose))),
                "sampled": total, "valid_source": valid_source, "evaluable": evaluable,
                "valid_source_fraction": valid_source/total if total else 0.0,
                "evaluable_grid_fraction": evaluable/total if total else 0.0,
                "evaluable_valid_source_fraction": evaluable/valid_source if valid_source else 0.0,
                "target_depth_missing": counts.get("target_depth_missing", 0),
                "occluded": counts.get("occluded_in_target", 0),
                "depth_inconsistent": counts.get("depth_inconsistent", 0),
                "outside": counts.get("outside_or_behind_target", 0),
            })
            print(f"{source_name}->{target_name} offset={offset} eval={evaluable} "
                  f"baseline={rows[-1]['baseline_m']:.3f}m")
    args.output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with (args.output / "pair_coverage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    ranked = sorted(rows, key=lambda row: row["evaluable"], reverse=True)
    summary = {
        "scene": args.scene.name, "stride_px": args.stride,
        "offsets": args.offsets, "pairs": len(rows),
        "median_evaluable": float(np.median([row["evaluable"] for row in rows])) if rows else 0,
        "top_pairs": ranked[:10],
        "selection_note": "Coverage audit only; final development pairs must also span independent scenes.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

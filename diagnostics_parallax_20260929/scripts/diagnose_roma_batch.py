"""Evaluate fixed RoMa candidates against ETH3D truth and initial Rop meshes."""
import argparse
import csv
import json
from pathlib import Path
import sys

import bootstrap
import numpy as np
import torch

from eth3d_geometry import correspondences_at_pixels, read_cameras, read_images, read_raw_depth

sys.path.insert(0, str(bootstrap.ROOT/"diagnostics_geometry_20260915/scripts"))
from geometry import Warp


def retention_at_wrong_acceptance(score, higher_better, correct, large, small, incorrect):
    result = {}
    signed = score if higher_better else -score
    wrong = np.sort(signed[incorrect])[::-1]
    for target in (.01, .05, .10):
        count = max(1, int(np.floor(target*len(wrong)))) if len(wrong) else 0
        threshold = wrong[count-1] if count else np.inf
        accepted = signed >= threshold
        result[str(target)] = {
            "threshold": float(threshold if higher_better else -threshold),
            "incorrect_acceptance": float((accepted&incorrect).sum()/max(incorrect.sum(), 1)),
            "all_correct_retention": float((accepted&correct).sum()/max(correct.sum(), 1)),
            "large_correct_retention": float((accepted&large).sum()/max(large.sum(), 1)),
            "small_correct_retention": float((accepted&small).sum()/max(small.sum(), 1)),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--rop-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.manifest.read_text())
    scene = bootstrap.ROOT/config["scene"]
    calibration, depth_root = scene/"dslr_calibration_jpg", scene/"ground_truth_depth/dslr_images"
    cameras, poses = read_cameras(calibration/"cameras.txt"), read_images(calibration/"images.txt")
    args.output.mkdir(parents=True, exist_ok=True)
    aggregate = []
    for source_name, target_name in config["pairs"]:
        pair_id = f"{Path(source_name).stem}_{Path(target_name).stem}"
        candidate_path = args.candidates/pair_id/"candidates.npz"
        rop_path = args.rop_root/pair_id/"initial_geometry.npz"
        if not rop_path.exists():
            short_id = f"{Path(source_name).stem.removeprefix('DSC_')}_{Path(target_name).stem.removeprefix('DSC_')}"
            legacy = args.rop_root/f"rop_eth3d_courtyard_{short_id}_cpu"/"initial_geometry.npz"
            rop_path = legacy if legacy.exists() else rop_path
        if not candidate_path.exists():
            raise FileNotFoundError(candidate_path)
        data, geometry = np.load(candidate_path), np.load(rop_path)
        pose1, pose2 = poses[f"dslr_images/{source_name}"], poses[f"dslr_images/{target_name}"]
        camera1, camera2 = cameras[pose1.camera_id], cameras[pose2.camera_id]
        depth1, depth2 = read_raw_depth(depth_root/source_name, camera1), read_raw_depth(depth_root/target_name, camera2)
        truth = correspondences_at_pixels(camera1, pose1, depth1, camera2, pose2, depth2,
                                          data["source_native"])
        e = np.linalg.norm(data["target_native"]-truth["q_star"], axis=1)
        meshes = [torch.from_numpy(geometry[name]).float() for name in ("mesh_ref", "mesh_tgt")]
        both = torch.cat([mesh.reshape(-1, 2) for mesh in meshes])
        origin, extent = both.min(0).values, both.max(0).values-both.min(0).values
        warps = [Warp(mesh, origin, extent) for mesh in meshes]
        with torch.no_grad():
            z1, _, valid1 = warps[0].invert(torch.from_numpy(data["source_512"]).float()/256-1)
            z2, _, valid2 = warps[1].invert(torch.from_numpy(data["target_512"]).float()/256-1)
            canvas1, canvas2 = (z1+1)*extent/2+origin, (z2+1)*extent/2+origin
            residual = (canvas1-canvas2).norm(dim=1).numpy()
            inverse_valid = (valid1&valid2).numpy()
        residual[~inverse_valid] = np.nan
        evaluable = truth["evaluable"] & data["valid"] & inverse_valid & np.isfinite(e)
        correctness = np.full(len(e), "unable", object)
        correctness[evaluable & (e <= 3)] = "correct"
        correctness[evaluable & (e > 3) & (e <= 6)] = "uncertain"
        correctness[evaluable & (e > 6)] = "incorrect"
        correct, incorrect = correctness == "correct", correctness == "incorrect"
        large, small = correct & (residual >= 10), correct & (residual < 10)
        confidence = np.minimum(data["confidence"], data["reverse_confidence"])
        filters = {
            "min_bidirectional_confidence": retention_at_wrong_acceptance(
                confidence, True, correct, large, small, incorrect),
            "cycle_error": retention_at_wrong_acceptance(
                data["cycle_512_px"], False, correct, large, small, incorrect),
            "current_shape_residual": retention_at_wrong_acceptance(
                residual, False, correct, large, small, incorrect),
        }
        pair_output = args.output/pair_id; pair_output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pair_output/"diagnosis.npz", e_native=e, r_canvas=residual,
                            correctness=correctness, gt_label=truth["label"],
                            confidence=data["confidence"], reverse_confidence=data["reverse_confidence"],
                            cycle_512_px=data["cycle_512_px"])
        rows = [{"index": i, "gt_label": truth["label"][i], "correctness": correctness[i],
                 "e_native_px": e[i], "r_canvas_px": residual[i],
                 "confidence": data["confidence"][i], "reverse_confidence": data["reverse_confidence"][i],
                 "cycle_512_px": data["cycle_512_px"][i]} for i in range(len(e))]
        with (pair_output/"candidates.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        row = {"id": pair_id, "candidates": len(e), "evaluable": int(evaluable.sum()),
               "correct": int(correct.sum()), "uncertain": int((correctness == "uncertain").sum()),
               "unable": int((correctness == "unable").sum()), "incorrect": int(incorrect.sum()),
               "correct_r_ge_10": int(large.sum()),
               "correct_r_ge_10_fraction": float(large.sum()/max(correct.sum(), 1)),
               "correct_r_median": float(np.median(residual[correct])) if correct.any() else None,
               "filters": filters}
        (pair_output/"report.json").write_text(json.dumps(row, indent=2)+"\n")
        aggregate.append(row); print("DONE", pair_id, flush=True)
    report = {"status": "completed", "protocol": {
        "candidate_set": "fixed 8px regular source grid with unfiltered RoMa predictions",
        "truth": "ETH3D depth/calibration; unavailable points excluded from correctness",
        "large_correct": "e<=3 native px and r>=10 canvas px",
        "filters": "thresholded independently at matched incorrect-acceptance rates"},
        "pairs": aggregate}
    (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print("ROMA_DIAGNOSIS_COMPLETED", args.output)


if __name__ == "__main__":
    main()

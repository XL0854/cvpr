"""A/B/C/D fixed-reference residual-grid diagnosis on a fixed RoMa candidate set."""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import bootstrap
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from eth3d_geometry import (correspondences_at_pixels, read_cameras, read_images,
                            read_raw_depth, scale_pixels)

sys.path.insert(0, str(bootstrap.ROOT/"diagnostics_geometry_20260915/scripts"))
from geometry import Warp, image_scores, render, structure, triangle_areas


def stats(values):
    values = values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    return {"n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90))}


def sampled_tps_determinants(warp):
    mesh = warp.controls.reshape(13, 13, 2)
    samples = []
    for u in (.2, .5, .8):
        for v in (.2, .5, .8):
            samples.append(((1-u)*(1-v)*mesh[:-1, :-1]+u*(1-v)*mesh[:-1, 1:]
                            +(1-u)*v*mesh[1:, :-1]+u*v*mesh[1:, 1:]).reshape(-1, 2))
    _, jacobian = warp.at(torch.cat(samples), True)
    return torch.linalg.det(jacobian)


def fit_target(base_ref, base_tgt, origin, extent, p512, q512, selected, weights,
               steps, device):
    ref_warp = Warp(base_ref, origin, extent)
    with torch.no_grad():
        z_ref, _, invertible = ref_warp.invert(p512/256-1)
    selected = selected & invertible
    ids = torch.nonzero(selected, as_tuple=False).flatten()
    if len(ids) < 16:
        return base_tgt.detach().clone(), {"status": "insufficient_points", "points": len(ids)}
    z, target, weight = z_ref[ids], q512[ids]/256-1, weights[ids]
    delta = torch.nn.Parameter(torch.zeros_like(base_tgt))
    optimizer = torch.optim.Adam([delta], lr=.2)
    initial_area = triangle_areas(base_tgt).detach()
    baseline_det = sampled_tps_determinants(Warp(base_tgt, origin, extent)).detach()
    baseline_sign = torch.sign(baseline_det)
    feasible_delta = delta.detach().clone()
    history = []
    for step in range(steps):
        optimizer.zero_grad()
        current = base_tgt+delta
        warp = Warp(current, origin, extent)
        error = (warp.at(z)-target)*256
        point_loss = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none", beta=2).mean(-1)
        geometry = (point_loss*weight).sum()/weight.sum().clamp_min(1e-6)
        anchor = delta.square().mean()
        grid = delta.reshape(13, 13, 2)
        smooth = (grid[1:]-grid[:-1]).square().mean()+(grid[:, 1:]-grid[:, :-1]).square().mean()
        ratio = triangle_areas(current)/initial_area.abs().clamp_min(1.)
        fold = F.relu(.2-ratio).square().mean()
        determinant = sampled_tps_determinants(warp)
        tps_fold = F.relu(.2*baseline_det.abs()-determinant*baseline_sign).square().mean()
        loss = geometry+.002*anchor+.02*smooth+10*fold+100*tps_fold
        if not torch.isfinite(loss):
            return base_tgt.detach().clone(), {"status": "nonfinite_reverted", "step": step}
        loss.backward(); optimizer.step()
        with torch.no_grad():
            delta.clamp_(-128, 128)
            candidate = base_tgt+delta
            candidate_det = sampled_tps_determinants(Warp(candidate, origin, extent))
            if (triangle_areas(candidate) > 0).all() and (candidate_det*baseline_sign > 0).all():
                feasible_delta.copy_(delta)
        if step % 25 == 0 or step == steps-1:
            history.append({"step": step, "loss": float(loss.detach()),
                            "geometry": float(geometry.detach()),
                            "anchor": float(anchor.detach()), "smooth": float(smooth.detach()),
                            "fold": float(fold.detach()), "tps_fold": float(tps_fold.detach())})
    final = base_tgt+delta.detach()
    final_det = sampled_tps_determinants(Warp(final, origin, extent)).detach()
    reverted = not ((triangle_areas(final) > 0).all() and (final_det*baseline_sign > 0).all())
    if reverted:
        final = base_tgt+feasible_delta
    return final, {
        "status": "completed", "points": len(ids), "steps": steps,
        "reverted_to_last_feasible": bool(reverted),
        "max_displacement_px": float((final-base_tgt).norm(dim=-1).max()), "history": history}


def spatial_count_control(first, second, cell):
    """Match two selections per source 8x8 cell using fixed-grid order."""
    kept = [np.zeros(len(cell), dtype=bool), np.zeros(len(cell), dtype=bool)]
    for value in range(64):
        ids = [np.flatnonzero(mask & (cell == value)) for mask in (first, second)]
        count = min(map(len, ids))
        if not count:
            continue
        for output, candidates in zip(kept, ids):
            chosen = candidates[np.rint(np.linspace(0, len(candidates)-1, count)).astype(int)]
            output[chosen] = True
    return kept


@torch.no_grad()
def evaluate(ref_mesh, tgt_mesh, origin, extent, p512, qstar512):
    ref, tgt = Warp(ref_mesh, origin, extent), Warp(tgt_mesh, origin, extent)
    z_ref, _, valid_ref = ref.invert(p512/256-1)
    z_tgt, _, valid_tgt = tgt.invert(qstar512/256-1)
    reprojection = (tgt.at(z_ref)-(qstar512/256-1)).norm(dim=-1)*256
    canvas = ((z_ref-z_tgt)*extent/2).norm(dim=-1)
    valid = valid_ref & valid_tgt & torch.isfinite(reprojection) & torch.isfinite(canvas)
    reprojection = torch.where(valid, reprojection, torch.full_like(reprojection, 512))
    canvas = torch.where(valid, canvas, torch.full_like(canvas, 512))
    return reprojection, canvas, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--rop-root", type=Path, required=True)
    parser.add_argument("--diagnosis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(args.device)
    config = json.loads(args.manifest.read_text())
    pairs = config["pairs"][:args.limit] if args.limit else config["pairs"]
    scene = bootstrap.ROOT/config["scene"]
    calibration, depth_root = scene/"dslr_calibration_jpg", scene/"ground_truth_depth/dslr_images"
    cameras, poses = read_cameras(calibration/"cameras.txt"), read_images(calibration/"images.txt")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for source_name, target_name in pairs:
        started = time.perf_counter()
        pair_id = f"{Path(source_name).stem}_{Path(target_name).stem}"
        pair_output = args.output/pair_id; pair_output.mkdir(parents=True, exist_ok=True)
        data = np.load(args.candidates/pair_id/"candidates.npz")
        geometry_path = args.rop_root/pair_id/"initial_geometry.npz"
        if not geometry_path.exists():
            short_id = f"{Path(source_name).stem.removeprefix('DSC_')}_{Path(target_name).stem.removeprefix('DSC_')}"
            legacy = args.rop_root/f"rop_eth3d_courtyard_{short_id}_cpu"/"initial_geometry.npz"
            geometry_path = legacy if legacy.exists() else geometry_path
        geometry = np.load(geometry_path)
        pose1, pose2 = poses[f"dslr_images/{source_name}"], poses[f"dslr_images/{target_name}"]
        camera1, camera2 = cameras[pose1.camera_id], cameras[pose2.camera_id]
        depth1, depth2 = read_raw_depth(depth_root/source_name, camera1), read_raw_depth(depth_root/target_name, camera2)
        truth = correspondences_at_pixels(camera1, pose1, depth1, camera2, pose2, depth2,
                                          data["source_native"])
        e = np.linalg.norm(data["target_native"]-truth["q_star"], axis=1)
        p512 = torch.from_numpy(data["source_512"]).float().to(device)
        q512 = torch.from_numpy(data["target_512"]).float().to(device)
        qstar512 = torch.from_numpy(scale_pixels(truth["q_star"],
                                                 (camera2.width, camera2.height), (512, 512))).float().to(device)
        base_ref = torch.from_numpy(geometry["mesh_ref"]).float().to(device)
        base_tgt = torch.from_numpy(geometry["mesh_tgt"]).float().to(device)
        both = torch.cat((base_ref.reshape(-1, 2), base_tgt.reshape(-1, 2)))
        origin, extent = both.min(0).values, both.max(0).values-both.min(0).values
        initial_reprojection, initial_canvas, initial_valid = evaluate(
            base_ref, base_tgt, origin, extent, p512, qstar512)
        cell = (data["source_512"][:, 0]//64).astype(int)+8*(data["source_512"][:, 1]//64).astype(int)
        held = (cell % 4) == 0
        train = ~held
        truth_evaluable = torch.from_numpy(truth["evaluable"] & data["valid"]).to(device)
        evaluation = truth_evaluable & torch.from_numpy(held).to(device)
        baseline_aligned = evaluation & (initial_canvas <= 3)
        baseline_difficult = evaluation & (initial_canvas >= 10)
        confidence = np.minimum(data["confidence"], data["reverse_confidence"])
        current_r = np.load(args.diagnosis_root/pair_id/"diagnosis.npz",
                            allow_pickle=True)["r_canvas"]
        selections = {
            "B_residual": train & data["valid"] & np.isfinite(current_r) & (current_r <= 6),
            "C_reliability": train & data["valid"] & (data["confidence"] >= .5)
                             & (data["reverse_confidence"] >= .5) & (data["cycle_512_px"] <= 2),
            "D_oracle_select": train & data["valid"] & truth["evaluable"] & (e <= 3),
        }
        weights = {
            "B_residual": np.ones(len(e), np.float32),
            "C_reliability": np.clip(confidence, .05, 1).astype(np.float32),
            "D_oracle_select": np.ones(len(e), np.float32),
        }
        c_control, d_control = spatial_count_control(
            selections["C_reliability"], selections["D_oracle_select"], cell)
        selections.update({"C_spatial_count_control": c_control,
                           "D_spatial_count_control": d_control})
        weights.update({"C_spatial_count_control": np.ones(len(e), np.float32),
                        "D_spatial_count_control": np.ones(len(e), np.float32)})
        meshes, optimization = {"A_baseline": base_tgt}, {"A_baseline": {"status": "not_optimized"}}
        for group in ("B_residual", "C_reliability", "D_oracle_select",
                      "C_spatial_count_control", "D_spatial_count_control"):
            fit_started = time.perf_counter()
            meshes[group], optimization[group] = fit_target(
                base_ref, base_tgt, origin, extent, p512, q512,
                torch.from_numpy(selections[group]).to(device),
                torch.from_numpy(weights[group]).to(device), args.steps, device)
            optimization[group]["elapsed_seconds"] = time.perf_counter()-fit_started
        image_paths = [scene/"images/dslr_images"/name for name in (source_name, target_name)]
        images = [cv2.resize(cv2.imread(str(path)), (512, 512), interpolation=cv2.INTER_AREA) for path in image_paths]
        tensors = [torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1))[None].to(device) for image in images]
        selection_audit = {}
        for group, mask in selections.items():
            selected_weights = weights[group][mask]
            selection_audit[group] = {
                "points": int(mask.sum()), "source_cells_8x8": int(len(np.unique(cell[mask]))),
                "weight_mean": float(selected_weights.mean()) if len(selected_weights) else None,
                "weight_min": float(selected_weights.min()) if len(selected_weights) else None,
                "weight_max": float(selected_weights.max()) if len(selected_weights) else None,
            }
        result = {"id": pair_id, "held_truth_points": int(evaluation.sum()),
                  "held_initial_aligned": int(baseline_aligned.sum()),
                  "held_initial_difficult": int(baseline_difficult.sum()),
                  "selection_audit": selection_audit, "optimization": optimization, "groups": {}}
        output_size = (max(16, int(float(extent[1]))), max(16, int(float(extent[0]))))
        ref_render, ref_mask = render(Warp(base_ref, origin, extent), tensors[0], output_size)
        baseline_area = triangle_areas(base_tgt).abs().sum()
        for group, mesh in meshes.items():
            reprojection, canvas, valid = evaluate(base_ref, mesh, origin, extent, p512, qstar512)
            target_render, target_mask = render(Warp(mesh, origin, extent), tensors[1], output_size)
            overlap = ref_mask & target_mask
            fusion = (ref_render*ref_mask[..., None]+target_render*target_mask[..., None]) \
                     / np.maximum(ref_mask.astype(float)+target_mask, 1)[..., None]
            cv2.imwrite(str(pair_output/f"{group}_fusion.jpg"), np.clip(fusion, 0, 255).astype(np.uint8))
            group_result = {
                "held_reprojection_512": stats(reprojection[evaluation]),
                "held_canvas_error": stats(canvas[evaluation]),
                "aligned_canvas_error": stats(canvas[baseline_aligned]) if baseline_aligned.any() else {"n": 0},
                "difficult_canvas_error": stats(canvas[baseline_difficult]) if baseline_difficult.any() else {"n": 0},
                "held_inversion_failure": float((~valid[evaluation]).float().mean()) if evaluation.any() else None,
                "structure": structure(Warp(mesh, origin, extent)),
                "triangle_area_ratio": float(triangle_areas(mesh).abs().sum()/baseline_area),
                "overlap_pixels": int(overlap.sum()),
                "image_auxiliary": image_scores(ref_render, target_render, overlap),
            }
            result["groups"][group] = group_result
            np.save(pair_output/f"{group}_mesh_tgt.npy", mesh.detach().cpu().numpy())
            rows.append({"pair": pair_id, "group": group,
                         "held_canvas_mean": group_result["held_canvas_error"]["mean"],
                         "held_canvas_median": group_result["held_canvas_error"]["median"],
                         "difficult_canvas_mean": group_result["difficult_canvas_error"].get("mean"),
                         "aligned_canvas_mean": group_result["aligned_canvas_error"].get("mean"),
                         "fold_fraction": group_result["structure"]["sampled_tps_fold_fraction"],
                         "triangle_area_ratio": group_result["triangle_area_ratio"],
                         "overlap_pixels": group_result["overlap_pixels"]})
        np.savez_compressed(pair_output/"held_point_metrics.npz",
                            evaluation=evaluation.cpu().numpy(),
                            initially_aligned=baseline_aligned.cpu().numpy(),
                            initially_difficult=baseline_difficult.cpu().numpy(), cell=cell,
                            **{f"{name}_canvas": evaluate(base_ref, mesh, origin, extent,
                                  p512, qstar512)[1].cpu().numpy() for name, mesh in meshes.items()})
        result["seconds"] = time.perf_counter()-started
        (pair_output/"result.json").write_text(json.dumps(result, indent=2)+"\n")
        print("DONE", pair_id, f"seconds={result['seconds']:.2f}", flush=True)
    with (args.output/"summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output/"run.json").write_text(json.dumps({"status": "completed", "steps": args.steps,
        "device": str(device), "pairs": len(pairs), "reference_side_fixed": True,
        "heldout_rule": "source 8x8 cell_id mod 4 == 0; never optimized"}, indent=2)+"\n")


if __name__ == "__main__":
    main()

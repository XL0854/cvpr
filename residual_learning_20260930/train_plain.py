"""Train/evaluate the plain residual-grid network on cached ETH3D samples."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diagnostics_parallax_20260929/scripts"))
sys.path.insert(0, str(ROOT / "diagnostics_geometry_20260915/scripts"))
from diagnose_residual_bottleneck import eval_points, sampled, triangles
from geometry import Warp
from model import PlainResidualGridNet, parameter_count


def load_records(cache: Path):
    manifest = json.loads((cache / "manifest.json").read_text())
    return manifest, {name: [row for row in manifest["records"] if row["split"] == name]
                      for name in {row["split"] for row in manifest["records"]}}


def load_sample(row, device):
    sample = torch.load(ROOT / row["path"], map_location=device, weights_only=False)
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in sample.items()}


def structural_terms(base, current, origin, extent):
    anchor = (current - base).square().mean()
    delta = (current - base).reshape(13, 13, 2)
    smooth = ((delta[1:] - delta[:-1]).square().mean()
              + (delta[:, 1:] - delta[:, :-1]).square().mean())
    initial_area = triangles(base).detach().abs().clamp_min(1.0)
    area_ratio = triangles(current) / initial_area
    triangle_barrier = F.relu(.2 - area_ratio).square().mean()
    base_det, _ = sampled(Warp(base, origin, extent))
    current_det, _ = sampled(Warp(current, origin, extent))
    tps_barrier = F.relu(.2 * base_det.detach().abs()
                         - current_det * torch.sign(base_det.detach())).square().mean()
    return anchor, smooth, triangle_barrier, tps_barrier


def losses(model, sample, weights):
    displacement = model(sample["input"][None])[0]
    base, ref = sample["tgt_mesh"], sample["ref_mesh"]
    current = base + displacement
    warp = Warp(current, sample["origin"], sample["extent"])
    train = ~sample["held"]
    predicted = warp.at(sample["z_ref"][train])
    error = (predicted - (sample["qstar_512"][train] / 256 - 1)) * 256
    geometry = F.smooth_l1_loss(error, torch.zeros_like(error), beta=2.0)

    aligned = train & (sample["initial_canvas"] <= 3.0)
    if aligned.any():
        aligned_error = ((warp.at(sample["z_ref"][aligned])
                          - (sample["qstar_512"][aligned] / 256 - 1)).norm(dim=-1) * 256)
        # Protect final geometric accuracy with a tolerance; displacement itself
        # is free to change if the correspondence error does not get worse.
        protection = F.relu(aligned_error - sample["initial_forward"][aligned] - .5).square().mean()
    else:
        protection = geometry.new_zeros(())

    anchor, smooth, tri, tps = structural_terms(base, current, sample["origin"], sample["extent"])
    overlap_uv = warp.at(sample["overlap_z"])
    overlap = F.relu(overlap_uv.abs() - .98).square().mean()
    total = (geometry + weights["protection"] * protection + weights["anchor"] * anchor
             + weights["smooth"] * smooth + weights["triangle"] * tri
             + weights["tps"] * tps + weights["overlap"] * overlap)
    terms = {"total": total, "geometry": geometry, "protection": protection,
             "anchor": anchor, "smooth": smooth, "triangle": tri,
             "tps": tps, "overlap": overlap}
    return total, terms, current


@torch.no_grad()
def evaluate(model, rows, device):
    output = []
    for row in rows:
        sample = load_sample(row, device)
        started = time.perf_counter()
        displacement = model(sample["input"][None])[0]
        seconds = time.perf_counter() - started
        current = sample["tgt_mesh"] + displacement
        held = sample["held"]
        initial_error = sample["initial_canvas"]

        def mesh_error(mesh):
            local_warp = Warp(mesh, sample["origin"], sample["extent"])
            z_target, _, target_valid = local_warp.invert(sample["qstar_512"] / 256 - 1)
            error = ((sample["z_ref"] - z_target) * sample["extent"] / 2).norm(dim=-1)
            valid = target_valid & torch.isfinite(error)
            return torch.where(valid, error, torch.full_like(error, 512.0)), valid

        final_error, valid = mesh_error(current)
        strong_path = ROOT / "diagnostics_parallax_20260929/runs/unified_strong_baseline_review" \
                      / row["pair"] / "C_LBFGS_mesh.npy"
        if strong_path.exists():
            strong_mesh = torch.from_numpy(np.load(strong_path)).float().reshape(-1, 2).to(device)
            strong_error, strong_valid = mesh_error(strong_mesh)
        else:
            strong_error = strong_valid = None
        aligned = held & (initial_error <= 3.0)
        warp = Warp(current, sample["origin"], sample["extent"])
        base_det, _ = sampled(Warp(sample["tgt_mesh"], sample["origin"], sample["extent"]))
        final_det, sv = sampled(warp)
        area_ratio = triangles(current).abs().sum() / triangles(sample["tgt_mesh"]).abs().sum()
        overlap_uv = warp.at(sample["overlap_z"])
        retained = (overlap_uv.abs() <= 1).all(-1).float().mean()
        output.append({
            "split": row["split"], "scene": row["scene"], "pair": row["pair"],
            "held_n": int(held.sum()), "A_held_mean": float(initial_error[held].mean()),
            "B_LBFGS_held_mean": float(strong_error[held].mean()) if strong_error is not None else None,
            "C_plain_held_mean": float(final_error[held].mean()),
            "aligned_n": int(aligned.sum()),
            "A_aligned_mean": float(initial_error[aligned].mean()) if aligned.any() else None,
            "B_LBFGS_aligned_mean": (float(strong_error[aligned].mean())
                                      if strong_error is not None and aligned.any() else None),
            "C_plain_aligned_mean": float(final_error[aligned].mean()) if aligned.any() else None,
            "B_LBFGS_held_inversion_failure": (float((~strong_valid[held]).float().mean())
                                                if strong_valid is not None else None),
            "C_plain_held_inversion_failure": float((~valid[held]).float().mean()),
            "sampled_fold_fraction": float((final_det * torch.sign(base_det) <= 0).float().mean()),
            "anisotropy_p95": float(torch.quantile(sv[:, 0] / sv[:, 1].clamp_min(1e-8), .95)),
            "triangle_area_ratio": float(area_ratio), "overlap_sample_retention": float(retained),
            "max_displacement_px": float(displacement.norm(dim=-1).max()),
            "network_seconds": seconds,
        })
    return output


def aggregate(rows):
    total = sum(row["held_n"] for row in rows)
    strong_rows = [row for row in rows if row["B_LBFGS_held_mean"] is not None]
    strong_total = sum(row["held_n"] for row in strong_rows)
    return {"pairs": len(rows), "held_n": total,
            "A_held_mean": sum(row["A_held_mean"] * row["held_n"] for row in rows) / max(total, 1),
            "B_LBFGS_pairs": len(strong_rows),
            "B_LBFGS_held_mean": (sum(row["B_LBFGS_held_mean"] * row["held_n"] for row in strong_rows)
                                  / strong_total if strong_total else None),
            "C_plain_held_mean": sum(row["C_plain_held_mean"] * row["held_n"] for row in rows) / max(total, 1),
            "B_improved_pairs": sum(row["B_LBFGS_held_mean"] < row["A_held_mean"] for row in strong_rows),
            "C_improved_pairs": sum(row["C_plain_held_mean"] < row["A_held_mean"] for row in rows),
            "mean_overlap_retention": float(np.mean([row["overlap_sample_retention"] for row in rows])),
            "max_fold_fraction": max((row["sampled_fold_fraction"] for row in rows), default=0.0),
            "mean_network_seconds": float(np.mean([row["network_seconds"] for row in rows]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-splits", nargs="+", default=["train", "validation", "development_test"])
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--max-displacement", type=float, default=32.0)
    parser.add_argument("--train-limit", type=int, default=0,
                        help="Implementation-only overfit check; first N fixed training records")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    manifest, records = load_records(args.cache)
    if args.train_split not in records or not records[args.train_split]:
        raise ValueError(f"empty training split: {args.train_split}")
    train_rows = records[args.train_split][:args.train_limit] if args.train_limit else records[args.train_split]
    model = PlainResidualGridNet(args.base_channels, args.max_displacement).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weights = {"protection": 1.0, "anchor": .002, "smooth": .02,
               "triangle": 10.0, "tps": 100.0, "overlap": 10.0}
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        order = list(train_rows); random.shuffle(order)
        epoch_terms = {key: 0.0 for key in ("total", "geometry", "protection", "anchor", "smooth", "triangle", "tps", "overlap")}
        for row in order:
            sample = load_sample(row, device)
            optimizer.zero_grad(set_to_none=True)
            loss, terms, _ = losses(model, sample, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss on {row['pair']}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in terms.items(): epoch_terms[key] += float(value.detach())
        for key in epoch_terms: epoch_terms[key] /= len(order)
        epoch_terms["epoch"] = epoch + 1
        history.append(epoch_terms)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(json.dumps(epoch_terms), flush=True)
    train_seconds = time.perf_counter() - started
    torch.save({"model": model.state_dict(), "model_class": "PlainResidualGridNet",
                "base_channels": args.base_channels, "max_displacement": args.max_displacement,
                "input_channels": 11, "derived_local_correlation_channels": 3,
                "grid": [13, 13], "reference_fixed": True,
                "weights": weights, "seed": args.seed, "epochs": args.epochs},
               args.output / "plain_residual_model.pth")
    (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    all_rows = []
    model.eval()
    for split_name in args.eval_splits:
        if split_name in records:
            all_rows.extend(evaluate(model, records[split_name], device))
    with (args.output / "per_pair.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    summary = {
        "status": "complete", "development_only": True, "device": str(device),
        "parameters": parameter_count(model), "train_seconds": train_seconds,
        "epochs": args.epochs, "train_pairs": len(train_rows),
        "weights": weights, "splits": {name: aggregate([r for r in all_rows if r["split"] == name])
                                        for name in args.eval_splits if any(r["split"] == name for r in all_rows)},
        "limitations": ["training scenes were previously used for diagnosis; kicker validation was newly added and frozen",
                        "the terrace locked test was not evaluated because validation met the early-stop condition",
                        "network-only CPU timing is not a deployment GPU benchmark"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

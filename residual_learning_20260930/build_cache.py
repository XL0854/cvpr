"""Build RGB inputs and depth-derived supervision for the plain residual net."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diagnostics_parallax_20260929/scripts"))
sys.path.insert(0, str(ROOT / "diagnostics_geometry_20260915/scripts"))
from eth3d_geometry import (common_visible_correspondences, read_cameras, read_images,
                            read_raw_depth, scale_pixels)
from geometry import Warp, render


def geometry_path(scene: str, pair: str) -> Path:
    base = ROOT / "diagnostics_parallax_20260929/runs"
    if scene == "courtyard":
        short = pair.replace("DSC_", "")
        return base / f"rop_eth3d_courtyard_{short}_cpu/initial_geometry.npz"
    return base / f"rop_{scene}_final5_cpu/{pair}/initial_geometry.npz"


def resize_canvas(array: np.ndarray, size: int, is_mask: bool = False) -> np.ndarray:
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(array, (size, size), interpolation=interpolation)


def make_input(ref_rgb, tgt_rgb, ref_mask, tgt_mask, size):
    ref = resize_canvas(ref_rgb, size).astype(np.float32) / 127.5 - 1.0
    tgt = resize_canvas(tgt_rgb, size).astype(np.float32) / 127.5 - 1.0
    mr = resize_canvas(ref_mask.astype(np.uint8), size, True).astype(np.float32)[..., None]
    mt = resize_canvas(tgt_mask.astype(np.uint8), size, True).astype(np.float32)[..., None]
    features = np.concatenate((ref, tgt, np.abs(ref - tgt), mr, mt), axis=-1)
    return torch.from_numpy(features.transpose(2, 0, 1)).contiguous()


def build_pair(scene_name, scene_root, source_name, target_name, stride, input_size):
    pair = f"{Path(source_name).stem}_{Path(target_name).stem}"
    geom_path = geometry_path(scene_name, pair)
    if not geom_path.exists():
        raise FileNotFoundError(f"missing frozen RopStitch geometry: {geom_path}")
    geometry = np.load(geom_path)
    ref_mesh = torch.from_numpy(geometry["mesh_ref"]).float().reshape(-1, 2)
    tgt_mesh = torch.from_numpy(geometry["mesh_tgt"]).float().reshape(-1, 2)
    both = torch.cat((ref_mesh, tgt_mesh))
    origin, extent = both.min(0).values, both.max(0).values - both.min(0).values
    ref_warp, tgt_warp = Warp(ref_mesh, origin, extent), Warp(tgt_mesh, origin, extent)

    calibration = scene_root / "dslr_calibration_jpg"
    cameras, poses = read_cameras(calibration / "cameras.txt"), read_images(calibration / "images.txt")
    pose1, pose2 = poses[f"dslr_images/{source_name}"], poses[f"dslr_images/{target_name}"]
    camera1, camera2 = cameras[pose1.camera_id], cameras[pose2.camera_id]
    depth_root = scene_root / "ground_truth_depth/dslr_images"
    truth = common_visible_correspondences(
        camera1, pose1, read_raw_depth(depth_root / source_name, camera1),
        camera2, pose2, read_raw_depth(depth_root / target_name, camera2), stride=stride)
    valid_ids = np.flatnonzero(truth["evaluable"])
    p512 = torch.from_numpy(scale_pixels(truth["p"][valid_ids],
                                         (camera1.width, camera1.height), (512, 512))).float()
    q512 = torch.from_numpy(scale_pixels(truth["q_star"][valid_ids],
                                         (camera2.width, camera2.height), (512, 512))).float()
    with torch.no_grad():
        z_ref, _, inv_ref = ref_warp.invert(p512 / 256 - 1)
        z_tgt, _, inv_tgt = tgt_warp.invert(q512 / 256 - 1)
        valid = inv_ref & inv_tgt
        p512, q512, z_ref, z_tgt = p512[valid], q512[valid], z_ref[valid], z_tgt[valid]
        initial_forward = (tgt_warp.at(z_ref) - (q512 / 256 - 1)).norm(dim=-1) * 256
        initial_canvas = ((z_ref - z_tgt) * extent / 2).norm(dim=-1)
    cells = (p512[:, 0].floor_divide(64).clamp(0, 7)
             + 8 * p512[:, 1].floor_divide(64).clamp(0, 7)).long()
    held = (cells % 4) == 0

    image_root = scene_root / "images/dslr_images"
    images = []
    for name in (source_name, target_name):
        image = cv2.imread(str(image_root / name), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unreadable image: {image_root/name}")
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        images.append(torch.from_numpy(image.astype(np.float32).transpose(2, 0, 1))[None])
    output_size = (max(16, int(float(extent[1]))), max(16, int(float(extent[0]))))
    ref_rgb, ref_mask = render(ref_warp, images[0], output_size)
    tgt_rgb, tgt_mask = render(tgt_warp, images[1], output_size)
    network_input = make_input(ref_rgb, tgt_rgb, ref_mask, tgt_mask, input_size)

    # Fixed canvas samples that belong to the initial overlap.  The loss keeps
    # their target samples inside the target image without forcing zero motion.
    axis = torch.linspace(-1, 1, 33)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    canvas_z = torch.stack((xx, yy), -1).reshape(-1, 2)
    with torch.no_grad():
        ref_uv, tgt_uv = ref_warp.at(canvas_z), tgt_warp.at(canvas_z)
        overlap = (ref_uv.abs() <= .98).all(-1) & (tgt_uv.abs() <= .98).all(-1)

    return {
        "scene": scene_name, "pair": pair, "source": source_name, "target": target_name,
        "input": network_input, "ref_mesh": ref_mesh, "tgt_mesh": tgt_mesh,
        "origin": origin, "extent": extent, "z_ref": z_ref, "qstar_512": q512,
        "held": held, "initial_forward": initial_forward, "initial_canvas": initial_canvas,
        "overlap_z": canvas_z[overlap], "baseline_overlap_samples": int(overlap.sum()),
        "output_size": torch.tensor(output_size), "valid_truth_points": int(len(p512)),
        "truth_stride_native": int(stride),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--input-size", type=int, default=192)
    args = parser.parse_args()
    args.output = args.output.resolve()
    split = json.loads(args.split.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for split_name, scenes in split["splits"].items():
        for scene_spec in scenes:
            scene_name = scene_spec["scene"]
            scene_root = ROOT / scene_spec["root"]
            for source, target in scene_spec["pairs"]:
                sample = build_pair(scene_name, scene_root, source, target,
                                    args.stride, args.input_size)
                path = args.output / f"{scene_name}__{sample['pair']}.pt"
                torch.save(sample, path)
                records.append({"split": split_name, "scene": scene_name,
                                "pair": sample["pair"], "path": str(path.relative_to(ROOT)),
                                "truth_points": sample["valid_truth_points"],
                                "train_points": int((~sample["held"]).sum()),
                                "held_points": int(sample["held"].sum())})
                print("CACHED", split_name, scene_name, sample["pair"], records[-1], flush=True)
    manifest = {"status": "complete", "source_split": str(args.split),
                "truth_stride_native": args.stride, "input_size": args.input_size,
                "records": records}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

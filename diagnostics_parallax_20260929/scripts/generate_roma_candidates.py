"""Generate a fixed dense RoMa candidate set without residual/confidence filtering."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import bootstrap
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from eth3d_geometry import scale_pixels

OLD_DIAG = bootstrap.ROOT / "diagnostics_geometry_20260915"
sys.path.insert(0, str(OLD_DIAG / "third_party/RoMa"))
sys.path.insert(0, str(OLD_DIAG / "third_party/python_deps"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def sample(field, pixels_512):
    grid = 2*(pixels_512+.5)/512-1
    if field.ndim == 2:
        field = field[..., None]
    return F.grid_sample(field.permute(2, 0, 1)[None], grid[None, None],
                         align_corners=False)[0, :, 0].T


def query_unfiltered(warp, certainty, pixels_512):
    half = warp.shape[1]//2
    forward, backward = warp[:, :half, 2:], warp[:, half:, :2]
    forward_certainty, backward_certainty = certainty[:, :half], certainty[:, half:]
    target = (sample(forward, pixels_512)+1)*256-.5
    back = (sample(backward, target)+1)*256-.5
    score = sample(forward_certainty, pixels_512)[:, 0]
    reverse_score = sample(backward_certainty, target)[:, 0]
    cycle = (back-pixels_512).norm(dim=-1)
    finite = (torch.isfinite(target).all(-1) & torch.isfinite(cycle)
              & torch.isfinite(score) & torch.isfinite(reverse_score))
    inside = ((target >= -.5) & (target < 511.5)).all(-1)
    return target, score, reverse_score, cycle, finite & inside


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spacing", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; RoMa stage not run")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(20260930)
    torch.backends.cudnn.benchmark = False
    config = json.loads(args.manifest.read_text())
    scene = bootstrap.ROOT / config["scene"]
    image_root = scene / "images/dslr_images"
    weights_dir = OLD_DIAG / "cache/torch/hub/checkpoints"
    roma_path, dino_path = weights_dir/"roma_outdoor.pth", weights_dir/"dinov2_vitl14_pretrain.pth"
    from romatch import roma_outdoor
    model = roma_outdoor(
        device=device,
        weights=torch.load(roma_path, map_location="cpu", weights_only=True),
        dinov2_weights=torch.load(dino_path, map_location="cpu", weights_only=True),
        coarse_res=560, upsample_res=864, symmetric=True,
        use_custom_corr=False, do_compile=False)
    model.eval().requires_grad_(False)
    axis = torch.arange(args.spacing/2, 512, args.spacing, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    source_512 = torch.stack((xx, yy), -1).reshape(-1, 2)
    args.output.mkdir(parents=True, exist_ok=True)
    run_report = {
        "status": "running", "device": str(device), "spacing_512_px": args.spacing,
        "candidate_policy": "regular source grid; no confidence, cycle, geometry, or RopStitch-residual filtering",
        "model": {"name": "RoMa outdoor", "commit": "77f8d68803526dcddfd9b7a46bc76125bdc25f15",
                  "coarse_res": 560, "upsample_res": 864, "symmetric": True,
                  "use_custom_corr": False, "roma_sha256": digest(roma_path),
                  "dinov2_sha256": digest(dino_path)},
        "pairs": []
    }
    for source_name, target_name in config["pairs"]:
        pair_id = f"{Path(source_name).stem}_{Path(target_name).stem}"
        output = args.output/pair_id
        output.mkdir(parents=True, exist_ok=True)
        if (output/"candidates.npz").exists():
            run_report["pairs"].append({"id": pair_id, "status": "resume_skip"})
            print("RESUME_SKIP", pair_id, flush=True)
            continue
        images_bgr = [cv2.imread(str(image_root/name)) for name in (source_name, target_name)]
        if any(image is None for image in images_bgr):
            raise FileNotFoundError(pair_id)
        native_sizes = [(image.shape[1], image.shape[0]) for image in images_bgr]
        images = [Image.fromarray(cv2.cvtColor(cv2.resize(image, (512, 512),
                                                       interpolation=cv2.INTER_AREA),
                                               cv2.COLOR_BGR2RGB)) for image in images_bgr]
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.inference_mode():
            warp, certainty = model.match(*images, device=device)
            target_512, score, reverse_score, cycle, valid = query_unfiltered(
                warp[0], certainty[0], source_512)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter()-started
        source_np, target_np = source_512.cpu().numpy(), target_512.cpu().numpy()
        source_native = scale_pixels(source_np, (512, 512), native_sizes[0])
        target_native = scale_pixels(target_np, (512, 512), native_sizes[1])
        np.savez_compressed(output/"candidates.npz",
                            source_512=source_np, target_512=target_np,
                            source_native=source_native, target_native=target_native,
                            confidence=score.cpu().numpy(),
                            reverse_confidence=reverse_score.cpu().numpy(),
                            cycle_512_px=cycle.cpu().numpy(), valid=valid.cpu().numpy())
        row = {"id": pair_id, "source": source_name, "target": target_name,
               "candidates": len(source_np), "valid_prediction": int(valid.sum()),
               "seconds": seconds, "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))}
        (output/"report.json").write_text(json.dumps(row, indent=2)+"\n")
        run_report["pairs"].append(row)
        print("DONE", pair_id, json.dumps(row), flush=True)
    run_report["status"] = "completed"
    (args.output/"run_report.json").write_text(json.dumps(run_report, indent=2)+"\n")
    print("ROMA_CANDIDATES_COMPLETED", args.output)


if __name__ == "__main__":
    main()

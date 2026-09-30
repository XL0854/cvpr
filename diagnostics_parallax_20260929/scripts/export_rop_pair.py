"""Export one frozen RopStitch alpha=0.5 baseline pair for geometry diagnosis."""
import argparse
from contextlib import ExitStack
import json
import time
from pathlib import Path
from unittest.mock import patch

import bootstrap
import cv2
import numpy as np
import torch
import torchvision.models as models
import network as official


def load_models(device):
    constructor = models.resnet.resnet18
    with patch.object(models.resnet, "resnet18", side_effect=lambda *a, **k: constructor(weights=None)), \
         patch.object(torch.cuda, "is_available", return_value=False):
        net, coef = official.Network(), official.CoefNetwork()
    checkpoints = [bootstrap.ROOT / "woCoefNet/model_homo/epoch100_model.pth",
                   bootstrap.ROOT / "wCoefNet/model_coef/epoch050_coefmodel.pth"]
    for module, checkpoint in zip((net, coef), checkpoints):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        module.load_state_dict(state, strict=True)
        module.to(device).eval().requires_grad_(False)
    return net, coef


def load_image(path, device):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy((image.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1))[None]
    return image, tensor.to(device)


def save_rgb_like(path, array):
    cv2.imwrite(str(path), np.clip(array, 0, 255).round().astype(np.uint8))


def mesh_overlay(image, mesh, color=(0, 255, 0)):
    output = np.ascontiguousarray(image.copy())
    mesh = np.asarray(mesh)
    for row in range(mesh.shape[0]):
        for col in range(mesh.shape[1]):
            point = tuple(np.rint(mesh[row, col]).astype(int))
            if row + 1 < mesh.shape[0]:
                cv2.line(output, point, tuple(np.rint(mesh[row + 1, col]).astype(int)), color, 1, cv2.LINE_AA)
            if col + 1 < mesh.shape[1]:
                cv2.line(output, point, tuple(np.rint(mesh[row, col + 1]).astype(int)), color, 1, cv2.LINE_AA)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input1", type=Path, required=True)
    parser.add_argument("--input2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wants_cuda = str(args.device).startswith("cuda")
    if wants_cuda and not torch.cuda.is_available():
        failure = {
            "status": "not_run",
            "reason": "CUDA unavailable during torch.cuda.is_available()",
            "count_as_performance_result": False,
        }
        (args.output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise RuntimeError("CUDA unavailable; baseline export not run and must not be counted as a performance result")
    device = torch.device(args.device)
    if wants_cuda:
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    image1, input1 = load_image(args.input1, device)
    image2, input2 = load_image(args.input2, device)
    cv2.imwrite(str(args.output / "input1_512.png"), image1)
    cv2.imwrite(str(args.output / "input2_512.png"), image2)
    net, coef = load_models(device)
    versions = [parameter._version for module in (net, coef) for parameter in module.parameters()]
    if wants_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    # The original implementation creates helper tensors with hard-coded
    # ``.cuda()`` calls. For a CPU-only functional check, redirect only those
    # calls at runtime; the project source and model computation stay intact.
    with ExitStack() as stack:
        if not wants_cuda:
            stack.enter_context(patch.object(torch.Tensor, "cuda",
                                             lambda tensor, *a, **k: tensor))
            stack.enter_context(patch.object(torch.cuda, "is_available", return_value=False))
        with torch.inference_mode():
            _, _, coefficients, mesh_ref, mesh_tgt = net(input1, input2, coef, 0.5)
            rigid = official.get_rigid_mesh(1, 512, 512)
            rendered, valid = official.get_stitched_result(input1, input2, rigid, mesh_ref, mesh_tgt)
    if wants_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not valid:
        raise RuntimeError("RopStitch baseline returned an invalid/oversized canvas")
    assert versions == [parameter._version for module in (net, coef) for parameter in module.parameters()]

    both = torch.cat((mesh_ref.reshape(-1, 2), mesh_tgt.reshape(-1, 2)), 0)
    origin = both.min(0).values
    mesh_canvas_ref = mesh_ref - origin
    mesh_canvas_tgt = mesh_tgt - origin
    ref = rendered["output_ref"][0]
    tgt = rendered["output_tgt"][0]
    ref_image = (ref[:3].permute(1, 2, 0).cpu().numpy() * 127.5)
    tgt_image = (tgt[:3].permute(1, 2, 0).cpu().numpy() * 127.5)
    ref_mask = ref[3:].mean(0).cpu().numpy()
    tgt_mask = tgt[3:].mean(0).cpu().numpy()
    fusion = rendered["stitched"].permute(1, 2, 0).cpu().numpy() * 127.5
    save_rgb_like(args.output / "warp_ref.png", ref_image)
    save_rgb_like(args.output / "warp_tgt.png", tgt_image)
    save_rgb_like(args.output / "mask_ref.png", ref_mask * 255)
    save_rgb_like(args.output / "mask_tgt.png", tgt_mask * 255)
    save_rgb_like(args.output / "fusion.png", fusion)
    save_rgb_like(args.output / "mesh_ref.png", mesh_overlay(ref_image, mesh_canvas_ref[0].cpu().numpy()))
    save_rgb_like(args.output / "mesh_tgt.png", mesh_overlay(tgt_image, mesh_canvas_tgt[0].cpu().numpy()))
    np.savez_compressed(args.output / "initial_geometry.npz",
                        rigid_mesh=rigid[0].cpu().numpy(),
                        mesh_ref=mesh_ref[0].cpu().numpy(), mesh_tgt=mesh_tgt[0].cpu().numpy(),
                        mesh_canvas_ref=mesh_canvas_ref[0].cpu().numpy(),
                        mesh_canvas_tgt=mesh_canvas_tgt[0].cpu().numpy(),
                        canvas_origin=origin.cpu().numpy(),
                        final_coef=coefficients[-1][0].cpu().numpy())
    result = {
        "status": "completed", "alpha": 0.5, "input_size": [512, 512],
        "output_size": [int(ref.shape[2]), int(ref.shape[1])],
        "elapsed_seconds_cold_single_pair": elapsed,
        "device": str(device),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if wants_cuda else None,
        "network_frozen": True, "coef_network_frozen": True,
        "timing_note": ("single cold diagnostic pass; not a benchmark result" if wants_cuda else
                        "CPU functional diagnostic because CUDA was unavailable; do not use for performance comparison"),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("ROP_PAIR_EXPORT_COMPLETED", args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

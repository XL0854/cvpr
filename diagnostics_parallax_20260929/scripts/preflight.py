"""Record resources and blockers without initializing a CUDA context."""
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import bootstrap


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def command(args, timeout=8):
    try:
        result = subprocess.run(args, text=True, capture_output=True, timeout=timeout)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(), "timed_out": False}
    except subprocess.TimeoutExpired as error:
        return {"returncode": None, "stdout": (error.stdout or "").strip(),
                "stderr": (error.stderr or "").strip(), "timed_out": True}


def find_eth3d():
    roots = [Path("/workspace/python_pro"), bootstrap.ROOT / "data"]
    found = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("cameras.txt"):
            text = str(path).lower()
            if "eth3d" in text or "dslr_calibration" in text or "rig_calibration" in text:
                found.append(str(path))
    return sorted(set(found))


def main():
    root = bootstrap.ROOT
    weights = {
        "rop_network": root / "woCoefNet/model_homo/epoch100_model.pth",
        "rop_coef": root / "wCoefNet/model_coef/epoch050_coefmodel.pth",
        "roma": root / "diagnostics_geometry_20260915/cache/torch/hub/checkpoints/roma_outdoor.pth",
        "dinov2": root / "diagnostics_geometry_20260915/cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth",
    }
    report = {
        "status": "checking",
        "date": "2026-09-30",
        "python": platform.python_version(),
        "gpu_check": command(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.used,memory.total",
                              "--format=csv,noheader"]),
        "weights": {name: {"path": str(path), "exists": path.exists(),
                            "bytes": path.stat().st_size if path.exists() else None}
                    for name, path in weights.items()},
        "eth3d_calibrations_found": find_eth3d(),
        "matcher": {
            "name": "RoMa outdoor",
            "repository_commit": "77f8d68803526dcddfd9b7a46bc76125bdc25f15",
            "weights": "roma_outdoor.pth + dinov2_vitl14_pretrain.pth",
            "candidate_policy_for_new_study": "fixed candidate set from RoMa before any RopStitch-residual filtering; exact confidence sampling policy to be locked after GT visualization",
            "previous_diagnostic_policy_not_automatically_reused": "confidence>=0.5, reverse confidence>=0.5, cycle<=2px, 8x8 balancing belonged to the old study and may bias this hypothesis test",
        },
        "fixed_conditions": {"alpha": 0.5, "network_frozen": True, "coef_network_frozen": True,
                             "image_inputs_to_deployable_method": "two RGB images only"},
        "required_data": {
            "minimal_one_scene": {
                "scene": "courtyard",
                "archives": ["courtyard_dslr_jpg.7z", "courtyard_dslr_depth.7z"],
                "compressed_estimate_gb": 0.8,
                "purpose": "distorted RGB + matching distorted depth and THIN_PRISM_FISHEYE calibration",
            },
            "suggested_five_scene_development": {
                "scenes": ["courtyard", "delivery_area", "electro", "kicker", "terrace"],
                "compressed_estimate_gb": 4.0,
                "pair_target": "20-30 pairs across independent scenes; exact pairs selected only after GT coverage audit",
            },
            "official_page": "https://www.eth3d.net/datasets",
        },
        "source_sha256": {
            str(path.relative_to(root)): sha256(path)
            for path in [root / "wCoefNet/Codes/network.py",
                         root / "woCoefNet/model_homo/epoch100_model.pth",
                         root / "wCoefNet/model_coef/epoch050_coefmodel.pth"]
        },
    }
    has_eth3d = bool(report["eth3d_calibrations_found"])
    has_gpu = report["gpu_check"]["returncode"] == 0
    if has_eth3d and has_gpu:
        report["status"] = "resources_detected"
    elif has_eth3d:
        report["status"] = "eth3d_ready_gpu_unavailable"
    elif has_gpu:
        report["status"] = "gpu_ready_eth3d_missing"
    else:
        report["status"] = "eth3d_and_gpu_unavailable"
    output = bootstrap.DIAG / "runs" / "preflight" / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("PREFLIGHT_WRITTEN", output)
    print("STATUS", report["status"])


if __name__ == "__main__":
    main()

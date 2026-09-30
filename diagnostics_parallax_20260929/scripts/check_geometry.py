"""CPU-only numerical checks for pose direction, projection, scale and visibility."""
import json
import tempfile
from pathlib import Path

import bootstrap
import numpy as np
from eth3d_geometry import (Camera, ImagePose, camera_to_world,
                            common_visible_correspondences, project, project_pinhole,
                            qvec_to_rotation, read_cameras, read_images,
                            scale_pixels, unproject, unproject_pinhole,
                            world_to_camera)


def main():
    checks = {}
    camera = Camera(1, "PINHOLE", 64, 48, np.array([80., 82., 31.5, 23.5]))
    identity = ImagePose(1, 1, "a.png", np.array([1., 0., 0., 0.]), np.zeros(3))
    pixels = np.array([[4., 5.], [31.5, 23.5], [58., 40.]])
    depths = np.array([2., 4., 8.])
    points = unproject_pinhole(camera, pixels, depths)
    reprojection, recovered_depth, valid = project_pinhole(camera, points)
    checks["identity_reprojection_max_px"] = float(np.max(np.abs(reprojection - pixels)))
    checks["identity_depth_max"] = float(np.max(np.abs(recovered_depth - depths)))
    assert valid.all() and checks["identity_reprojection_max_px"] < 1e-10

    angle = np.deg2rad(7.)
    q = np.array([np.cos(angle/2), 0., np.sin(angle/2), 0.])
    pose = ImagePose(2, 1, "b.png", q, np.array([-0.25, 0.03, 0.1]))
    world = camera_to_world(pose, points)
    roundtrip = world_to_camera(pose, world)
    checks["pose_roundtrip_max"] = float(np.max(np.abs(roundtrip - points)))
    checks["rotation_orthogonality_max"] = float(np.max(np.abs(qvec_to_rotation(q).T @ qvec_to_rotation(q) - np.eye(3))))
    assert checks["pose_roundtrip_max"] < 1e-12

    resized = scale_pixels(pixels, (64, 48), (512, 512))
    restored = scale_pixels(resized, (512, 512), (64, 48))
    checks["resize_roundtrip_max_px"] = float(np.max(np.abs(restored - pixels)))
    assert checks["resize_roundtrip_max_px"] < 1e-12

    depth = np.full((48, 64), 4., np.float32)
    visibility = common_visible_correspondences(camera, identity, depth, camera, identity, depth,
                                                stride=4, absolute_depth_tolerance=1e-6,
                                                relative_depth_tolerance=1e-6)
    checks["identity_visibility_evaluable"] = int(visibility["evaluable"].sum())
    checks["identity_visibility_total"] = int(len(visibility["evaluable"]))
    assert visibility["evaluable"].all()

    missing = depth.copy(); missing[20:28, 28:36] = np.inf
    visibility_missing = common_visible_correspondences(camera, identity, depth, camera, identity, missing,
                                                        stride=4, absolute_depth_tolerance=1e-6,
                                                        relative_depth_tolerance=1e-6)
    labels, counts = np.unique(visibility_missing["label"], return_counts=True)
    checks["missing_depth_labels"] = {str(k): int(v) for k, v in zip(labels, counts)}
    assert checks["missing_depth_labels"].get("target_depth_missing", 0) > 0

    distorted = Camera(2, "THIN_PRISM_FISHEYE", 6048, 4032, np.array([
        3411.42, 3410.02, 3041.29, 2014.07, 0.21047, 0.21102,
        -5.36231e-06, 0.00051541, -0.158023, 0.406856,
        -8.46499e-05, 0.000861313]))
    distorted_pixels = np.array([[0., 0.], [3040.79, 2013.57],
                                 [6047., 4031.], [1100.25, 3200.75]])
    distorted_depths = np.array([2., 4., 8., 3.])
    distorted_points = unproject(distorted, distorted_pixels, distorted_depths)
    distorted_reprojection, distorted_recovered_depth, distorted_valid = project(
        distorted, distorted_points)
    checks["thin_prism_roundtrip_max_px"] = float(
        np.max(np.abs(distorted_reprojection - distorted_pixels)))
    checks["thin_prism_depth_max"] = float(
        np.max(np.abs(distorted_recovered_depth - distorted_depths)))
    checks["thin_prism_pixel_convention_principal_point"] = [3040.79, 2013.57]
    assert distorted_valid.all() and checks["thin_prism_roundtrip_max_px"] < 1e-4

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        (directory / "cameras.txt").write_text("# test\n1 PINHOLE 64 48 80 82 31.5 23.5\n")
        (directory / "images.txt").write_text(
            "# test\n1 1 0 0 0 0 0 0 1 a.png\n10 10 -1\n"
            "2 1 0 0 0 -0.25 0.03 0.1 1 b.png\n\n")
        checks["parsed_cameras"] = len(read_cameras(directory / "cameras.txt"))
        checks["parsed_images"] = len(read_images(directory / "images.txt"))
        assert checks["parsed_cameras"] == 1 and checks["parsed_images"] == 2

    output = bootstrap.DIAG / "runs" / "geometry_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "passed", "checks": checks}, indent=2) + "\n")
    print("GEOMETRY_CHECKS_PASSED", output)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

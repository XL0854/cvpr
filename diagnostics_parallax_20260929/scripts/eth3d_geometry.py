"""Strict ETH3D/COLMAP geometry helpers used only for evaluation.

The THIN_PRISM_FISHEYE equations follow ETH3D's official implementation in
``third_party/camera-model-implementations``.  ETH3D calibration coordinates
put (0, 0) at the top-left image corner, while NumPy array coordinates below
put (0, 0) at the centre of the top-left pixel.  Thus cx and cy are shifted by
-0.5 exactly as prescribed by the official implementation README.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True)
class ImagePose:
    image_id: int
    camera_id: int
    name: str
    qvec: np.ndarray
    tvec: np.ndarray

    @property
    def rotation_world_to_camera(self):
        return qvec_to_rotation(self.qvec)


def _data_lines(path):
    return [line.strip() for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def read_cameras(path):
    cameras = {}
    for line in _data_lines(path):
        fields = line.split()
        camera = Camera(int(fields[0]), fields[1], int(fields[2]), int(fields[3]),
                        np.asarray(fields[4:], dtype=np.float64))
        expected = {"PINHOLE": 4, "THIN_PRISM_FISHEYE": 12}.get(camera.model)
        if expected is None:
            raise ValueError(f"Unsupported camera model: {camera.model}")
        if len(camera.params) != expected:
            raise ValueError(f"{camera.model} expects {expected} parameters, got {len(camera.params)}")
        cameras[camera.camera_id] = camera
    return cameras


def read_images(path):
    """Read COLMAP images.txt, whose records occupy two lines."""
    lines = Path(path).read_text().splitlines()
    images = {}
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"Invalid image pose line: {line}")
        pose = ImagePose(
            image_id=int(fields[0]),
            qvec=np.asarray(fields[1:5], dtype=np.float64),
            tvec=np.asarray(fields[5:8], dtype=np.float64),
            camera_id=int(fields[8]),
            name=" ".join(fields[9:]),
        )
        images[pose.name] = pose
        index += 1
        # The following line is the non-ground-truth feature observation list.
        # It may be empty but is still structurally present in COLMAP text format.
        if index < len(lines):
            index += 1
    return images


def qvec_to_rotation(qvec):
    q = np.asarray(qvec, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def read_raw_depth(path, camera):
    values = np.fromfile(path, dtype="<f4")
    expected = camera.width * camera.height
    if values.size != expected:
        raise ValueError(f"Depth size mismatch: {values.size} floats, expected {expected}")
    return values.reshape(camera.height, camera.width)


def _pixel_intrinsics(camera):
    fx, fy, cx, cy = camera.params[:4]
    if camera.model == "THIN_PRISM_FISHEYE":
        cx -= 0.5
        cy -= 0.5
    return fx, fy, cx, cy


def unproject_pinhole(camera, pixels, depth):
    if camera.model != "PINHOLE":
        raise ValueError(f"Expected PINHOLE, got {camera.model}")
    pixels = np.asarray(pixels, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    fx, fy, cx, cy = camera.params
    x = (pixels[:, 0] - cx) * depth / fx
    y = (pixels[:, 1] - cy) * depth / fy
    return np.column_stack((x, y, depth))


def project_pinhole(camera, points_camera):
    if camera.model != "PINHOLE":
        raise ValueError(f"Expected PINHOLE, got {camera.model}")
    points = np.asarray(points_camera, dtype=np.float64)
    fx, fy, cx, cy = camera.params
    z = points[:, 2]
    pixels = np.column_stack((fx * points[:, 0] / z + cx,
                              fy * points[:, 1] / z + cy))
    valid = np.isfinite(pixels).all(axis=1) & (z > 0)
    valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width)
    valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height)
    return pixels, z, valid


def _thin_prism_polynomial(camera, xy):
    """Polynomial/tangential part of ETH3D's official camera model."""
    xy = np.asarray(xy, dtype=np.float64)
    x, y = xy[:, 0], xy[:, 1]
    k1, k2, p1, p2, k3, k4, sx1, sy1 = camera.params[4:]
    x2, xy_product, y2 = x*x, x*y, y*y
    r2 = x2 + y2
    radial = k1*r2 + k2*r2**2 + k3*r2**3 + k4*r2**4
    dx = 2*p1*xy_product + p2*(r2 + 2*x2) + sx1*r2
    dy = 2*p2*xy_product + p1*(r2 + 2*y2) + sy1*r2
    return np.column_stack((x + radial*x + dx, y + radial*y + dy))


def _thin_prism_polynomial_jacobian(camera, xy):
    """Return J as [d(out_x)/dx, d(out_x)/dy; d(out_y)/dx, d(out_y)/dy]."""
    x, y = np.asarray(xy, dtype=np.float64).T
    k1, k2, p1, p2, k3, k4, sx1, sy1 = camera.params[4:]
    x2, xy_product, y2 = x*x, x*y, y*y
    r2, r4, r6, r8 = x2 + y2, (x2+y2)**2, (x2+y2)**3, (x2+y2)**4
    term = (2*p1*x + 2*p2*y + 2*k1*xy_product + 4*k2*xy_product*r2
            + 6*k3*xy_product*r4 + 8*k4*xy_product*r6)
    j00 = (1 + k1*r2 + k2*r4 + k3*r6 + k4*r8 + 2*k1*x2
           + 4*k2*x2*r2 + 6*k3*x2*r4 + 8*k4*x2*r6
           + 6*p2*x + 2*p1*y + 2*sx1*x)
    j01 = term + 2*sx1*y
    j10 = term + 2*sy1*x
    j11 = (1 + k1*r2 + k2*r4 + k3*r6 + k4*r8 + 2*k1*y2
           + 4*k2*y2*r2 + 6*k3*y2*r4 + 8*k4*y2*r6
           + 2*p2*x + 6*p1*y + 2*sy1*y)
    return np.stack((j00, j01, j10, j11), axis=1).reshape(-1, 2, 2)


def project_thin_prism(camera, points_camera):
    if camera.model != "THIN_PRISM_FISHEYE":
        raise ValueError(f"Expected THIN_PRISM_FISHEYE, got {camera.model}")
    points = np.asarray(points_camera, dtype=np.float64)
    z = points[:, 2]
    normalized = points[:, :2] / z[:, None]
    radius = np.linalg.norm(normalized, axis=1)
    fisheye = normalized.copy()
    nonzero = radius > 1e-12
    fisheye[nonzero] *= (np.arctan(radius[nonzero]) / radius[nonzero])[:, None]
    distorted = _thin_prism_polynomial(camera, fisheye)
    fx, fy, cx, cy = _pixel_intrinsics(camera)
    pixels = distorted * np.asarray([fx, fy]) + np.asarray([cx, cy])
    valid = np.isfinite(pixels).all(axis=1) & (z > 0)
    valid &= (pixels[:, 0] >= -0.5) & (pixels[:, 0] < camera.width - 0.5)
    valid &= (pixels[:, 1] >= -0.5) & (pixels[:, 1] < camera.height - 0.5)
    return pixels, z, valid


def unproject_thin_prism(camera, pixels, depth, max_iterations=100):
    if camera.model != "THIN_PRISM_FISHEYE":
        raise ValueError(f"Expected THIN_PRISM_FISHEYE, got {camera.model}")
    pixels = np.asarray(pixels, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    fx, fy, cx, cy = _pixel_intrinsics(camera)
    target = (pixels - np.asarray([cx, cy])) / np.asarray([fx, fy])
    estimate = target.copy()
    active = np.ones(len(estimate), dtype=bool)
    for _ in range(max_iterations):
        if not active.any():
            break
        ids = np.flatnonzero(active)
        residual = _thin_prism_polynomial(camera, estimate[ids]) - target[ids]
        jacobian = _thin_prism_polynomial_jacobian(camera, estimate[ids])
        determinant = jacobian[:, 0, 0]*jacobian[:, 1, 1] - jacobian[:, 0, 1]*jacobian[:, 1, 0]
        safe = np.abs(determinant) > 1e-15
        step = np.zeros_like(residual)
        step[safe, 0] = (jacobian[safe, 1, 1]*residual[safe, 0]
                         - jacobian[safe, 0, 1]*residual[safe, 1]) / determinant[safe]
        step[safe, 1] = (-jacobian[safe, 1, 0]*residual[safe, 0]
                         + jacobian[safe, 0, 0]*residual[safe, 1]) / determinant[safe]
        estimate[ids] -= step
        converged = np.sum(residual*residual, axis=1) < 1e-10
        active[ids[converged | ~safe]] = False
    theta = np.linalg.norm(estimate, axis=1)
    scale = np.ones_like(theta)
    denominator = theta*np.cos(theta)
    nonzero = denominator > 1e-12
    scale[nonzero] = np.sin(theta[nonzero]) / denominator[nonzero]
    normalized = estimate * scale[:, None]
    return np.column_stack((normalized*depth[:, None], depth))


def unproject(camera, pixels, depth):
    if camera.model == "PINHOLE":
        return unproject_pinhole(camera, pixels, depth)
    return unproject_thin_prism(camera, pixels, depth)


def project(camera, points_camera):
    if camera.model == "PINHOLE":
        return project_pinhole(camera, points_camera)
    return project_thin_prism(camera, points_camera)


def camera_to_world(pose, points_camera):
    rotation = pose.rotation_world_to_camera
    return (rotation.T @ (np.asarray(points_camera) - pose.tvec).T).T


def world_to_camera(pose, points_world):
    return (pose.rotation_world_to_camera @ np.asarray(points_world).T).T + pose.tvec


def scale_pixels(pixels, source_size, target_size):
    """OpenCV-style pixel-centre resize mapping, sizes are (width, height)."""
    pixels = np.asarray(pixels, dtype=np.float64)
    scale = np.asarray(target_size, dtype=np.float64) / np.asarray(source_size, dtype=np.float64)
    return (pixels + 0.5) * scale - 0.5


def correspondences_at_pixels(camera1, pose1, depth1, camera2, pose2, depth2, pixels,
                              absolute_depth_tolerance=0.02,
                              relative_depth_tolerance=0.01,
                              boundary_relative_range=0.03):
    """Evaluate source pixels with explicit missing/occlusion labels."""
    if depth1.shape != (camera1.height, camera1.width):
        raise ValueError("depth1 shape does not match camera1")
    if depth2.shape != (camera2.height, camera2.width):
        raise ValueError("depth2 shape does not match camera2")
    p = np.asarray(pixels, dtype=np.float64)
    rounded = np.rint(p).astype(int)
    source_inside = ((rounded[:, 0] >= 1) & (rounded[:, 0] < camera1.width-1)
                     & (rounded[:, 1] >= 1) & (rounded[:, 1] < camera1.height-1))
    d1 = np.full(len(p), np.nan, dtype=np.float64)
    inside_ids = np.flatnonzero(source_inside)
    d1[inside_ids] = depth1[rounded[inside_ids, 1], rounded[inside_ids, 0]]
    source_valid = source_inside & np.isfinite(d1) & (d1 > 0)

    boundary = np.zeros(len(p), dtype=bool)
    for i in np.flatnonzero(source_valid):
        x, y = rounded[i]
        patch = depth1[y-1:y+2, x-1:x+2]
        finite = patch[np.isfinite(patch) & (patch > 0)]
        # ETH3D laser-scan depth is sparse. Missing neighbours are unknown,
        # rather than evidence that a valid centre sample lies on a boundary.
        # Mark uncertainty only when observed neighbours demonstrate a jump.
        if finite.size >= 2 and (finite.max() - finite.min()) > boundary_relative_range * max(d1[i], 1e-12):
            boundary[i] = True

    points1 = unproject(camera1, p[source_valid], d1[source_valid])
    world = camera_to_world(pose1, points1)
    points2 = world_to_camera(pose2, world)
    q_valid, predicted_depth, projected = project(camera2, points2)

    q = np.full((len(p), 2), np.nan, dtype=np.float64)
    z2 = np.full(len(p), np.nan, dtype=np.float64)
    projection_valid = np.zeros(len(p), dtype=bool)
    source_indices = np.flatnonzero(source_valid)
    q[source_indices] = q_valid
    z2[source_indices] = predicted_depth
    projection_valid[source_indices] = projected

    q_round = np.rint(np.nan_to_num(q, nan=-1)).astype(int)
    target_depth = np.full(len(p), np.nan, dtype=np.float64)
    inside = projection_valid.copy()
    ids = np.flatnonzero(inside)
    target_depth[ids] = depth2[q_round[ids, 1], q_round[ids, 0]]
    target_valid = np.isfinite(target_depth) & (target_depth > 0)
    tolerance = absolute_depth_tolerance + relative_depth_tolerance * np.abs(z2)
    consistent = target_valid & (np.abs(target_depth - z2) <= tolerance)

    label = np.full(len(p), "invalid_source_depth", dtype=object)
    label[source_valid & boundary] = "uncertain_source_boundary"
    label[source_valid & ~boundary & ~projection_valid] = "outside_or_behind_target"
    label[source_valid & ~boundary & projection_valid & ~target_valid] = "target_depth_missing"
    conflict = source_valid & ~boundary & projection_valid & target_valid & ~consistent
    label[conflict & (target_depth < z2 - tolerance)] = "occluded_in_target"
    label[conflict & ~(target_depth < z2 - tolerance)] = "depth_inconsistent"
    evaluable = source_valid & ~boundary & projection_valid & consistent
    label[evaluable] = "evaluable"
    return {"p": p, "q_star": q, "source_depth": d1, "target_predicted_depth": z2,
            "target_observed_depth": target_depth, "evaluable": evaluable, "label": label}


def common_visible_correspondences(camera1, pose1, depth1, camera2, pose2, depth2,
                                   stride=8, absolute_depth_tolerance=0.02,
                                   relative_depth_tolerance=0.01,
                                   boundary_relative_range=0.03):
    """Build regularly sampled common-visible correspondences."""
    ys = np.arange(1, camera1.height - 1, stride)
    xs = np.arange(1, camera1.width - 1, stride)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    pixels = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
    return correspondences_at_pixels(
        camera1, pose1, depth1, camera2, pose2, depth2, pixels,
        absolute_depth_tolerance, relative_depth_tolerance,
        boundary_relative_range)

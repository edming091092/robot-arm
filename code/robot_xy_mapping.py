"""Planar camera-to-MG400 mapping. Math and strictly read-only robot sampling.

Pixel convention: undistorted pixels, original image size and original K.
No crop, resize or getOptimalNewCameraMatrix is used in the mapping.
"""
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from mg400_tcp import Channel, Feedback


def finite(values, count):
    try:
        array = np.asarray(values, np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Coordinates must be numbers") from exc
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f"Expected {count} finite coordinates")
    return array


def load_intrinsics(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    if data.get("calibration_type") != "camera_intrinsics_only":
        raise ValueError("Select camera_calibration.json from the camera calibration program")
    matrix = np.asarray(data.get("camera_matrix"), np.float64)
    distortion = np.asarray(data.get("distortion_coefficients"), np.float64).ravel()
    size = finite(data.get("image_size"), 2)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[0, 0] <= 0
            or matrix[1, 1] <= 0 or abs(np.linalg.det(matrix)) < 1e-12
            or len(distortion) not in (4, 5, 8, 12, 14) or not np.isfinite(distortion).all()
            or any(v <= 0 or int(v) != v for v in size)):
        raise ValueError("Invalid camera matrix, distortion coefficients or image size")
    return data, hashlib.sha256(raw).hexdigest()


def undistort_pixel(uv, intrinsics):
    uv = finite(uv, 2)
    width, height = intrinsics["image_size"]
    if not (0 <= uv[0] < width and 0 <= uv[1] < height):
        raise ValueError("Raw pixel is outside the calibrated image")
    matrix = np.asarray(intrinsics["camera_matrix"], np.float64)
    distortion = np.asarray(intrinsics["distortion_coefficients"], np.float64)
    result = cv2.undistortPoints(uv.reshape(1, 1, 2), matrix, distortion, P=matrix)
    return result.reshape(2)


def project(matrix, points):
    points = np.asarray(points, np.float64).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ np.asarray(matrix).T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-10):
        raise ValueError("Mapping has an invalid projective denominator")
    return homogeneous[:, :2] / homogeneous[:, 2:]


def hull(points):
    return cv2.convexHull(np.asarray(points, np.float32).reshape(-1, 1, 2))


def inside(points_hull, point):
    return cv2.pointPolygonTest(np.asarray(points_hull, np.float32).reshape(-1, 1, 2),
                                tuple(map(float, point)), False) >= 0


def check_geometry(points, name):
    points = np.asarray(points, np.float64)
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    if singular[0] <= 1e-8 or singular[1] / singular[0] < 0.02:
        raise ValueError(f"{name}: points are collinear or too narrowly distributed")
    if cv2.contourArea(hull(points)) < 1:
        raise ValueError(f"{name}: covered area is too small")


def validate_sample(sample, image_size):
    if sample.get("role") not in ("fit", "validation"):
        raise ValueError("Point role must be fit or validation")
    uv = finite([sample["u"], sample["v"]], 2)
    finite([sample[a] for a in ("x", "y", "z", "r")], 4)
    if not (0 <= uv[0] < image_size[0] and 0 <= uv[1] < image_size[1]):
        raise ValueError("Point is outside the undistorted image")


def pose_consistency_issues(samples, plane_tolerance_mm=1.0, r_tolerance_deg=1.0):
    if not samples:
        return []
    issues = []
    heights = [p["z"] for p in samples]
    if max(heights) - min(heights) > plane_tolerance_mm:
        issues.append("Points are not at the same TCP Z; use one fixed plane")
    reference_r = samples[0]["r"]
    if any(abs((p["r"] - reference_r + 180) % 360 - 180) > r_tolerance_deg for p in samples):
        issues.append("Tool orientation R changed; collect with a fixed orientation")
    return issues


def fit_mapping(samples, intrinsics, intrinsic_sha256, user, tool, max_error_mm=2.0,
                min_fit=9, min_validation=3, plane_tolerance_mm=1.0, r_tolerance_deg=1.0,
                preview_only=False):
    if not math.isfinite(max_error_mm) or max_error_mm <= 0:
        raise ValueError("Error threshold must be positive")
    if user != 0 or min_fit < 4 or (min_validation != 0 and min_validation < 3):
        raise ValueError("Use User 0, at least 4 fit points and 3 independent validation points")
    for sample in samples:
        validate_sample(sample, intrinsics["image_size"])
    for i, sample in enumerate(samples):
        for previous in samples[:i]:
            if (math.hypot(sample["u"] - previous["u"], sample["v"] - previous["v"]) < 2
                    or math.hypot(sample["x"] - previous["x"], sample["y"] - previous["y"]) < 0.5):
                raise ValueError("Duplicate points cannot be reused as independent validation")
    fit = [p for p in samples if p["role"] == "fit"]
    validation = [p for p in samples if p["role"] == "validation"]
    if len(fit) < min_fit or len(validation) < min_validation:
        raise ValueError(f"Need {min_fit} fit and {min_validation} separate validation points; "
                         f"currently {len(fit)} and {len(validation)}")
    z_values = [p["z"] for p in samples]
    reference_r = samples[0]["r"]
    issues = pose_consistency_issues(samples, plane_tolerance_mm, r_tolerance_deg)
    if issues and not preview_only:
        raise ValueError("; ".join(issues))
    source = np.array([[p["u"], p["v"]] for p in fit], np.float64)
    target = np.array([[p["x"], p["y"]] for p in fit], np.float64)
    check_geometry(source, "Image points")
    check_geometry(target, "Robot points")
    source_hull = hull(source)
    validation_uv = np.array([[p["u"], p["v"]] for p in validation], np.float64).reshape(-1, 2)
    if validation:
        if len(validation) < 3:
            raise ValueError("Use either zero or at least 3 validation points")
        check_geometry(validation_uv, "Validation image points")
    if not all(inside(source_hull, p) for p in validation_uv):
        raise ValueError("Validation points must lie inside the fit-point boundary")
    matrix, mask = cv2.findHomography(source, target, 0 if preview_only else cv2.RANSAC, max_error_mm)
    if matrix is None or mask is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("Homography estimation failed")
    all_inliers = bool(np.all(mask))
    if all_inliers:
        matrix, _ = cv2.findHomography(source, target, 0)
    if matrix is None or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("Homography refinement failed")
    boundary = source_hull.reshape(-1, 2)
    denominators = np.column_stack([boundary, np.ones(len(boundary))]) @ matrix[2]
    if not (np.all(denominators > 1e-10) or np.all(denominators < -1e-10)):
        raise ValueError("Projective horizon crosses the calibrated region")
    fit_errors = np.linalg.norm(project(matrix, source) - target, axis=1)
    expected = np.array([[p["x"], p["y"]] for p in validation], np.float64).reshape(-1, 2)
    predicted = project(matrix, validation_uv)
    validation_errors = np.linalg.norm(predicted - expected, axis=1)
    if preview_only:
        all_inliers = bool(np.all(fit_errors <= max_error_mm))
    fit_passed = bool(not issues and all_inliers and max(fit_errors) <= max_error_mm)
    independently_validated = len(validation) >= 3
    passed = bool(not preview_only and fit_passed and independently_validated
                  and max(validation_errors) <= max_error_mm)
    return {
        "schema_version": 1, "calibration_type": "planar_camera_to_robot_xy", "passed": passed,
        "preview_only": preview_only, "fit_passed": fit_passed,
        "independent_validation_performed": independently_validated,
        "fit_method": "all_points_least_squares" if preview_only else "ransac_then_refine",
        "quality_issues": issues,
        "pixel_space": "undistorted_pixels_original_K_no_crop", "H": matrix.tolist(),
        "image_size": intrinsics["image_size"], "camera_matrix": intrinsics["camera_matrix"],
        "distortion_coefficients": intrinsics["distortion_coefficients"],
        "intrinsics_sha256": intrinsic_sha256, "user_index": user, "tool_index": tool,
        "plane_tcp_z_mm": float(np.mean(z_values)), "reference_r_deg": reference_r,
        "max_error_threshold_mm": max_error_mm, "all_fit_points_are_inliers": all_inliers,
        "fit_count": len(fit), "validation_count": len(validation),
        "fit_errors_mm": fit_errors.tolist(), "validation_errors_mm": validation_errors.tolist(),
        "max_fit_error_mm": float(max(fit_errors)),
        "rms_fit_error_mm": float(np.sqrt(np.mean(fit_errors ** 2))),
        "fit_predictions_xy": project(matrix, source).tolist(),
        "max_validation_error_mm": float(max(validation_errors)) if validation else None,
        "rms_validation_error_mm": float(np.sqrt(np.mean(validation_errors ** 2))) if validation else None,
        "validation_predictions_xy": predicted.tolist(),
        "valid_pixel_hull": boundary.tolist(), "valid_robot_xy_hull": hull(target).reshape(-1, 2).tolist(),
        "samples": samples,
        "notes": ["Validation points are excluded from fitting.",
                  "No robot motion or automatic Z selection is performed.",
                  "Only valid at the calibrated plane, camera mount, focus, resolution, User and Tool."],
    }


def pixel_to_robot(uv, mapping, pixel_space="raw", allow_preview=False):
    if mapping.get("passed") is not True and not (allow_preview and mapping.get("preview_only") is True):
        raise ValueError("Mapping has not passed independent validation")
    if mapping.get("pixel_space") != "undistorted_pixels_original_K_no_crop":
        raise ValueError("Unsupported mapping pixel convention")
    point = undistort_pixel(uv, mapping) if pixel_space == "raw" else finite(uv, 2)
    if pixel_space not in ("raw", "undistorted"):
        raise ValueError("pixel_space must be raw or undistorted")
    if not inside(mapping["valid_pixel_hull"], point):
        raise ValueError("Point is outside the calibrated region; extrapolation refused")
    return project(mapping["H"], [point])[0]


def read_robot_pose(ip, user, tool):
    """Only RobotMode() and GetPose() are sent, plus read-only feedback."""
    dashboard, feedback = Channel(ip, 29999), Feedback(ip)
    try:
        def check_state():
            state = feedback.read(timeout=3, require_new=True)
            mode = int(dashboard.request("RobotMode()"))
            if (state["robot_type"] != 1 or state["error"] or mode not in (4, 5, 10)
                    or state["mode"] not in (4, 5, 10)):
                raise ValueError("MG400 must be stationary and without alarms; release the jog button")
            if state["user"] != user or state["tool"] != tool:
                raise ValueError(f"Select User {user} and Tool {tool} in the official software first. "
                                 f"Feedback currently reports User {state['user']}, Tool {state['tool']}")
            return state
        def pose():
            values = [float(v) for v in dashboard.request("GetPose()").split(",")]
            if len(values) not in (4, 6) or not all(math.isfinite(v) for v in values):
                raise ValueError("Invalid GetPose response")
            return values[:4]
        check_state()
        first = pose()
        time.sleep(0.2)
        second = pose()
        check_state()
        if (math.dist(first[:3], second[:3]) > 0.1
                or abs((first[3] - second[3] + 180) % 360 - 180) > 0.1):
            raise ValueError("Robot is still moving; wait and read again")
        return second
    finally:
        dashboard.close()
        feedback.close()


def save_samples(path, samples):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "role", "u", "v", "x", "y", "z", "r"])
        writer.writeheader()
        writer.writerows(samples)


def load_samples(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [{"id": row["id"], "role": row["role"],
             **{a: float(row[a]) for a in ("u", "v", "x", "y", "z", "r")}} for row in rows]

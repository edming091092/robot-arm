"""Camera intrinsics with the supplied 9x6-inner-corner chessboard.
Default: open camera and capture. SPACE saves, C calibrates, Q exits.
This program does not import a robot driver or send robot commands.
"""
import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit("Missing dependencies. Run: python -m pip install opencv-python numpy") from exc

ROOT = Path(__file__).resolve().parents[1]
PATTERN = (9, 6)  # INTERNAL corners, not the 10x7 square count.


def load_image(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def detect(image, accurate=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if accurate and hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, PATTERN, cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            return corners.astype(np.float32).reshape(-1, 1, 2)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if not accurate:
        flags |= cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, PATTERN, flags)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)).reshape(-1, 1, 2)


def same_view(corners, previous, image_size):
    """Reject near-duplicate frames, including reversed corner ordering."""
    diagonal = math.hypot(*image_size)
    for old in previous:
        distances = [float(np.sqrt(np.mean(np.sum((corners - version) ** 2, axis=2))))
                     for version in (old, old[::-1])]
        if min(distances) / diagonal < 0.008:
            return True
    return False


def object_points(square_mm):
    points = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    points[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * square_mm
    return points


def solve_calibration(image_points, image_size, square_mm, max_rms=1.0):
    if len(image_points) < 10:
        raise ValueError("At least 10 different valid views are required; 15-25 are recommended")
    obj = object_points(square_mm)
    objects = [obj.copy() for _ in image_points]
    rms, matrix, distortion, rotations, translations = cv2.calibrateCamera(
        objects, image_points, image_size, None, None)
    if (not np.isfinite(matrix).all() or not np.isfinite(distortion).all()
            or not math.isfinite(rms) or matrix[0, 0] <= 0 or matrix[1, 1] <= 0):
        raise ValueError("Calibration returned invalid parameters; capture more varied board poses")
    errors = []
    normals = []
    for corners, rotation, translation in zip(image_points, rotations, translations):
        projected, _ = cv2.projectPoints(obj, rotation, translation, matrix, distortion)
        projected = projected.reshape(-1, 1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum((corners - projected) ** 2, axis=2)))))
        normal = cv2.Rodrigues(rotation)[0][:, 2]
        if normal[2] < 0:
            normal = -normal
        normals.append(normal)
    normal_span = max(math.degrees(math.acos(float(np.clip(np.dot(a, b), -1, 1))))
                      for a in normals for b in normals)
    warnings = []
    if rms > max_rms:
        warnings.append(f"RMS {rms:.3f} px exceeds review threshold {max_rms:.3f} px")
    if max(errors) > 2 * max_rms:
        warnings.append("Some individual images have high reprojection error; inspect the CSV report")
    if normal_span < 10:
        warnings.append("Insufficient board tilt diversity; capture multiple visibly tilted poses")
    if not (0 <= matrix[0, 2] < image_size[0] and 0 <= matrix[1, 2] < image_size[1]):
        warnings.append("Principal point is outside the image; inspect camera model and capture diversity")
    return {
        "schema_version": 1,
        "calibration_type": "camera_intrinsics_only",
        "camera_model": "opencv_pinhole_5_coefficient",
        "opencv_version": cv2.__version__,
        "image_size": list(image_size),
        "pattern_inner_corners": list(PATTERN),
        "square_size_mm": square_mm,
        "view_count": len(image_points),
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.ravel().tolist(),
        "rms_reprojection_error_px": float(rms),
        "per_view_rms_px": errors,
        "board_normal_span_degrees": normal_span,
        "quality_review": {"threshold_px": max_rms, "warnings": warnings,
                           "needs_review": bool(warnings)},
        "board_to_camera_rvecs": [v.ravel().tolist() for v in rotations],
        "board_to_camera_tvecs_mm": [v.ravel().tolist() for v in translations],
        "notes": ["Reprojection errors are training residuals, not independent accuracy measurements.",
                  "Per-view board poses are NOT robot-base transforms.",
                  "A separate camera-to-robot plane mapping is required before robotic picking.",
                  "Keep camera resolution, focus and zoom unchanged after calibration."],
    }


def read_square_size(value, folder=None):
    if value is None and folder and (folder / "board.json").is_file():
        metadata = json.loads((folder / "board.json").read_text(encoding="utf-8"))
        if metadata.get("pattern_inner_corners") != list(PATTERN):
            raise ValueError("Saved board pattern does not match 9x6 inner corners")
        value = metadata["square_size_mm"]
    if value is None:
        value = float(input("請輸入實際列印後每一個方格的邊長（mm，不是整張寬度）："))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Square size must be a positive finite number in mm")
    return float(value)


def calibrate_folder(folder, square_mm, min_views, max_rms):
    paths = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    corners_all, accepted, rejected = [], [], []
    image_size = None
    for path in paths:
        image = load_image(path)
        if image is None:
            rejected.append((path.name, "unreadable image")); continue
        size = (image.shape[1], image.shape[0])
        if image_size is not None and size != image_size:
            rejected.append((path.name, "image resolution mismatch")); continue
        corners = detect(image)
        if corners is None:
            rejected.append((path.name, "9x6 corners not found")); continue
        if same_view(corners, corners_all, size):
            rejected.append((path.name, "near-duplicate board pose")); continue
        image_size = size
        corners_all.append(corners)
        accepted.append(path.name)
        annotated = image.copy()
        cv2.drawChessboardCorners(annotated, PATTERN, corners, True)
        save_image(folder / "annotated" / path.name, annotated)
    for name, reason in rejected:
        print(f"[SKIP] {name}: {reason}")
    if len(accepted) < min_views:
        raise ValueError(f"Only {len(accepted)} distinct valid views; need {min_views}. Images remain saved.")
    result = solve_calibration(corners_all, image_size, square_mm, max_rms)
    result["images"] = accepted
    result["rejected_images"] = [{"file": name, "reason": reason} for name, reason in rejected]
    result["created_at"] = datetime.now().astimezone().isoformat()
    output = folder / "camera_calibration.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    np.savez(folder / "camera_calibration.npz", camera_matrix=np.asarray(result["camera_matrix"]),
             distortion_coefficients=np.asarray(result["distortion_coefficients"]),
             image_size=np.asarray(image_size), square_size_mm=square_mm)
    with (folder / "per_view_errors.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream); writer.writerow(["image", "rms_reprojection_error_px"])
        writer.writerows(zip(accepted, result["per_view_rms_px"]))
    sample = load_image(folder / accepted[0])
    corrected = cv2.undistort(sample, np.asarray(result["camera_matrix"]),
                             np.asarray(result["distortion_coefficients"]))
    save_image(folder / "annotated" / "undistorted_example.png", corrected)
    print(f"\nCalibration saved: {output}")
    print(f"Views: {len(accepted)}; RMS: {result['rms_reprojection_error_px']:.4f} px")
    for warning in result["quality_review"]["warnings"]:
        print("[REVIEW]", warning)
    print("This calibrates the lens only. Robot XY mapping is a separate step.")
    return result


def open_camera(args, expected_size=None):
    backend = cv2.CAP_DSHOW if __import__("os").name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open camera {args.camera}; close other camera applications or try --camera 1")
    width, height = expected_size or (args.width, args.height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def capture(args):
    square_mm = read_square_size(args.square_mm, args.images)
    folder = args.images or ROOT / "calibration" / datetime.now().strftime("session_%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=True)
    metadata = folder / "board.json"
    if metadata.exists():
        old_size = read_square_size(None, folder)
        if not math.isclose(old_size, square_mm):
            raise ValueError("Square size differs from the saved session; use a new folder")
    metadata.write_text(json.dumps({"pattern_inner_corners": list(PATTERN), "square_size_mm": square_mm,
                                    "camera_index": args.camera}, indent=2), encoding="utf-8")
    cap = open_camera(args)
    collected = []
    image_size = None
    print("棋盤：10x7 格，9x6 內角點；相機保持固定，移動並傾斜棋盤。")
    print("SPACE 拍攝；C 校正；Q 離開。請收集 15-25 張不同位置、大小及傾斜角度的照片。")
    print("照片資料夾：", folder)
    print("Keep focus/zoom fixed. Do not use repeated identical views or a screen image of the board.")
    try:
        cv2.namedWindow("Chessboard Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Chessboard Calibration", 1100, 700)
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera frame read failed")
            size = (frame.shape[1], frame.shape[0])
            if image_size is None:
                image_size = size
                print("Actual camera resolution:", size)
            if size != image_size:
                raise RuntimeError("Camera resolution changed during capture")
            corners = detect(frame, accurate=False)
            display = frame.copy()
            if corners is not None:
                cv2.drawChessboardCorners(display, PATTERN, corners, True)
            color = (0, 200, 0) if corners is not None else (0, 0, 255)
            cv2.putText(display, f"9x6: {'FOUND' if corners is not None else 'NOT FOUND'} | Saved this run: {len(collected)}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(display, "SPACE: save | C: calibrate | Q: exit", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 80, 0), 2)
            cv2.imshow("Chessboard Calibration", display)
            key = cv2.waitKey(1) & 255
            if key in (ord("q"), 27):
                break
            if key == 32:
                corners = detect(frame)
                if corners is None:
                    print("[NOT SAVED] Entire 9x6 board must be visible and sharp")
                elif same_view(corners, collected, size):
                    print("[NOT SAVED] Pose too similar; move or tilt the board")
                else:
                    path = folder / (datetime.now().strftime("image_%H%M%S_%f") + ".png")
                    save_image(path, frame)
                    collected.append(corners)
                    print(f"[SAVED {len(collected)}] {path.name}")
            if key == ord("c"):
                try:
                    calibrate_folder(folder, square_mm, args.min_views, args.max_rms)
                except ValueError as exc:
                    print("[CALIBRATION]", exc)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def preview(args):
    if not args.calibration:
        raise ValueError("--calibration must point to camera_calibration.json")
    result = json.loads(args.calibration.read_text(encoding="utf-8"))
    if result.get("calibration_type") != "camera_intrinsics_only":
        raise ValueError("Not a camera calibration result from this program")
    size = tuple(result["image_size"])
    matrix = np.asarray(result["camera_matrix"], np.float64)
    distortion = np.asarray(result["distortion_coefficients"], np.float64)
    cap = open_camera(args, size)
    try:
        cv2.namedWindow("Original | Undistorted", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Original | Undistorted", 1400, 650)
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera read failed")
            if (frame.shape[1], frame.shape[0]) != size:
                raise ValueError("Live resolution differs from calibration; refusing to apply mismatched parameters")
            corrected = cv2.undistort(frame, matrix, distortion)
            cv2.imshow("Original | Undistorted", np.hstack([frame, corrected]))
            if cv2.waitKey(1) & 255 in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("capture", "calibrate", "preview", "check"), default="capture")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--square-mm", type=float)
    parser.add_argument("--images", type=Path, help="Capture session / folder of calibration photographs")
    parser.add_argument("--image", type=Path, help="Single board image for corner detection only")
    parser.add_argument("--calibration", type=Path, help="Saved camera_calibration.json for live preview")
    parser.add_argument("--min-views", type=int, default=15)
    parser.add_argument("--max-rms", type=float, default=1.0)
    args = parser.parse_args()
    if args.min_views < 10 or args.width <= 0 or args.height <= 0:
        parser.error("Minimum views must be at least 10; resolution must be positive")
    if not math.isfinite(args.max_rms) or args.max_rms <= 0:
        parser.error("--max-rms must be positive and finite")
    try:
        if args.mode == "capture":
            capture(args)
        elif args.mode == "calibrate":
            if not args.images or not args.images.is_dir():
                raise ValueError("Provide --images with a folder containing real camera photographs")
            square_mm = read_square_size(args.square_mm, args.images)
            calibrate_folder(args.images, square_mm, args.min_views, args.max_rms)
        elif args.mode == "preview":
            preview(args)
        else:
            if not args.image:
                raise ValueError("Provide --image for corner detection")
            image = load_image(args.image)
            if image is None:
                raise ValueError("Cannot read the image")
            corners = detect(image)
            if corners is None:
                raise ValueError("9x6 inner corners not found")
            cv2.drawChessboardCorners(image, PATTERN, corners, True)
            output = ROOT / "calibration" / "chessboard_check.png"
            save_image(output, image)
            print(f"Detected {len(corners)} corners (9x6). Saved: {output}")
            print("This single image checks the pattern only; it is NOT a lens calibration.")
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Any captured images remain saved.")
    except (ValueError, OSError, RuntimeError, cv2.error) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()

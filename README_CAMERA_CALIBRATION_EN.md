# Chessboard Camera Calibration

Program: `code/camera_calibrate.py`. This calibrates the camera lens only. It does not connect to or move the robot.

## Your board

The supplied `F02WGqp.png` contains **10 x 7 squares and 9 x 6 internal corners (54 points)**. The program is configured for these 9 x 6 internal corners. Detection of all 54 points has been verified against the supplied image.

Print the pattern without stretching it, keep the white border, and mount it flat on a rigid surface. Measure one actual printed square edge in millimeters with a ruler. Check that horizontal and vertical square edges have the same length. Do not guess the millimeter size from the image file. The original PNG alone cannot calibrate your physical camera.

## Installation

Use the same Python interpreter selected in VS Code. If needed, install:

```powershell
python -m pip install -r requirements_camera.txt
```

On the current PC, OpenCV and NumPy are available in `D:\miniconda3\python.exe`. A new PC needs its own working interpreter and these packages. No robot SDK is required for camera calibration.

## Capture and calibrate in VS Code

1. Open the complete `robot arm` folder and select the working Python interpreter.
2. Close other applications using the camera.
3. Open `code/camera_calibrate.py` and select **Run Python File in Terminal**.
4. In the terminal, enter the measured printed square edge length in mm. For example, enter `25` only if your square really measures 25 mm.
5. The program opens camera index 0 at a requested 1280 x 720 resolution. It prints the actual resolution used. The internal-corner pattern must be fully visible.
6. Keep the camera fixed. Move the board between the center, edges and corners of the image; vary distance and tilt. Hold it still before each capture. Do not change focus, zoom or resolution. Lock autofocus using your camera controls when possible.
7. Press **SPACE** to save a photograph. Very similar views are rejected. The program saves the original frame without drawn corner markers.
8. Collect **15-25 sharp photographs with varied board poses**, then press **C** to calculate calibration. At least 15 valid, distinct views are required by default. Repeated photographs of one fixed, front-facing board are not sufficient.
9. Read the output and review warnings. Press **Q** or Escape to release the camera and close the window. If calibration needs more images, continue capturing before pressing C again.

Changing the chessboard pose is necessary during lens calibration even though the final camera mount stays fixed. Later camera-to-robot plane mapping uses the fixed working plane separately.

## Terminal examples

Run these from the `robot arm` folder. Replace `python` with the full interpreter path if it is not on PATH.

```powershell
# Capture: replace 25 with the measured square size.
python code/camera_calibrate.py --square-mm 25

# Use another camera index.
python code/camera_calibrate.py --camera 1 --square-mm 25

# Use an explicitly named session folder (also supports resuming capture).
python code/camera_calibrate.py --square-mm 25 --images calibration/session_a

# Recalculate from photographs already saved in that session.
python code/camera_calibrate.py --mode calibrate --images calibration/session_a

# Show live original and undistorted images side by side.
python code/camera_calibrate.py --mode preview --calibration calibration/session_a/camera_calibration.json

# Check the supplied image's corner pattern only; no camera is opened.
python code/camera_calibrate.py --mode check --image F02WGqp.png
```

If you do not specify a session folder, each capture run creates a new timestamped folder under `calibration`. The terminal prints its full path. Each session stores its own square-size metadata. Do not mix photographs from different cameras, focus settings or resolutions.

## Output files

| File | Contents |
| --- | --- |
| `board.json` | Board pattern, measured square size, camera index |
| `image_*.png` | Original captured photographs |
| `camera_calibration.json` | Camera matrix K, distortion coefficients, resolution, square size, pose and error reports |
| `camera_calibration.npz` | Camera parameters for loading with NumPy |
| `per_view_errors.csv` | Reprojection RMS in pixels for each accepted image |
| `annotated/` | Detected corners and an undistorted example |

The default RMS review threshold is **1 pixel**, which is a screening value, not a guarantee of picking accuracy. Inspect high-error images, board flatness, blur, reflections and pose diversity. Board tilt diversity is also checked. The program does not automatically remove poor images to hide a high error.

Reprojection errors use the same views used to fit the model. Independently validate correction using new camera photographs. The per-image board-to-camera poses in the JSON are not robot-base transforms.

## Next step: robot coordinates

This program does **not** calculate a homography to the MG400 or modify `config.json`. The next step is to collect matching image points and MG400 XY coordinates on a fixed plane, then calculate and validate that mapping. Keep the pixel convention consistent: if you calibrate the robot mapping using undistorted pixels, detection points must use the same undistortion and camera matrix without an unaccounted crop or resize.

Official references:

- https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

Offline checks use the supplied pattern and synthetic camera projections with known parameters. No physical camera calibration or robot movement is performed by those checks.

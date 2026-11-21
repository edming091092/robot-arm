# Robot Arm

> **Complete folder including sam3.pt:** [Download all archive volumes](https://github.com/edming091092/robot-arm/releases/tag/complete-folder-2026-11-21) and extract `.001` with 7-Zip. See [full download instructions](FULL_DOWNLOAD.md).

English desktop interface for SAM3 text-prompt segmentation, calibrated camera-to-robot XY conversion, and Dobot MG400 movement to a manually entered Z.

## Install (Windows)

Install 64-bit CPython 3.10-3.14 with Tk and VS Code with the Python extension. Create the environment on each computer; do not copy a virtual environment from another PC.

From the project folder, run with your installed Python:

```powershell
python code/sam3_runtime.py --install
```

CPU is the default. For a compatible NVIDIA GPU:

```powershell
python code/sam3_runtime.py --install --device cuda
```

Select `.venv/Scripts/python.exe` as the VS Code interpreter. Put your separately obtained SAM3 checkpoint in `models/sam3.pt`, or choose its location in the interface. The complete release includes model weights, installers, images and calibration sessions. The automatic Source code ZIP excludes large binaries; use the full release download below.

## Start

```powershell
.venv/Scripts/python.exe code/sam3_robot_move/app.py
```

You can also open that file in VS Code and select **Run Python File in Terminal**.

1. Select **Load XY Mapping** and load your `robot_xy_mapping.json` generated from paired image and robot coordinates. Lens calibration alone is insufficient.
2. Select **Open Camera**. Keep the calibrated camera position, lens settings and image resolution unchanged.
3. Enter an English prompt (for example `coin`), choose CPU or GPU, and select **Start Live Detection**.
4. Enter **Hover Z (mm)** in robot coordinates. This is a commanded TCP height, not a measured distance above the object.
5. Click a detected object in **Latest Detection** to display its robot X/Y and entered Z, then select **Move Above Selected**. **One-Click Hover** chooses the highest-confidence mapped object.
6. Restart live detection for the next target after moving.

The application keeps the current R orientation and moves vertically to clearance, horizontally over the target, then to the entered Z. No vacuum or gripper commands are sent. Robot motion starts only when a motion button is pressed. Enable the robot in the official software and select TCP/IP control first. Set the correct robot IP, camera index, User and Tool in `config.json`.

Approximate mappings are accepted and labeled. Their calibration error remains. Planar XY conversion applies to the calibrated plane: entering Z does not compensate for object-height parallax. Motion requires a fresh camera result and a reachable route. Optional `pickup_workspace_mm` bounds apply only when configured; legacy random-motion bounds are not included.

## Calibration and shared modules

- `code/camera_calibrate(1).py`: lens calibration using the 9 x 6 internal-corner board.
- `code/calibrate_robot_xy(2).py`: pair image points with measured robot points and export XY mapping.
- `code/sam3_robot_move/app.py`: simplified combined interface.
- `code/integrated_pick/app.py`: shared integrated interface and movement logic.
- Other Python modules implement SAM3 inference, environment setup, coordinate mapping, communication and route checks. Keep the folder layout intact.

See the English calibration and SAM3 guides for details. Calibration files must be produced or supplied for the actual camera and robot setup.

## Verification

The simplified entry, saved-mapping import, segmentation-mask coordinate conversion, manual Z input and motion route have been checked with a mock robot. This upload does not claim a complete physical camera-guided motion test. Supervise physical motion and keep the travel path clear.

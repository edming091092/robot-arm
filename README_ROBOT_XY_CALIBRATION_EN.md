# Camera-to-Robot XY: Nine-Point Conversion Preview

Run `code/calibrate_robot_xy.py` (the existing renamed `calibrate_robot_xy(2).py` has the same update). The interface now computes H from nine fit points without requiring three extra validation points. This program sends no movement commands.

## Reuse the nine points already recorded

1. Close the old interface and run the updated program.
2. Select `calibration/robot_xy_20261105_174430_118048/camera_calibration.json`.
3. Click **Load saved points.csv** and select the CSV in that same session folder. The nine points and their saved undistorted image are loaded.
4. Click **Compute H + preview XY**. The interface fits all nine recorded points and selects Inspect mode automatically.
5. Click inside the boundary covered by those nine points to see predicted robot X/Y in mm. No robot movement is performed. Opening the live camera is unnecessary when inspecting the saved image.

The program refuses a CSV associated with different lens intrinsics. Loading existing points creates a new output session; the original CSV is preserved.

## Current recorded data

The existing nine-point session has already been converted offline. Its outputs are:

- `calibration/robot_xy_20261105_174430_118048/robot_xy_mapping.json`
- `calibration/robot_xy_20261105_174430_118048/conversion_preview.csv`

Fit RMS is approximately **11.125 mm**, and maximum fit error is approximately **19.468 mm**. These are residuals on the same points used for fitting, not independent accuracy measurements.

Recorded TCP Z ranges from -202.009476 to -75.911560 mm, a spread of about 126 mm. R also varies. Those data do not satisfy the fixed-plane/fixed-orientation collection assumptions. The preview still computes H as requested, but the result is marked `preview_only: true`, `passed: false`, and includes the data-quality issues. It is not approved for automatic picking by this program.

## Recording a new nine-point set

Select the actual lens calibration JSON, open the matching camera, click a physical mark before jogging the robot over it, manually align the suction TCP with that mark, and read or enter the stationary robot pose. Click Add matched pair and repeat for nine well-distributed marks. Use the same physical plane, TCP Z, R, User and Tool throughout. Camera mount, focus, zoom and resolution must stay fixed.

The displayed image uses undistorted pixels with original K, full original resolution and no crop. Screen clicks are converted back to full-resolution coordinates. Snapshots and `points.csv` are saved automatically.

## Conversion helpers

For deliberate offline inspection of this preview mapping:

```python
import json
from pathlib import Path
from robot_xy_mapping import pixel_to_robot

mapping = json.loads(Path('path/to/robot_xy_mapping.json').read_text(encoding='utf-8'))
x, y = pixel_to_robot([u, v], mapping, pixel_space='raw', allow_preview=True)
```

Use `pixel_space='raw'` for original camera image pixels, or `pixel_space='undistorted'` for the full-resolution undistorted image with the same K and no crop. Resized SAM3 input coordinates must first be converted to original-image coordinates. Extrapolation outside the fit-point boundary remains refused.

`allow_preview` defaults to False. The default helper therefore continues to reject an unvalidated preview mapping. Strict independent-validation fitting remains available in the core API; this GUI no longer requires collecting extra validation points.

## Offline recalculation

```powershell
python code/calibrate_robot_xy.py --fit-csv path/to/points.csv --intrinsics path/to/camera_calibration.json --output path/to/robot_xy_mapping.json
```

This fits the nine points without opening a camera or robot connection. It saves the preview even when recorded Z/R or fit residuals require review. Geometrically degenerate points or an invalid homography are still rejected.

The new behavior was checked with the actual saved nine-point CSV, the hidden Tk interface, known mappings, invalid point sets, CSV/image loading and the distinction between preview conversion and normally validated conversion. No physical motion was performed during this update.

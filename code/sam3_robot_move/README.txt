SAM3 - Camera to Robot Coordinates

START
Open app.py in VS Code and choose Run Python File in Terminal.
Keep this folder inside the original robot arm/code folder: it reuses the
project's detection, calibration and MG400 communication modules.
On a new computer, set up the project Python dependencies first using
code/sam3_runtime.py --install (see the project SAM3 setup instructions).

WORKFLOW
1. Click Load XY Mapping and select your robot_xy_mapping.json.
   Use the robot XY mapping from paired image/robot points, not board.json
   or camera_calibration.json, which cannot provide robot coordinates alone.
2. Click Open Camera. Keep the calibrated camera fixed in the same position.
3. Enter an English prompt such as coin or mouse. Choose CPU or GPU (CUDA).
   Click Start Live Detection.
4. Enter Hover Z in millimeters in the robot coordinate system. This is the
   commanded TCP height, not the distance above the object's surface.
5. Click an object in Latest Detection. Its mapped X, Y and entered Z appear.
   Click Move Above Selected to move. One-Click Hover chooses the highest
   confidence object that has a mapped position and starts the same motion.
6. Start live detection again to choose another target after a move.

BEHAVIOR
The program takes an interior point of the segmented object, transforms its
image coordinates into calibrated robot X/Y, and uses your entered Z.
R stays at the robot's current orientation. Motion raises vertically if
needed, travels at the higher of current Z and Hover Z, then reaches Hover Z.
The robot must already be enabled and idle in TCP/IP control mode.
No gripper, vacuum, grasping or release commands are sent.

The old random-movement X/Y/Z limits are not used here. Existing optional
pickup_workspace_mm limits in config.json still apply if configured.
Approximate XY mappings are accepted for hover demonstrations and labeled
Approximate; their errors remain. Targets must be within the calibrated
image area. The camera resolution must match the calibration.
This planar XY mapping applies to the calibrated plane; object height can
shift the mapped XY. Entering Z does not measure object height or fix that.
Old target frames are refreshed before starting motion. Keep the travel
path clear and watch the robot during motion.

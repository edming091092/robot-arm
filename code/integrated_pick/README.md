# SAM3 and MG400 Vision Hover

Run the app in VS Code or PowerShell.

```powershell
& 'D:\sam3_env\python.exe' 'D:\Desktop\robot arm\code\integrated_pick\app.py'
```

Keep this folder under `code` so the app can use the shared SAM3, XY mapping and MG400 modules.

1. Click **Load XY Mapping** and choose your existing mapping. The app labels estimates from preview mappings **Approximate XY**. Use **Calibrate XY** when you want to improve the mapping.
2. Enter **Hover Z** in millimeters. Check that the tool and arm have clearance along the route. If you have measured XYZ travel limits, you can enter them in **Workspace Setup**.
3. Open the configured camera and start live detection. Click an object in **Latest Detection**, then click **Move Above Selected**. **One-Click Hover** uses the highest confidence mapped object. The arm keeps its current R, travels at clearance, and stops at Hover Z.

The controller checks inverse kinematics at sampled points along the planned route. Frames older than 15 seconds trigger fresh detection for a new selection. Hover targets from an approximate mapping can be offset from the object.

The fixed-coordinate test reached X 263.76, Y 226.03, Z 50 mm on the real robot. The integrated approximate-mapping hover flow was tested with a simulated robot.

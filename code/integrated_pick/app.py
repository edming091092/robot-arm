"""SAM3 camera detection, XY preview and single-target MG400 hover motion."""
import json
import math
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from sam3_runtime import ensure_environment

if __name__ == "__main__":
    ensure_environment(Path(__file__).resolve())

from mg400_tcp import MG400
from robot_xy_mapping import pixel_to_robot
from sam3_prompt import PromptApp, cv2, np
from test_target_move import check_route

MAX_FRAME_AGE = 15


def parse_height(text, label):
    if not text.strip():
        raise ValueError(f"Enter {label} in millimeters.")
    try:
        height = float(text)
    except ValueError:
        raise ValueError(f"Enter a numeric value for {label} in millimeters.") from None
    if not math.isfinite(height):
        raise ValueError(f"Enter a finite value for {label} in millimeters.")
    return height


def suction_center(mask):
    # Pad the image so that an object touching the border still has a finite interior.
    binary = np.pad((mask > 0).astype(np.uint8), 1)
    distances = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, radius, _, (u, v) = cv2.minMaxLoc(distances)
    if radius <= 0:
        raise ValueError("Empty object mask")
    return (u - 1, v - 1)


def workspace_check(cfg, pose):
    limits = cfg.get("pickup_workspace_mm")
    if limits is None:
        return
    if not isinstance(limits, dict):
        raise ValueError("Set pickup_workspace_mm in config.json to the measured pickup area.")
    for axis, value in zip("xyz", pose[:3]):
        bounds = limits.get(axis)
        if (not isinstance(bounds, list) or len(bounds) != 2
                or not all(type(n) in (int, float) and math.isfinite(n) for n in bounds)
                or not math.isfinite(value) or not bounds[0] <= value <= bounds[1]):
            raise ValueError(f"{axis.upper()} {value:.3f} is outside the configured workspace.")


def check_hover(cfg, mapping, xy, hover_z):
    if not mapping or (mapping.get("passed") is not True and mapping.get("preview_only") is not True):
        raise ValueError("Load an XY mapping for hover.")
    if (any(mapping.get(key) != cfg.get(key) for key in ("user_index", "tool_index"))
            or mapping.get("camera_index", cfg.get("camera_index")) != cfg.get("camera_index")):
        raise ValueError("Mapping camera, User or Tool differs from config.json.")
    if len(xy) != 2 or not all(math.isfinite(n) for n in (*xy, hover_z)):
        raise ValueError("Enter a finite hover position.")
    workspace_check(cfg, [*xy, hover_z])


def move_above(cfg, mapping, xy, hover_z, frame_at, report):
    check_hover(cfg, mapping, xy, hover_z)
    bot = MG400(cfg["robot_ip"], cfg["user_index"], cfg["tool_index"], speed=3)
    try:
        if bot.mode() != 5:
            raise RuntimeError("Enable the robot and leave it idle before moving.")
        bot.check_frames(bot.feedback.read(timeout=3, require_new=True))
        current = bot.pose()
        clearance = max(current[2], hover_z)
        waypoints = [
            ("Raising to clearance", [*current[:2], clearance, current[3]]),
            ("Moving above target", [*xy, clearance, current[3]]),
            ("Moving to Hover Z", [*xy, hover_z, current[3]]),
        ]
        def check(pose):
            workspace_check(cfg, pose)
        check(current)
        for _, pose in waypoints:
            check(pose)
        if time.monotonic() - frame_at > MAX_FRAME_AGE:
            raise RuntimeError("Target frame expired before travel. Detect again.")
        check_route(bot, current, [pose for _, pose in waypoints])
        actual = bot.prepare()
        if (math.dist(actual[:3], current[:3]) > 0.5
                or abs((actual[3] - current[3] + 180) % 360 - 180) > 1):
            raise RuntimeError("Start position changed during the route check. Detect again.")
        previous = current
        for label, pose in waypoints:
            if previous[:3] == pose[:3]:
                continue
            if time.monotonic() - frame_at > MAX_FRAME_AGE and label != "Moving to Hover Z":
                raise RuntimeError("Target frame expired before travel. Detect again.")
            report(label)
            bot.move(pose, check_pose=check)
            previous = pose
        actual = bot.pose()
        if math.dist(actual[:3], [*xy, hover_z]) > 1:
            raise RuntimeError("The robot did not reach the hover position.")
        return actual
    except Exception:
        if bot.queue_owned:
            bot.stop()
        raise
    finally:
        bot.close()


class PickApp(PromptApp):
    def __init__(self, root, *, simple_mode=False):
        self.cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
        self.mapping = None
        self.mapping_path = None
        self.targets = []
        self.selected = None
        self.frozen = False
        self.picking = False
        self.pending_detection = False
        self.frame_at = self.result_at = 0
        self.frame_source = self.result_source = None
        super().__init__(root, None)
        root.title("SAM3 and MG400 Vision Hover")
        row = ttk.Frame(root, padding=8)
        row.pack(fill="x", before=self.canvas)
        self.mapping_button = ttk.Button(row, text="Load XY Mapping", command=self.load_mapping)
        self.mapping_button.pack(side="left", padx=4)
        self.calibrate_button = ttk.Button(row, text="Calibrate XY", command=self.calibrate_xy)
        if not simple_mode:
            self.calibrate_button.pack(side="left", padx=4)
            ttk.Button(row, text="Workspace Setup", command=self.workspace_setup).pack(side="left", padx=4)
        self.hover_z = tk.StringVar(value=self.cfg.get("safe_z_mm") if self.cfg.get("safe_z_mm") is not None else "")
        ttk.Label(row, text="Hover Z (mm)").pack(side="left", padx=4)
        ttk.Entry(row, textvariable=self.hover_z, width=9).pack(side="left")
        self.hover_z.trace_add("write", lambda *_: self.show_target())
        ttk.Button(row, text="Move Above Selected", command=lambda: self.start_hover(False)).pack(side="left", padx=8)
        ttk.Button(row, text="One-Click Hover", command=lambda: self.start_hover(True)).pack(side="left")
        self.target_text = tk.StringVar(value="Load a mapping, then click an object in Latest Detection to select it.")
        ttk.Label(root, textvariable=self.target_text, wraplength=1080, padding=8).pack(fill="x", before=self.canvas)
        self.pick_readiness = tk.StringVar()
        ttk.Label(root, textvariable=self.pick_readiness, wraplength=1080, padding=(8, 0, 8, 8)).pack(fill="x", before=self.canvas)
        self.canvas.bind("<Button-1>", self.select_object)
        self.camera_index.set(str(self.cfg.get("camera_index", 0)))
        self.update_pick_readiness()

    def load_mapping(self):
        if self.busy or self.picking:
            self.status.set("Wait for the current operation to finish. Stop live detection before changing mappings.")
            return
        filename = filedialog.askopenfilename(title="Select robot_xy_mapping.json", initialdir=str(ROOT / "calibration"),
                                              filetypes=[("Mapping JSON", "*.json")])
        if not filename:
            return
        self.apply_mapping(Path(filename))

    def calibrate_xy(self):
        if self.busy or self.picking:
            self.status.set("Stop live detection and wait for the current operation to finish.")
            return
        try:
            args = [sys.executable, str(CODE / "calibrate_robot_xy(2).py"), "--validate-for-pick",
                    "--camera", str(self.cfg["camera_index"])]
            if self.mapping_path is not None:
                intrinsics = self.mapping_path.parent / "camera_calibration.json"
                if intrinsics.is_file():
                    args.extend(["--intrinsics", str(intrinsics)])
            self.stop_camera()
            self.clear_targets()
            handoff_dir = tempfile.TemporaryDirectory(prefix="mg400_calibration_")
            handoff = Path(handoff_dir.name) / "mapping.txt"
            try:
                child = subprocess.Popen([*args, "--handoff", str(handoff)])
            except OSError:
                handoff_dir.cleanup()
                raise
        except (OSError, KeyError) as exc:
            messagebox.showerror("XY Calibration", str(exc))
            return
        self.set_busy(True)
        self.status.set("Collect 9 fit points and 3 validation points at the same Z and R. Close calibration after PASS.")

        def finish_calibration():
            if child.poll() is None:
                self.root.after(250, finish_calibration)
                return
            self.set_busy(False)
            try:
                if child.returncode == 0 and handoff.is_file():
                    self.apply_mapping(Path(handoff.read_text(encoding="utf-8")))
                else:
                    self.status.set("Calibration closed without a PASS mapping. Open Calibrate XY to continue.")
            except OSError as exc:
                messagebox.showerror("XY Calibration", str(exc))
            finally:
                handoff_dir.cleanup()

        self.root.after(250, finish_calibration)

    def workspace_setup(self):
        if self.busy or self.picking:
            self.status.set("Wait for the current operation to finish before editing pickup settings.")
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("Workspace Setup")
        dialog.transient(self.root)
        dialog.grab_set()
        ranges = self.cfg.get("pickup_workspace_mm") or {}
        fields = {}
        ttk.Label(dialog, text="Enter the measured XYZ motion limits in millimeters.").grid(row=0, column=0, columnspan=3, padx=10, pady=8)
        for row, axis in enumerate("xyz", 1):
            ttk.Label(dialog, text=f"{axis.upper()} minimum / maximum").grid(row=row, column=0, padx=10, sticky="w")
            fields[axis] = [tk.StringVar(value=value) for value in ranges.get(axis, ["", ""])]
            for col, value in enumerate(fields[axis], 1):
                ttk.Entry(dialog, width=12, textvariable=value).grid(row=row, column=col, padx=4, pady=2)
        verified = tk.BooleanVar(value=False)
        ttk.Checkbutton(dialog, text="I checked these motion limits on the robot", variable=verified).grid(row=8, column=0, columnspan=3, pady=6)

        def save_setup():
            try:
                limits = {axis: [parse_height(field.get(), f"{axis.upper()} limit") for field in values]
                          for axis, values in fields.items()}
                if any(low >= high for low, high in limits.values()):
                    raise ValueError("Each minimum must be lower than its maximum.")
                path = ROOT / "config.json"
                config = json.loads(path.read_text(encoding="utf-8-sig"))
                config.update(pickup_workspace_mm=limits, hover_workspace_verified=verified.get())
                temp = path.with_suffix(".pickup.tmp")
                temp.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                temp.replace(path)
                self.cfg = config
                self.update_pick_readiness()
                dialog.destroy()
            except (ValueError, OSError) as exc:
                messagebox.showerror("Workspace Setup", str(exc), parent=dialog)

        ttk.Button(dialog, text="Save Settings", command=save_setup).grid(row=9, column=0, columnspan=3, pady=8)

    def apply_mapping(self, filename):
        try:
            mapping = json.loads(filename.read_text(encoding="utf-8-sig"))
            if (mapping.get("calibration_type") != "planar_camera_to_robot_xy"
                    or "invalidated_reason" in mapping
                    or not all(key in mapping for key in ("image_size", "camera_matrix", "distortion_coefficients", "H", "valid_pixel_hull"))):
                raise ValueError("Select a robot XY mapping file.")
            resume = self.continuous
            camera_index = mapping.get("camera_index")
            if self.capture is not None and camera_index is not None and self.image_path != f"camera:{camera_index}":
                self.stop_camera()
            self.mapping = mapping
            self.mapping_path = Path(filename)
            self.clear_targets()
            if camera_index is not None:
                self.camera_index.set(str(camera_index))
            self.apply_mapping_resolution()
            if self.capture is not None:
                self.continuous = resume
                self.set_busy(False)
            kind = "Validated mapping" if mapping.get("passed") is True else "Approximate mapping"
            next_step = "Start Live Detection to refresh the targets." if self.capture is not None else "Open the camera and start live detection."
            self.target_text.set(f"{kind} loaded from {self.mapping_path.parent.name}. {next_step}")
            self.status.set("Mapping changed. Waiting for a new detection result.")
            self.update_pick_readiness()
        except (ValueError, OSError) as exc:
            messagebox.showerror("XY Mapping", str(exc))

    def clear_targets(self):
        self.last_result = self.overlay = self.selected = None
        self.targets = []
        self.frozen = False
        self.set_busy(self.busy)
        self.render()

    def toggle_camera(self):
        if self.picking:
            return
        opening = self.capture is None and not self.busy
        super().toggle_camera()
        if opening and self.capture is not None:
            self.clear_targets()

    def apply_mapping_resolution(self):
        self.camera_resolution = tuple(self.mapping["image_size"])
        if self.capture is not None:
            self.capture.resolution = self.camera_resolution
            self.image = None
            self.canvas.delete("all")

    def render(self):
        super().render()
        self.update_pick_readiness()

    def update_pick_readiness(self):
        if not hasattr(self, "pick_readiness"):
            return
        if self.picking:
            status = "Move in progress."
        else:
            missing = []
            if self.capture is None or self.image_path != f"camera:{self.cfg.get('camera_index')}":
                missing.append("open the configured live camera")
            if self.mapping is None:
                missing.append("load an XY mapping")
            if not self.hover_z.get().strip():
                missing.append("enter Hover Z")
            if not missing:
                if self.last_result is None or time.monotonic() - self.result_at > MAX_FRAME_AGE:
                    missing.append("obtain a fresh live detection")
                elif not any(obj["robot_xy_mm"] is not None for obj, _ in self.targets):
                    missing.append("detect a target inside the mapped area")
            status = "Motion requirements — " + "; ".join(missing) if missing else "Fresh mapped target available."
            if self.capture is not None and self.last_result is not None and self.result_source == self.image_path:
                age = int(max(0, time.monotonic() - self.result_at))
                status += f" | Target frame: {age} s old (limit {MAX_FRAME_AGE} s)"
        if self.pick_readiness.get() != status:
            self.pick_readiness.set(status)

    def choose_image(self):
        if not self.picking:
            super().choose_image()
            if self.last_result is None:
                self.clear_targets()

    def run(self):
        if self.picking:
            return
        if not self.busy and not self.continuous:
            self.clear_targets()
        super().run()

    def infer(self):
        self.frame_at = self.camera_frame_at if self.capture is not None else time.monotonic()
        self.frame_source = self.image_path
        super().infer()

    def accept_detection(self, result):
        if self.frozen:
            return False
        overlay, masks, meta = result
        self.result_at, self.result_source = self.frame_at, self.frame_source
        self.selected = None
        self.targets = []
        issue = "Load an XY mapping to convert suction centers."
        can_map = self.mapping and tuple(self.mapping["image_size"]) == tuple(meta["image_size"])
        if self.mapping and not can_map:
            issue = "Frame resolution differs from the XY mapping."
        for obj, mask in zip(meta["objects"], masks):
            uv = suction_center(mask)
            xy = None
            if can_map:
                try:
                    xy = pixel_to_robot(uv, self.mapping, allow_preview=True).tolist()
                except ValueError as exc:
                    issue = str(exc)
            obj.update(suction_center_uv=list(uv), robot_xy_mm=xy)
            self.targets.append((obj, mask))
            cv2.drawMarker(overlay, uv, (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
            if xy is not None:
                cv2.putText(overlay, f"X {xy[0]:.1f}  Y {xy[1]:.1f}", (uv[0], max(15, uv[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        mapped = sum(obj["robot_xy_mm"] is not None for obj, _ in self.targets)
        self.target_text.set(f"Mapped centers: {mapped}/{len(self.targets)}. Click an object to select it."
                             if mapped else issue)
        return True

    def select_object(self, event):
        box = self.canvas.bbox("detection")
        if self.picking or not box or not self.targets or not (box[0] <= event.x < box[2] and box[1] <= event.y < box[3]):
            return
        height, width = self.overlay.shape[:2]
        u = int((event.x - box[0]) * width / (box[2] - box[0]))
        v = int((event.y - box[1]) * height / (box[3] - box[1]))
        hits = [obj for obj, mask in self.targets if mask[v, u]]
        if hits:
            self.select_target(min(hits, key=lambda obj: obj["area_pixels"]))

    def select_target(self, obj):
        self.selected = obj
        self.continuous = False
        self.frozen = True
        self.set_busy(self.busy)
        self.show_target()

    def show_target(self):
        self.update_pick_readiness()
        if self.selected is None:
            return
        xy = self.selected["robot_xy_mm"]
        if xy is None:
            self.target_text.set(f"Object {self.selected['id']} is outside the mapped area or has no valid XY.")
            return
        suffix = " | Approximate XY" if self.mapping.get("passed") is not True else ""
        self.target_text.set(f"Object {self.selected['id']} | X {xy[0]:.2f} mm | Y {xy[1]:.2f} mm | "
                             f"Z {self.hover_z.get() or 'enter Hover Z'}{suffix}")

    def start_hover(self, best):
        if self.picking:
            return
        try:
            if self.last_result is None or self.capture is None or self.result_source != f"camera:{self.cfg.get('camera_index')}":
                raise ValueError("Motion requires a live frame from the configured camera.")
            if self.mapping is None:
                raise ValueError("Load an XY mapping first.")
            candidates = [obj for obj, _ in self.targets if obj["robot_xy_mm"] is not None]
            obj = max(candidates, key=lambda item: item["confidence"] or 0, default=None) if best else self.selected
            if not obj or obj["robot_xy_mm"] is None:
                raise ValueError("Select a detected object with mapped XY first.")
            hover_z = parse_height(self.hover_z.get(), "Hover Z")
            check_hover(self.cfg, self.mapping, obj["robot_xy_mm"], hover_z)
        except (ValueError, KeyError, TypeError) as exc:
            messagebox.showerror("Move Above Target", str(exc))
            return
        if time.monotonic() - self.result_at > MAX_FRAME_AGE:
            if self.busy:
                self.frozen = False
                self.continuous = True
                self.selected = None
                self.set_busy(True)
            else:
                self.clear_targets()
                self.continuous = True
                self.infer()
            self.target_text.set("Refreshing the target. Select a new result before moving.")
            return
        self.select_target(obj)
        self.pending_detection = self.busy
        self.picking = True
        self.set_busy(True)
        self.status.set(f"Preparing move above object {obj['id']}")
        mapping, frame_at = self.mapping, self.result_at

        def worker():
            try:
                actual = move_above(self.cfg, mapping, obj["robot_xy_mm"], hover_z, frame_at,
                                    lambda status: self.mailbox.put(("move_status", status)))
            except Exception as exc:
                self.mailbox.put(("move_done", (False, str(exc))))
            else:
                self.mailbox.put(("move_done", (True, f"Hover reached: X {actual[0]:.2f}, Y {actual[1]:.2f}, Z {actual[2]:.2f} mm.")))

        threading.Thread(target=worker).start()

    def handle_message(self, kind, value):
        if kind == "move_status":
            self.status.set(value)
            return True
        if kind == "move_done":
            self.picking = False
            self.clear_targets()
            self.set_busy(self.pending_detection)
            self.frozen = self.pending_detection
            success, message = value
            self.status.set(message)
            if not success:
                messagebox.showerror("Move stopped", message)
            return True
        if self.picking and kind in ("status", "done", "error"):
            if kind in ("done", "error"):
                self.pending_detection = False
            return True
        return False

    def close(self):
        if self.picking:
            self.status.set("The robot is moving. Wait for it to finish before closing.")
            return
        super().close()


if __name__ == "__main__":
    root = tk.Tk()
    PickApp(root)
    root.mainloop()

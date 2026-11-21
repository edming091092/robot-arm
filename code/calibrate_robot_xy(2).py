"""GUI for planar camera-to-robot XY calibration. No automatic robot motion.
Run in VS Code; choose the camera_calibration.json created in the previous step.
"""
import argparse
import json
import queue
import shutil
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit("Install camera dependencies: python -m pip install opencv-python numpy") from exc

from robot_xy_mapping import (finite, fit_mapping, load_intrinsics, load_samples,
                              pixel_to_robot, pose_consistency_issues, read_robot_pose, save_samples, validate_sample)

ROOT = Path(__file__).resolve().parents[1]


class MappingApp:
    def __init__(self, root, intrinsics_path, cfg, camera, validate_for_pick=False, handoff=None):
        self.root, self.cfg = root, cfg
        self.validate_for_pick = validate_for_pick
        self.handoff = handoff
        self.intrinsics, self.intrinsic_hash = load_intrinsics(intrinsics_path)
        self.size = tuple(self.intrinsics["image_size"])
        self.matrix = np.asarray(self.intrinsics["camera_matrix"], np.float64)
        self.distortion = np.asarray(self.intrinsics["distortion_coefficients"], np.float64)
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(
            self.matrix, self.distortion, None, self.matrix, self.size, cv2.CV_32FC1)
        self.output = ROOT / "calibration" / datetime.now().strftime("robot_xy_%Y%m%d_%H%M%S_%f")
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "camera_calibration.json").write_bytes(Path(intrinsics_path).read_bytes())
        self.samples, self.result = [], None
        self.cap = self.frame = self.selected = None
        self.frozen = False
        self.busy = False
        self.closed = False
        self.mailbox = queue.Queue()
        self.photo = None
        self.camera = tk.StringVar(value=str(camera))
        self.session_camera_index = int(camera)
        self.view_mode = tk.StringVar(value="Camera closed")
        self.role = tk.StringVar(value="fit")
        self.status = tk.StringVar(value="Open camera, click a visible mark, then jog TCP to the same mark.")
        self.threshold = tk.StringVar(value=str(cfg.get("calibration_max_error_mm", 2)))
        self.fields = {a: tk.StringVar() for a in "xyzr"}
        self.counts = tk.StringVar(value="Fit points: 0 / 9")
        root.title("MG400 Camera-to-Robot XY Calibration - No Automatic Motion")
        root.geometry("1320x850")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.build_ui(intrinsics_path)
        self.update_counts()
        root.after(40, self.tick)

    def build_ui(self, path):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text=f"Intrinsics: {path}", wraplength=1250).pack(anchor="w")
        ttk.Label(top, text=f"Resolution: {self.size[0]} x {self.size[1]} | "
                  f"User {self.cfg['user_index']} | Tool {self.cfg['tool_index']} | "
                  "Undistorted pixels, original K, no crop").pack(anchor="w")
        ttk.Label(top, textvariable=self.view_mode, foreground="#075e93").pack(anchor="w", pady=(4, 0))
        middle = ttk.Frame(self.root, padding=10)
        middle.pack(fill="both", expand=True)
        scale = min(900 / self.size[0], 510 / self.size[1], 1)
        self.display_size = (max(1, round(self.size[0] * scale)), max(1, round(self.size[1] * scale)))
        self.canvas = tk.Canvas(middle, width=self.display_size[0], height=self.display_size[1],
                                background="#17283b", highlightthickness=0)
        self.canvas.pack(side="left", anchor="n")
        self.canvas.bind("<Button-1>", self.on_click)
        controls = ttk.Frame(middle, padding=(15, 0))
        controls.pack(side="left", fill="both", expand=True)
        row = ttk.Frame(controls); row.pack(fill="x")
        ttk.Label(row, text="Camera index:").pack(side="left")
        ttk.Entry(row, textvariable=self.camera, width=5).pack(side="left", padx=5)
        ttk.Button(controls, text="1. Open camera", command=self.open_camera).pack(fill="x", pady=5)
        ttk.Button(controls, text="Load saved points.csv", command=self.load_points).pack(fill="x", pady=5)
        ttk.Button(controls, text="Show Live / Refresh", command=self.show_live).pack(fill="x", pady=5)
        self.freeze_button = ttk.Button(controls, text="Freeze Image", command=self.toggle_freeze)
        self.freeze_button.pack(fill="x")
        ttk.Label(controls, text="Click the mark before moving the robot.\n"
                  "The image freezes automatically.\nJog using DobotStudio Pro, then release Jog.",
                  wraplength=340).pack(anchor="w", pady=10)
        roles = [("Fit point (collect 9)", "fit"), ("Inspect converted XY (no motion)", "inspect")]
        if self.validate_for_pick:
            roles.insert(1, ("Independent validation point (collect 3)", "validation"))
        for text, value in roles:
            ttk.Radiobutton(controls, text=text, variable=self.role, value=value).pack(anchor="w")
        self.read_button = ttk.Button(controls, text="2. Read stationary robot X/Y/Z/R", command=self.read_pose)
        self.read_button.pack(fill="x", pady=10)
        ttk.Label(controls, text="Or enter the actual pose from the official software:").pack(anchor="w")
        for axis in "xyzr":
            row = ttk.Frame(controls); row.pack(fill="x", pady=2)
            ttk.Label(row, text=axis.upper(), width=3).pack(side="left")
            ttk.Entry(row, textvariable=self.fields[axis], width=22).pack(side="left")
        ttk.Button(controls, text="3. Add matched pair", command=self.add_pair).pack(fill="x", pady=8)
        ttk.Label(controls, textvariable=self.counts).pack(anchor="w")
        row = ttk.Frame(controls); row.pack(fill="x", pady=5)
        ttk.Label(row, text="Max error (mm):").pack(side="left")
        ttk.Entry(row, textvariable=self.threshold, width=7).pack(side="left", padx=5)
        ttk.Button(controls, text="4. Fit and Validate" if self.validate_for_pick else "4. Compute H + preview XY",
                   command=self.fit).pack(fill="x", pady=5)
        ttk.Button(controls, text="Delete selected pair", command=self.delete_pair).pack(fill="x")
        columns = ("id", "role", "u", "v", "x", "y", "z", "r")
        self.table = ttk.Treeview(self.root, columns=columns, show="headings", height=7)
        for column in columns:
            self.table.heading(column, text=column.upper())
            self.table.column(column, width=110, anchor="center")
        self.table.pack(fill="x", padx=10)
        ttk.Label(self.root, textvariable=self.status, wraplength=1250,
                  foreground="#064d7c", padding=10).pack(fill="x")
        ttk.Label(self.root, text=f"Saved session: {self.output}", wraplength=1250,
                  padding=(10, 0, 10, 10)).pack(fill="x")

    def open_camera(self):
        try:
            if self.busy:
                return
            index = int(self.camera.get())
            if index < 0:
                raise ValueError("Camera index must be zero or greater")
            if self.samples and index != self.session_camera_index:
                raise ValueError(f"These points belong to camera {self.session_camera_index}. "
                                 "Use that index to refresh this session.")
            if self.cap:
                self.cap.release()
                self.cap = None
            backend = cv2.CAP_DSHOW if __import__("os").name == "nt" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(index, backend)
            if not self.cap.isOpened():
                self.cap.release(); self.cap = None
                raise ValueError("Cannot open camera; close other camera applications or try another index")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.size[1])
            self.session_camera_index = index
            self.frame = self.selected = None
            self.canvas.delete("frame")
            self.photo = None
            for value in self.fields.values():
                value.set("")
            self.frozen = False
            self.status.set("Live undistorted image. Click a mark to freeze and select it.")
        except (ValueError, OSError, cv2.error) as exc:
            messagebox.showerror("Camera", str(exc))

    def show_live(self):
        if self.busy:
            return
        if self.cap is None:
            self.open_camera()
            return
        self.frozen = False
        self.selected = None
        for value in self.fields.values():
            value.set("")
        self.status.set("Live image refreshing. Click the current coin or mark to select it.")

    def tick(self):
        if self.closed:
            return
        if self.cap is None:
            self.view_mode.set("Saved image — click Show Live / Refresh" if self.frame is not None else "Camera closed")
        else:
            self.view_mode.set("FROZEN — click Show Live / Refresh to update" if self.frozen else "LIVE — camera updating")
        self.freeze_button.config(text="Resume Live" if self.frozen else "Freeze Image")
        if self.cap is not None and not self.frozen:
            try:
                ok, raw = self.cap.read()
                if not ok:
                    raise ValueError("Camera read failed")
                if (raw.shape[1], raw.shape[0]) != self.size:
                    raise ValueError("Actual resolution differs from camera calibration; no resizing is allowed")
                self.frame = cv2.remap(raw, self.map_x, self.map_y, cv2.INTER_LINEAR)
            except (ValueError, cv2.error) as exc:
                self.cap.release(); self.cap = None
                self.status.set(str(exc))
        try:
            ok, value = self.mailbox.get_nowait()
            self.busy = False
            self.read_button.config(state="normal")
            if ok:
                for axis, number in zip("xyzr", value):
                    self.fields[axis].set(f"{number:.6f}")
                self.status.set("Stationary pose read. Confirm the TCP is on the selected mark, then Add matched pair.")
            else:
                messagebox.showerror("Robot pose", str(value))
        except queue.Empty:
            pass
        if self.frame is not None:
            display = self.frame.copy()
            for p in self.samples:
                color = (40, 190, 40) if p["role"] == "fit" else (0, 150, 255)
                xy = (round(p["u"]), round(p["v"]))
                cv2.circle(display, xy, 7, color, 2)
                cv2.putText(display, str(p["id"]), (xy[0] + 8, xy[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if self.selected is not None:
                cv2.drawMarker(display, tuple(map(round, self.selected)), (255, 80, 0),
                               cv2.MARKER_CROSS, 22, 2)
            display = cv2.resize(display, self.display_size)
            ok, png = cv2.imencode(".png", display)
            if ok:
                self.photo = tk.PhotoImage(data=png.tobytes(), format="png")
                self.canvas.delete("frame")
                self.canvas.create_image(0, 0, anchor="nw", image=self.photo, tags="frame")
        self.root.after(50, self.tick)

    def toggle_freeze(self):
        if self.busy:
            return
        if self.frozen or self.frame is None:
            self.show_live()
            return
        self.frozen = True
        self.selected = None
        for value in self.fields.values():
            value.set("")
        self.status.set("Frozen: click a mark, or Show Live / Refresh to update the image.")

    def on_click(self, event):
        if self.frame is None or self.busy:
            return
        width, height = self.display_size
        if not (0 <= event.x < width and 0 <= event.y < height):
            return
        self.frozen = True
        self.selected = [event.x * self.size[0] / width, event.y * self.size[1] / height]
        for value in self.fields.values():
            value.set("")
        if self.role.get() == "inspect":
            try:
                if self.result is None:
                    raise ValueError("Compute H first")
                xy = pixel_to_robot(self.selected, self.result, pixel_space="undistorted", allow_preview=True)
                self.status.set(f"PREVIEW: X={xy[0]:.3f}, Y={xy[1]:.3f} mm. "
                                "No independent validation; no movement is sent.")
            except ValueError as exc:
                self.status.set(str(exc))
        else:
            self.status.set(f"Selected u={self.selected[0]:.2f}, v={self.selected[1]:.2f}. "
                            "Jog TCP to this same physical mark, then read the pose.")

    def read_pose(self):
        if self.busy:
            return
        if self.selected is None or self.role.get() == "inspect":
            messagebox.showinfo("Select point", "Click a fit/validation mark in the image first")
            return
        self.busy = True
        self.read_button.config(state="disabled")
        self.status.set("Reading robot pose only. No robot motion commands are sent.")
        def worker():
            try:
                pose = read_robot_pose(self.cfg["robot_ip"], self.cfg["user_index"], self.cfg["tool_index"])
                self.mailbox.put((True, pose))
            except Exception as exc:
                self.mailbox.put((False, exc))
        threading.Thread(target=worker, daemon=True).start()

    def changed(self):
        self.result = None
        save_samples(self.output / "points.csv", self.samples)
        existing = self.output / "robot_xy_mapping.json"
        if existing.exists():
            old = json.loads(existing.read_text(encoding="utf-8"))
            old["passed"] = False
            old["invalidated_reason"] = "Samples changed; fit and validate again"
            existing.write_text(json.dumps(old, indent=2), encoding="utf-8")
        self.table.delete(*self.table.get_children())
        for p in self.samples:
            self.table.insert("", "end", iid=str(p["id"]), values=[p["id"], p["role"],
                              *(f"{p[a]:.3f}" for a in ("u", "v", "x", "y", "z", "r"))])
        self.update_counts()

    def update_counts(self):
        fit_count = sum(p["role"] == "fit" for p in self.samples)
        validation_count = sum(p["role"] == "validation" for p in self.samples)
        text = f"Fit points: {fit_count} / 9"
        if self.validate_for_pick:
            text += f" | Validation: {validation_count} / 3"
        self.counts.set(text)

    def load_points(self):
        if self.busy:
            return
        selected = filedialog.askopenfilename(title="Load your saved points.csv",
                                              initialdir=str(ROOT / "calibration"),
                                              filetypes=[("Point CSV", "*.csv")])
        if not selected:
            return
        try:
            path = Path(selected)
            source_intrinsics = path.parent / "camera_calibration.json"
            _, digest = load_intrinsics(source_intrinsics)
            if digest != self.intrinsic_hash:
                raise ValueError("Select the camera_calibration.json saved beside this points.csv first")
            samples = load_samples(path)
            for p in samples:
                validate_sample(p, self.size)
            if self.validate_for_pick and (issues := pose_consistency_issues(samples)):
                raise ValueError("\n".join(issues) + "\nStart a new set of points for pickup.")
            if len({str(p['id']) for p in samples}) != len(samples):
                raise ValueError("Duplicate point IDs in CSV")
            for snapshot in (path.parent / "snapshots").glob("*.png"):
                target = self.output / "snapshots" / snapshot.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if snapshot.resolve() != target.resolve():
                    shutil.copy2(snapshot, target)
            images = sorted((self.output / "snapshots").glob("*.png"))
            if images:
                image = cv2.imdecode(np.fromfile(str(images[-1]), dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is not None and (image.shape[1], image.shape[0]) == self.size:
                    self.frame = image
                    self.frozen = True
            self.samples = samples
            self.selected = None
            self.changed()
            self.status.set(f"Loaded {len(samples)} saved points. Click Compute H + preview XY.")
        except (ValueError, OSError, KeyError, cv2.error) as exc:
            messagebox.showerror("Load saved points", str(exc))

    def add_pair(self):
        try:
            if self.busy:
                return
            if self.selected is None or self.role.get() not in ("fit", "validation"):
                raise ValueError("Select a fit or validation mark first")
            pose = finite([self.fields[a].get() for a in "xyzr"], 4)
            sample = {"id": max((int(p["id"]) for p in self.samples), default=0) + 1,
                      "role": self.role.get(), "u": self.selected[0], "v": self.selected[1],
                      **dict(zip("xyzr", map(float, pose)))}
            validate_sample(sample, self.size)
            if self.validate_for_pick and (issues := pose_consistency_issues([*self.samples, sample])):
                first = self.samples[0]
                raise ValueError("\n".join(issues) + f"\nFirst point: Z={first['z']:.3f} mm, R={first['r']:.3f}°.")
            for previous in self.samples:
                if (np.linalg.norm(np.array(self.selected) - [previous["u"], previous["v"]]) < 2
                        or np.linalg.norm(pose[:2] - [previous["x"], previous["y"]]) < 0.5):
                    raise ValueError("This duplicates an existing point; choose another physical mark")
            self.samples.append(sample)
            self.changed()
            snapshot = self.output / "snapshots" / f"point_{sample['id']:03d}.png"
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            ok, encoded = cv2.imencode(".png", self.frame)
            if not ok:
                raise ValueError("Could not encode point snapshot")
            encoded.tofile(str(snapshot))
            self.selected = None
            for value in self.fields.values():
                value.set("")
            self.status.set("Pair saved. Click another mark in the frozen image, or return to live.")
        except (ValueError, OSError, cv2.error) as exc:
            messagebox.showerror("Matched pair", str(exc))

    def delete_pair(self):
        if self.busy:
            return
        ids = set(self.table.selection())
        if ids:
            self.samples = [p for p in self.samples if str(p["id"]) not in ids]
            self.changed()

    def fit(self):
        try:
            if self.busy:
                return
            result = fit_mapping(self.samples, self.intrinsics, self.intrinsic_hash,
                                 self.cfg["user_index"], self.cfg["tool_index"], float(self.threshold.get()),
                                 min_validation=3 if self.validate_for_pick else 0,
                                 preview_only=not self.validate_for_pick)
            result.update({"created_at": datetime.now().astimezone().isoformat(),
                           "camera_index": int(self.camera.get())})
            path = self.output / "robot_xy_mapping.json"
            path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            self.result = result
            message = (f"PREVIEW: fit max {result['max_fit_error_mm']:.3f} mm, "
                       f"RMS {result['rms_fit_error_mm']:.3f} mm. No independent validation. "
                       + "; ".join(result["quality_issues"]) + f" Saved: {path}")
            if self.validate_for_pick:
                message = (f"{'PASS' if result['passed'] else 'FAIL'}. Independent XY error: "
                           f"{result['max_validation_error_mm']:.3f} mm maximum. Saved to {path}")
            self.status.set(message)
            self.role.set("inspect")
        except (ValueError, OSError, cv2.error) as exc:
            messagebox.showerror("Fit and validate", str(exc))

    def close(self):
        if self.handoff is not None and self.result is not None and self.result.get("passed") is True:
            try:
                self.handoff.write_text(str(self.output / "robot_xy_mapping.json"), encoding="utf-8")
            except OSError as exc:
                messagebox.showerror("Calibration result", str(exc))
                return
        self.closed = True
        if self.cap:
            self.cap.release()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intrinsics", type=Path)
    parser.add_argument("--camera", type=int)
    parser.add_argument("--validate-for-pick", action="store_true", help="Collect independent points for a motion mapping")
    parser.add_argument("--handoff", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--fit-csv", type=Path, help="Offline fitting from saved points.csv; no camera or robot")
    parser.add_argument("--output", type=Path, help="Output JSON for --fit-csv")
    parser.add_argument("--max-error-mm", type=float, default=2.0)
    args = parser.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8-sig"))
    if type(cfg.get("user_index")) is not int or cfg["user_index"] != 0:
        raise SystemExit("This program requires user_index=0 in config.json")
    if type(cfg.get("tool_index")) is not int or not 0 <= cfg["tool_index"] <= 9:
        raise SystemExit("Set the calibrated tool_index in config.json")
    if args.fit_csv:
        if not args.intrinsics or not args.output:
            parser.error("--fit-csv requires --intrinsics and --output")
        intrinsics, digest = load_intrinsics(args.intrinsics)
        result = fit_mapping(load_samples(args.fit_csv), intrinsics, digest, cfg["user_index"],
                             cfg["tool_index"], args.max_error_mm, min_validation=0, preview_only=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(f"PREVIEW: fit max {result['max_fit_error_mm']:.3f} mm; "
              f"RMS {result['rms_fit_error_mm']:.3f} mm. No independent validation. {args.output}")
        return
    root = tk.Tk()
    root.withdraw()
    if args.intrinsics is None:
        candidates = sorted((ROOT / "calibration").glob("session_*/camera_calibration.json"))
        selected = filedialog.askopenfilename(title="Select your camera_calibration.json",
                                              initialdir=str(candidates[-1].parent if candidates else ROOT / "calibration"),
                                              filetypes=[("Camera calibration JSON", "*.json")])
        if not selected:
            root.destroy()
            return
        args.intrinsics = Path(selected)
    try:
        app = MappingApp(root, args.intrinsics, cfg,
                         args.camera if args.camera is not None else cfg.get("camera_index", 0),
                         validate_for_pick=args.validate_for_pick, handoff=args.handoff)
        root.deiconify()
        if app.intrinsics.get("quality_review", {}).get("needs_review"):
            messagebox.showwarning("Camera calibration", "The lens calibration has quality warnings. Review it before using this mapping.")
        root.mainloop()
    except (ValueError, OSError, cv2.error, tk.TclError) as exc:
        messagebox.showerror("Camera-to-robot calibration", str(exc))
        root.destroy()


if __name__ == "__main__":
    main()

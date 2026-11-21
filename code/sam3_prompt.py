"""Standalone SAM3 text-prompt segmentation for camera frames and image files.
No robot drivers, camera calibration, coordinate mapping or robot commands.
GUI: run directly. CLI: --model sam3.pt --image photo.jpg --prompt coin.
"""
import argparse
import hashlib
import json
import math
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from sam3_runtime import ensure_environment

if __name__ == "__main__":
    ensure_environment(Path(__file__).resolve())

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit("Install dependencies in this interpreter: python -m pip install opencv-python numpy") from exc

ROOT = Path(__file__).resolve().parents[1]


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot read the selected image")
    return image


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode {path.name}")
    encoded.tofile(str(path))


def render_masks(image, masks, confidences):
    """Keep outputs at original image resolution; IDs start at 1."""
    height, width = image.shape[:2]
    overlay, objects, binary_masks = image.copy(), [], []
    palette = [(50, 210, 80), (240, 140, 50), (60, 190, 240), (190, 70, 220), (220, 200, 60)]
    for index, mask in enumerate(masks):
        raw = np.asarray(mask)
        if raw.ndim != 2:
            raise ValueError("SAM3 returned an unexpected mask shape")
        binary = cv2.resize(raw.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST) > 0.5
        ys, xs = np.where(binary)
        if not len(xs):
            continue
        object_id = len(objects) + 1
        color = palette[(object_id - 1) % len(palette)]
        overlay[binary] = (0.55 * overlay[binary] + 0.45 * np.asarray(color)).astype(np.uint8)
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        score = (float(confidences[index])
                 if index < len(confidences) and confidences[index] is not None else None)
        if score is not None and not math.isfinite(score):
            score = None
        objects.append({"id": object_id, "confidence": score,
                        "area_pixels": int(binary.sum()), "bbox_xyxy_pixels": bbox})
        binary_masks.append(binary.astype(np.uint8) * 255)
        label = f"#{object_id}" + (f" {score:.2f}" if score is not None else "")
        scale = max(0.6, min(2, width / 1500))
        cv2.putText(overlay, label, (bbox[0], max(20, bbox[1] - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
        cv2.putText(overlay, label, (bbox[0], max(20, bbox[1] - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
    return overlay, objects, binary_masks


class Segmenter:
    def __init__(self):
        self.predictor = None
        self.model_key = None
        self.image_key = None

    def predict(self, image, weights, prompt, confidence=0.35, device="auto", status=lambda text: None):
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Enter a text prompt, e.g. coin, cup, red block")
        if not math.isfinite(confidence) or not 0 < confidence <= 1:
            raise ValueError("Confidence must be greater than 0 and at most 1")
        weights = Path(weights).expanduser().resolve()
        if not weights.is_file() or weights.stat().st_size < 1_000_000:
            raise ValueError("Choose the complete local sam3.pt file, not a link or Git LFS pointer")
        try:
            import torch
            from ultralytics.models.sam import SAM3SemanticPredictor
        except ImportError as exc:
            raise RuntimeError(f"SAM3 dependencies could not be imported from {sys.executable}. "
                               "Run code/sam3_runtime.py --install with your selected Python. "
                               f"Import error: {exc}") from exc
        selected = "0" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        if selected != "cpu" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this interpreter; select Auto or CPU")
        model_key = (str(weights), weights.stat().st_mtime_ns, selected)
        class LocalCheckpointPredictor(SAM3SemanticPredictor):
            def get_model(self):
                # The installed builder opens the checkpoint as a file object.
                # Map its filename instead of allocating another full in-RAM
                # checkpoint copy. The patch is temporary; package files remain unchanged.
                from unittest.mock import patch
                from ultralytics.models.sam import build_sam3
                original_load = build_sam3.torch_load
                def mapped_load(file, *args, **kwargs):
                    filename = getattr(file, "name", file)
                    kwargs.update(map_location="cpu", mmap=True)
                    return original_load(filename, *args, **kwargs)
                with patch.object(build_sam3, "torch_load", mapped_load):
                    return super().get_model()
        started = time.perf_counter()
        if self.predictor is None or self.model_key != model_key:
            self.predictor = None
            self.image_key = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            status("Loading local SAM3 model (first run may take a while)...")
            self.predictor = LocalCheckpointPredictor(overrides={
                "model": str(weights), "task": "segment", "mode": "predict",
                "conf": confidence, "device": selected, "half": selected != "cpu",
                "save": False, "verbose": False,
            })
            self.model_key = model_key
        self.predictor.args.conf = confidence
        image_key = (image.shape, hashlib.sha256(image.tobytes()).hexdigest())
        if image_key != self.image_key:
            status("Encoding image...")
            self.predictor.set_image(image)
            self.image_key = image_key
        status(f"Recognizing prompt: {prompt}")
        results = self.predictor(text=[prompt])
        masks, scores = [], []
        for result in results or []:
            if result.masks is None:
                continue
            current = result.masks.data.detach().cpu().numpy()
            current_scores = (result.boxes.conf.detach().cpu().numpy().tolist()
                              if result.boxes is not None else [None] * len(current))
            masks.extend(current)
            scores.extend(current_scores)
        overlay, objects, binary = render_masks(image, masks, scores)
        metadata = {"prompt": prompt, "confidence_threshold": confidence,
                    "model_file": weights.name, "device": selected,
                    "image_size": [image.shape[1], image.shape[0]],
                    "image_pixel_sha256": image_key[1], "object_count": len(objects),
                    "elapsed_seconds": time.perf_counter() - started, "objects": objects,
                    "notes": ["Image segmentation only; no robot or coordinate mapping.",
                              "Prompt matches are model predictions, not verified labels."]}
        return overlay, binary, metadata


def export_result(folder, overlay, masks, metadata):
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("Result folder is not empty; choose a new folder to avoid mixing old masks")
    folder.mkdir(parents=True, exist_ok=True)
    write_image(folder / "segmentation.png", overlay)
    for obj, mask in zip(metadata["objects"], masks):
        name = f"masks/object_{obj['id']:03d}.png"
        write_image(folder / name, mask)
        obj["mask_file"] = name
    (folder / "result.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return folder


def default_weights():
    path = ROOT / "models" / "sam3.pt"
    return str(path) if path.is_file() else ""


class PromptApp:
    DEVICES = {"Auto": "auto", "CPU": "cpu", "GPU (CUDA)": "0"}

    def __init__(self, root, model):
        self.root = root
        self.segmenter = Segmenter()
        self.image = self.overlay = self.photo = None
        self.image_path = None
        self.last_result = None
        self.busy = False
        self.closed = False
        self.capture = None
        self.continuous = False
        self.mailbox = queue.Queue()
        self.model = tk.StringVar(value=model or default_weights())
        self.prompt = tk.StringVar(value="coin")
        self.confidence = tk.StringVar(value="0.35")
        self.device = tk.StringVar(value="Auto")
        self.camera_index = tk.StringVar(value="0")
        self.original = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Open the camera or choose an image, then enter a prompt and start detection.")
        root.title("SAM3 Object Segmentation")
        root.geometry("1150x850")
        root.protocol("WM_DELETE_WINDOW", self.close)
        controls = ttk.Frame(root, padding=10); controls.pack(fill="x")
        ttk.Label(controls, text="SAM3 model").grid(row=0, column=0, sticky="w")
        self.model_entry = ttk.Entry(controls, textvariable=self.model)
        self.model_entry.grid(row=0, column=1, columnspan=4, sticky="ew", padx=5)
        self.model_button = ttk.Button(controls, text="Choose sam3.pt", command=self.choose_model)
        self.model_button.grid(row=0, column=5)
        self.image_button = ttk.Button(controls, text="Select Image", command=self.choose_image)
        self.image_button.grid(row=1, column=0, pady=10, sticky="ew")
        ttk.Label(controls, text="Text prompt (English recommended)").grid(row=1, column=1, sticky="e")
        self.prompt_entry = ttk.Entry(controls, textvariable=self.prompt, width=35)
        self.prompt_entry.grid(row=1, column=2, columnspan=3, sticky="ew", padx=5)
        self.prompt_entry.bind("<Return>", lambda event: self.run())
        self.run_button = ttk.Button(controls, text="Detect Objects", command=self.run)
        self.run_button.grid(row=1, column=5, sticky="ew")
        ttk.Label(controls, text="Confidence threshold").grid(row=2, column=0)
        self.conf_entry = ttk.Entry(controls, textvariable=self.confidence, width=8)
        self.conf_entry.grid(row=2, column=1, sticky="w", padx=5)
        ttk.Label(controls, text="Device").grid(row=2, column=2, sticky="e")
        self.device_box = ttk.Combobox(controls, textvariable=self.device, values=tuple(self.DEVICES), state="readonly", width=12)
        self.device_box.grid(row=2, column=3, sticky="w")
        ttk.Checkbutton(controls, text="Show Original", variable=self.original, command=self.render).grid(row=2, column=4)
        self.save_button = ttk.Button(controls, text="Save Results", command=self.save, state="disabled")
        self.save_button.grid(row=2, column=5)
        self.camera_button = ttk.Button(controls, text="Open Camera", command=self.toggle_camera)
        self.camera_button.grid(row=3, column=0, sticky="ew", pady=8)
        camera_options = ttk.Frame(controls)
        camera_options.grid(row=3, column=1, columnspan=4, sticky="w", padx=5)
        ttk.Label(camera_options, text="Camera index").pack(side="left")
        self.camera_entry = ttk.Entry(camera_options, textvariable=self.camera_index, width=5)
        self.camera_entry.pack(side="left", padx=5)
        ttk.Label(camera_options, text="Live feed on the left, latest result on the right").pack(side="left", padx=10)
        controls.columnconfigure(2, weight=1)
        self.canvas = tk.Canvas(root, width=1050, height=580, background="#16293b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10)
        self.canvas.bind("<Configure>", lambda event: self.render())
        ttk.Label(root, textvariable=self.status, wraplength=1080, padding=10).pack(fill="x")
        ttk.Label(root, text="Try prompts such as coin, cup, red block, or screw.",
                  padding=(10, 0, 10, 10)).pack(fill="x")
        root.after(100, self.poll)
        root.after(60, self.camera_tick)

    def toggle_camera(self):
        if self.capture is not None:
            self.stop_camera()
            self.status.set("Camera closed.")
            return
        if self.busy:
            return
        try:
            index = int(self.camera_index.get())
            if index < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Camera", "Enter a camera index of 0 or higher.")
            return
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if not capture.isOpened():
            capture.release()
            messagebox.showerror("Camera", f"Could not open camera {index}. Check the index and close other apps using the camera.")
            return
        self.capture = capture
        self.image = self.overlay = self.last_result = None
        self.canvas.delete("all")
        self.image_path = f"camera:{index}"
        self.original.set(False)
        self.camera_button.config(text="Close Camera")
        self.camera_entry.config(state="disabled")
        self.set_busy(False)
        self.status.set("Camera ready. Enter a prompt and click Start Live Detection.")

    def stop_camera(self):
        self.continuous = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.camera_button.config(text="Open Camera")
        self.camera_entry.config(state="normal")
        self.set_busy(self.busy)
        self.render()

    def camera_tick(self):
        if self.closed:
            return
        if self.capture is not None:
            ok, frame = self.capture.read()
            if ok:
                self.image = frame
                self.render()
                if self.continuous and not self.busy:
                    self.infer()
            else:
                self.stop_camera()
                self.status.set("Could not read a camera frame. Reopen the camera.")
        self.root.after(60, self.camera_tick)

    def choose_model(self):
        path = filedialog.askopenfilename(title="Select a local SAM3 model", filetypes=[("SAM3 checkpoint", "*.pt")])
        if path:
            self.model.set(path)

    def choose_image(self):
        path = filedialog.askopenfilename(title="Select an image", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if path:
            try:
                image = read_image(path)
                self.stop_camera()
                self.image = image
                self.image_path = path
                self.overlay = self.last_result = None
                self.save_button.config(state="disabled")
                self.original.set(False)
                self.status.set(f"Loaded {path}. Enter a prompt and click Detect Objects.")
                self.render()
            except (ValueError, OSError, cv2.error) as exc:
                messagebox.showerror("Image", str(exc))

    def render(self):
        if self.closed or self.image is None:
            return
        width, height = max(100, self.canvas.winfo_width()), max(100, self.canvas.winfo_height())
        if self.capture is not None and not self.original.get():
            panels = [(self.image, "Live Camera"), (self.overlay, "Latest Detection")]
        else:
            panels = [(self.image if self.original.get() or self.overlay is None else self.overlay, "")]
        self.canvas.delete("all")
        self.photo = []
        panel_width = width // len(panels)
        for index, (image, label) in enumerate(panels):
            center = panel_width * index + panel_width // 2
            if label:
                self.canvas.create_text(center, 12, text=label, fill="white")
            if image is None:
                continue
            scale = min(panel_width / image.shape[1], (height - 24) / image.shape[0], 1)
            display = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
            ok, png = cv2.imencode(".png", display, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if ok:
                photo = tk.PhotoImage(data=png.tobytes(), format="png")
                self.photo.append(photo)
                self.canvas.create_image(center, (height + 24) // 2, image=photo, anchor="center",
                                         tags=("detection",) if image is self.overlay else ())

    def set_busy(self, busy):
        self.busy = busy
        state = "disabled" if busy else "normal"
        for widget in (self.image_button, self.model_button, self.model_entry,
                       self.prompt_entry, self.conf_entry):
            widget.config(state=state)
        self.run_button.config(text="Stop Live Detection" if self.continuous else
                               ("Start Live Detection" if self.capture is not None else "Detect Objects"),
                               state="normal" if self.continuous else state)
        self.camera_button.config(state="normal" if self.capture is not None else state)
        if self.continuous:
            self.prompt_entry.config(state="normal")
            self.conf_entry.config(state="normal")
            self.device_box.config(state="readonly")
        else:
            self.device_box.config(state="disabled" if busy else "readonly")
        self.save_button.config(state="normal" if self.last_result is not None else "disabled")

    def run(self):
        if self.continuous:
            self.continuous = False
            self.set_busy(self.busy)
            self.status.set("Stopping after the current frame finishes." if self.busy else "Live detection stopped.")
            return
        if self.busy:
            return
        if self.image is None:
            messagebox.showinfo("Image Source", "Open the camera or select an image first.")
            return
        self.continuous = self.capture is not None
        self.infer()

    def infer(self):
        try:
            confidence = float(self.confidence.get())
            if not math.isfinite(confidence) or not 0 < confidence <= 1:
                raise ValueError("Confidence must be greater than 0 and at most 1.")
            model, prompt = self.model.get(), self.prompt.get().strip()
            device = self.DEVICES[self.device.get()]
            if not prompt:
                raise ValueError("Enter a text prompt such as coin.")
            image = self.image.copy()
        except ValueError as exc:
            self.continuous = False
            self.set_busy(False)
            messagebox.showerror("Settings", str(exc)); return
        if self.capture is None:
            self.last_result = self.overlay = None
            self.render()
        self.set_busy(True)
        if self.last_result is None:
            self.status.set("Running detection. The first run includes model loading.")
        def worker():
            try:
                result = self.segmenter.predict(image, model, prompt, confidence, device,
                                                status=lambda value: self.mailbox.put(("status", value)))
                self.mailbox.put(("done", result))
            except Exception as exc:
                self.segmenter.predictor = None
                self.segmenter.image_key = None
                self.mailbox.put(("error", str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def poll(self):
        if self.closed:
            return
        while True:
            try:
                kind, value = self.mailbox.get_nowait()
            except queue.Empty:
                break
            if self.handle_message(kind, value):
                continue
            if kind == "status":
                if self.last_result is None:
                    self.status.set(value)
            elif kind == "done":
                if not self.accept_detection(value):
                    self.set_busy(False)
                    continue
                self.last_result = value
                self.overlay, _, metadata = value
                self.set_busy(False)
                device_label = "CPU" if metadata["device"] == "cpu" else f"GPU {metadata['device']}"
                self.status.set(f"Prompt: {metadata['prompt']} | Objects: {metadata['object_count']} | "
                                f"Time: {metadata['elapsed_seconds']:.2f} s | Device: {device_label}"
                                + (" | No matches. Try another prompt or lower the threshold." if metadata['object_count'] == 0 else ""))
                self.render()
            else:
                self.continuous = False
                self.last_result = self.overlay = None
                self.set_busy(False)
                self.status.set("Detection failed.")
                self.render()
                messagebox.showerror("SAM3", value + "\nIf CUDA runs out of memory, close other GPU apps or select CPU.")
        self.root.after(100, self.poll)

    def accept_detection(self, result):
        return True

    def handle_message(self, kind, value):
        return False

    def save(self):
        if self.last_result is None:
            return
        overlay, masks, metadata = self.last_result
        metadata["source_image"] = self.image_path
        parent = filedialog.askdirectory(title="Choose a folder for the results", initialdir=str(ROOT))
        if not parent:
            return
        try:
            folder = Path(parent) / datetime.now().strftime("sam3_%Y%m%d_%H%M%S_%f")
            export_result(folder, overlay, masks, metadata)
            self.status.set(f"Saved the overlay, object masks, and JSON to {folder}")
        except (ValueError, OSError, cv2.error) as exc:
            messagebox.showerror("Save Results", str(exc))

    def close(self):
        if self.capture is not None:
            self.stop_camera()
        if self.busy:
            self.status.set("Detection is still running. Wait for it to finish or interrupt the program in the terminal.")
            return
        self.closed = True
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--device", choices=("auto", "cpu", "0"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.image or args.prompt:
        if not args.image or not args.prompt:
            parser.error("CLI mode requires both --image and --prompt")
        image = read_image(args.image)
        overlay, masks, metadata = Segmenter().predict(image, args.model or default_weights(), args.prompt,
                                                      args.confidence, args.device, status=print)
        metadata["source_image"] = str(args.image)
        folder = args.output or ROOT / "sam3_results" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        export_result(folder, overlay, masks, metadata)
        print(f"Objects: {metadata['object_count']}; time: {metadata['elapsed_seconds']:.2f}s; saved: {folder}")
    else:
        root = tk.Tk()
        PromptApp(root, str(args.model) if args.model else None)
        root.mainloop()


if __name__ == "__main__":
    main()

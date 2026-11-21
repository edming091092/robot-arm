"""Portable SAM3 environment discovery and installation.

From the project folder, run with your available Python:
    python code/sam3_runtime.py --install
    python code/sam3_runtime.py --install --device cuda
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("torch", "torchvision", "ultralytics", "cv2", "numpy", "clip", "tkinter")


def environment_ready():
    if any(importlib.util.find_spec(name) is None for name in REQUIRED):
        return False
    try:
        version = importlib.metadata.version("ultralytics").split(".")[:3]
        return tuple(int(part) for part in version) >= (8, 4, 30)
    except (ValueError, importlib.metadata.PackageNotFoundError):
        return False


def environment_python(folder):
    return folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_environment(script):
    if environment_ready():
        return
    # Preserve discovery of the previously installed environment on the current
    # project drive while making the project-local venv the portable default.
    candidates = [environment_python(ROOT / ".venv"), Path(ROOT.anchor) / "sam3_env" / "python.exe"]
    probe = (f"import sys; sys.path.insert(0, {str(Path(__file__).parent)!r}); "
             "from sam3_runtime import environment_ready; sys.exit(0 if environment_ready() else 1)")
    for executable in candidates:
        if not executable.is_file() or executable.resolve() == Path(sys.executable).resolve():
            continue
        try:
            result = subprocess.run([str(executable), "-c", probe], capture_output=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            print(f"Starting SAM3 with {executable}", flush=True)
            raise SystemExit(subprocess.call([str(executable), str(script), *sys.argv[1:]]))
    command = subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()), "--install"])
    raise SystemExit(f"SAM3 packages are missing from {sys.executable}\n"
                     f"Run this command in the terminal, then start the app again:\n{command}")


def install(device):
    executable = environment_python(ROOT / ".venv")
    if not executable.is_file():
        if sys.implementation.name != "cpython" or not (3, 10) <= sys.version_info[:2] <= (3, 14):
            raise SystemExit("The selected PyTorch release supports CPython 3.10 through 3.14.")
        if struct.calcsize("P") != 8:
            raise SystemExit("Select a 64-bit Python interpreter.")
        folder = ROOT / ".venv"
        if folder.exists() and any(folder.iterdir()):
            raise SystemExit(f"{folder} contains an incomplete environment. Rename that folder and retry.")
        print(f"Creating {folder} using Python {sys.version.split()[0]}", flush=True)
        venv.EnvBuilder(with_pip=True).create(folder)
    version = subprocess.check_output([str(executable), "-c", "import sys,json; print(json.dumps(list(sys.version_info[:2])))"], text=True)
    if not (3, 10) <= tuple(json.loads(version)) <= (3, 14):
        raise SystemExit("The existing .venv needs a Python version compatible with PyTorch 2.11.")
    index = "https://download.pytorch.org/whl/" + ("cu128" if device == "cuda" else "cpu")
    commands = [
        ["--upgrade", "pip"],
        ["torch==2.11.0", "torchvision==0.26.0", "--index-url", index],
        ["ultralytics==8.4.30", "opencv-python", "numpy",
         "https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip"],
    ]
    for packages in commands:
        subprocess.run([str(executable), "-m", "pip", "install", *packages], check=True)
    check = ("import torch,torchvision,cv2,numpy,tkinter,clip; "
             "from ultralytics.models.sam import SAM3SemanticPredictor; "
             "print('SAM3 imports OK'); print('CUDA available:',torch.cuda.is_available())")
    subprocess.run([str(executable), "-c", check], check=True)
    print(f"Setup complete. Select this interpreter in VS Code:\n{executable}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.install:
        try:
            install(args.device)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"Installation failed with exit code {exc.returncode}. Read the pip error above and retry.") from exc
    else:
        print(f"Python {sys.version.split()[0]} at {sys.executable}")
        print(f"SAM3 packages available: {environment_ready()}")

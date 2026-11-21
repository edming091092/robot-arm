# Standalone SAM3 Text-Prompt Segmentation

The interface entry point is `code/sam3_prompt.py`. Its shared environment helper is `code/sam3_runtime.py`.

It loads a local SAM3 checkpoint and displays object masks matching a text prompt on camera frames or a selected image.

## Setup on Windows

Open the complete `robot arm` folder in VS Code. Select an installed 64-bit CPython 3.10, 3.11, 3.12, 3.13, or 3.14 interpreter. The setup helper creates `.venv` with that interpreter and lets pip select the corresponding platform wheels.

Run the following in a terminal opened in `robot arm`. Use the full path of your chosen Python executable if the `python` command is unavailable.

For CPU inference:

```powershell
python code/sam3_runtime.py --install
```

For an NVIDIA GPU with a CUDA 12.8-compatible driver:

```powershell
python code/sam3_runtime.py --install --device cuda
```

After installation, select `.venv/Scripts/python.exe` with **Python: Select Interpreter**, then run `code/sam3_prompt.py` or `code/integrated_pick/app.py`. Both entry points discover the project environment when the current interpreter lacks SAM3 packages. A complete current interpreter is used directly. The installer can be retried after a failed download.

Python 3.12 inference was tested on this PC. Python 3.13 startup and environment discovery were checked. Windows wheel availability was confirmed for Python 3.10-3.14. Full inference has not been tested on every Python version or GPU.

## Use the interface

For live recognition, select the camera index (start with `0`), click **Open Camera**, enter the prompt, and click **Start Live Detection**. The left panel shows the live camera and the right panel shows the most recent completed result. SAM3 processes one frame at a time and takes the latest available frame for each new prediction. **Stop Live Detection** stops submitting frames while the preview continues. **Close Camera** releases the camera.

The device menu offers **Auto**, **CPU**, and **GPU (CUDA)**. During continuous recognition, a device change applies to the next prediction after the current one finishes. The model reloads when the selected device changes.

For an image file:

1. Check the model path, or click the model browse button to select the complete local `sam3.pt` file. The default path is `models/sam3.pt` inside this project folder.
2. Click **Select Image**.
3. Enter a short descriptive prompt. English examples: `coin`, `cup`, `red block`, `screw`. Each run uses the entire entry as one concept; this is not a conversational instruction parser or an automatic translation service.
4. Click **Detect Objects**, or press Enter in the prompt entry.
5. Matching masks are colored and numbered. The status line reports the object count and elapsed time. A zero-object result is supported; try another prompt or lower the confidence threshold if appropriate.
6. Change the prompt and run again on the same image. The program reuses the loaded model and image features where possible.
7. Click **Save Results** to choose a destination. A fresh timestamped subfolder contains `segmentation.png`, individual binary PNG masks and `result.json`.

The first run includes model loading and image encoding, so it takes longer. Auto uses GPU 0 when CUDA is available and CPU otherwise. GPU inference uses FP16 to reduce memory use. CPU inference can be considerably slower. If GPU memory is insufficient, close other GPU programs or select CPU. Model predictions still require visual review.

Checkpoint loading uses PyTorch memory mapping to avoid an additional full in-RAM checkpoint copy. The temporary loader adjustment is contained in this program; installed package files are not changed. SAM3 still needs substantial system RAM and GPU memory; it is not a lightweight model.

## Command-line mode

```powershell
python code/sam3_prompt.py --model models/sam3.pt --image 'path\photo.png' --prompt 'red cup' --confidence 0.35 --device auto --output 'path\new_results_folder'
```

The output folder must be new or empty so old mask files cannot be confused with a new prediction. Source images are not overwritten.

## Other computers

Use a compatible Python environment with PyTorch, Ultralytics and OpenCV, plus Tkinter for the interface. Choose PyTorch according to that computer's GPU/driver; do not assume this PC's CUDA installation is portable. Obtain SAM3 weights through the official model access process and comply with the model's terms. The installed Ultralytics version must provide `SAM3SemanticPredictor`.

This PC uses Python 3.12.13, PyTorch 2.11.0+cu128, Ultralytics 8.4.30 and OpenCV 4.13.0 for SAM3. The interface can browse to any local checkpoint path; the D: convenience paths are not required on another computer.

Official text-prompt API: https://docs.ultralytics.com/models/sam-3/

## Validation on this PC

The real local checkpoint was tested with the existing coin photograph and the prompt `coin`. It produced 30 candidate object masks, an overlay and JSON at `sam3_results/test_coin`. The first run took approximately 80.56 seconds including model loading/encoding. This confirms execution, not a manually verified object-detection accuracy score.

The interface was also checked using a hidden Tk window for image loading, asynchronous results, saving, zero-result rendering, Unicode paths and stale-result clearing after a failed run. No robot modules or motion commands are used.

Memory-mapped checkpoint loading reference: https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html

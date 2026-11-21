# Complete robot arm folder

Download **all** `robot-arm-full.7z.00x` files from this release into the same folder.
Open `robot-arm-full.7z.001` with 7-Zip and extract it. Do not extract the volumes separately.

The archive contains the complete `robot arm` directory: SAM3 weights at
`robot arm/models/sam3.pt`, source code, all recorded calibration sessions,
images, results, logs, manuals, installers and the original Python cache files.
The source repository shows code and files below GitHub's ordinary file-size
limit. Large files and caches are included in the complete release archive.

After extraction, open `robot arm/code/sam3_robot_move/app.py` in VS Code.
Install the Python dependencies on the destination computer first; the
archive does not contain a Python installation or copied virtual environment.
Calibration belongs to the physical camera/robot setup used to record it.

`SHA256SUMS.txt` verifies each archive volume. `FULL_FOLDER_MANIFEST.json`
records the path, byte length and SHA256 of every original file, including
the model. GitHub's automatic Source code ZIP does not contain large binaries.

Optional bulk download with authenticated GitHub CLI:

gh release download complete-folder-2026-11-21 --repo edming091092/robot-arm --pattern "robot-arm-full.7z.*" --pattern "SHA256SUMS.txt" --dir robot-arm-download

Use gh auth login first if required. Then extract robot-arm-full.7z.001 with 7-Zip.

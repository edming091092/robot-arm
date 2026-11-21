"""English GUI: SAM3 text detection -> calibrated robot XY -> user-entered Z.

Run this file in VS Code. Shared detection, mapping and motion code stays in
the parent project rather than duplicating a second implementation here.
"""
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))


def main():
    from sam3_runtime import ensure_environment
    ensure_environment(Path(__file__).resolve())

    import tkinter as tk
    from tkinter import ttk
    from integrated_pick.app import PickApp

    root = tk.Tk()
    app = PickApp(root, simple_mode=True)
    root.title("SAM3 - Camera to Robot Coordinates")
    ttk.Label(
        root,
        text=("1. Load XY Mapping   2. Open Camera   3. Enter a prompt and start live detection\n"
              "4. Enter Hover Z (robot coordinates, mm)   5. Click an object, then Move Above Selected"),
        padding=8,
    ).pack(fill="x", before=app.canvas)
    root.mainloop()


if __name__ == "__main__":
    main()

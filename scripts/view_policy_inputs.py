"""Live view of the two images the DROID policy receives (exterior + wrist, 224x224 after resize_with_pad).

Run alongside an episode started with ``run_droid_runtime.py --preview-dir DIR``:

    python scripts/view_policy_inputs.py DIR

Read-only: it only reads the files the runner writes, so it cannot affect the episode.
"""
import argparse
import json
from pathlib import Path
import time
import tkinter as tk


def load_latest(directory):
    """(meta, {camera key: PIL image}) for the newest complete write, or None."""
    from PIL import Image
    try:
        meta = json.loads((directory/"latest.json").read_text())
        images = {}
        for camera in meta["cameras"]:
            with Image.open(directory/camera["file"]) as image:
                images[camera["key"]] = image.convert("RGB").copy()
        return meta, images
    except (OSError, ValueError, KeyError):
        return None  # first write pending or a file replaced mid-read: try again next poll


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", type=Path)
    p.add_argument("--scale", type=int, default=2, help="display magnification (default 2: 448 px)")
    p.add_argument("--period-ms", type=int, default=100)
    args = p.parse_args()
    from PIL import ImageTk

    root = tk.Tk()
    root.title("DROID policy inputs")
    panels, captions = {}, {}
    status = tk.Label(root, text=f"waiting for {args.directory}/latest.json ...", anchor="w", justify="left")
    status.grid(row=2, column=0, columnspan=2, sticky="we", padx=6, pady=4)
    state = {"mtime": None, "photos": {}}

    def poll():
        try:
            mtime = (args.directory/"latest.json").stat().st_mtime_ns
        except OSError:
            mtime = None
        if mtime is not None and mtime != state["mtime"]:
            latest = load_latest(args.directory)
            if latest is not None:
                meta, images = latest
                state["mtime"] = mtime
                for column, (key, image) in enumerate(images.items()):
                    if key not in panels:
                        captions[key] = tk.Label(root, text=key.rsplit("/", 1)[-1])
                        captions[key].grid(row=0, column=column)
                        panels[key] = tk.Label(root)
                        panels[key].grid(row=1, column=column, padx=4)
                    size = (image.width*args.scale, image.height*args.scale)
                    state["photos"][key] = ImageTk.PhotoImage(image.resize(size))  # keep a reference
                    panels[key].configure(image=state["photos"][key])
                    captions[key].configure(text=f"{key.rsplit('/', 1)[-1]}  ({image.width}x{image.height})")
                state["meta"], state["shown_at"] = meta, time.monotonic_ns()
        meta = state.get("meta")
        if meta is not None:
            age = (time.monotonic_ns()-meta["offered_ns"])/1e9
            gripper = meta["state"][7] if len(meta["state"]) > 7 else float("nan")
            status.configure(text=f"step {meta['step']}   image age {age:.1f} s   gripper closure {gripper:.2f}\n"
                                  f"instruction: {meta['instruction']}")
        root.after(args.period_ms, poll)

    poll()
    root.mainloop()


if __name__ == "__main__":
    main()

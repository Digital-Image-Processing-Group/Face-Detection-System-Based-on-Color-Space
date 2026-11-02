"""Exercise the real Tk window using real evaluated photos and save window captures.

Only file-dialog answers are supplied programmatically; decoding, worker execution,
rendering, save command and Tk event loop are the actual application paths.
"""
import ctypes
from ctypes import wintypes
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import ImageGrab
import tkinter as tk
from src.app.gui import Application
from src.training.dataset import data_path, read_manifest, write_json, fingerprint


def main():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except OSError:
        pass
    report = json.loads((ROOT/"reports/evaluation.json").read_text(encoding="utf-8"))
    rows = report["results"]["full"]
    photos = {r.image_id: data_path(ROOT/"data/manifest.csv", r.image_path) for r in read_manifest(ROOT/"data/manifest.csv")}
    sources = json.loads((ROOT/"data/sources/wider_fair_downloads.json").read_text(encoding="utf-8"))["records"]
    photos.update({r["image_id"]: ROOT/"data"/r["image_path"] for r in sources})
    frontal = next(r for r in rows["test"]["per_image"] if "frontal" in r["scene_tags"])
    no_face = next(r for r in rows["validation"]["per_image"] if r["face"]["tp"]+r["face"]["fn"] == 0 and r["candidate_count"] == 0)
    multiple = next(r for r in rows["external_scenes"]["per_image"] if "multi_person" in r["scene_tags"] and r["candidate_count"] >= 2)
    folder = ROOT/"reports/gui"
    folder.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    app = Application(root, ROOT/"models/skin_model.json", ROOT/"config/pipeline.json")
    root.attributes("-topmost", True)
    checks = []

    def pump(predicate, timeout=30):
        started = time.monotonic()
        while not predicate():
            root.update()
            time.sleep(.02)
            if time.monotonic()-started > timeout:
                raise TimeoutError("GUI action did not finish")
        for _ in range(8):
            root.update()
            time.sleep(.03)

    def capture(name):
        root.update()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        rect = wintypes.RECT()
        if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            raise OSError("Cannot obtain native Tk window rectangle")
        target = folder/name
        ImageGrab.grab(bbox=(rect.left, rect.top, rect.right, rect.bottom), all_screens=True).save(target)
        return target

    try:
        for name, row in (("single", frontal), ("no_face", no_face), ("multiple", multiple)):
            with patch("src.app.gui.filedialog.askopenfilename", return_value=str(photos[row["image_id"]])):
                app.choose()
            pump(lambda: not app.busy)
            assert app.payload is not None
            assert len(app.photos) == 4
            assert len(app.payload[1]["candidates"]) == row["candidate_count"]
            screenshot = capture(name+".png")
            checks.append({"case": name, "real_image_id": row["image_id"], "candidate_count": row["candidate_count"],
                           "screenshot": screenshot.relative_to(ROOT).as_posix(), "screenshot_sha256": fingerprint(screenshot)})
            if name == "single":
                root.geometry("820x680")
                pump(lambda: root.winfo_width() == 820)
                capture("resized.png")
                root.geometry("1120x850")
                pump(lambda: root.winfo_width() == 1120)
                with patch("src.app.gui.filedialog.askdirectory", return_value=str(folder)):
                    app.save()
                pump(lambda: not app.busy)
                saved = [p for p in folder.iterdir() if p.is_dir() and p.name.startswith(photos[row["image_id"]].stem+"_")]
                assert saved
                target = sorted(saved)[-1]
                assert len(list(target.iterdir())) == 6
                checks[-1]["gui_saved_result"] = target.relative_to(ROOT).as_posix()
        write_json(ROOT/"reports/gui_verification.json", {"native_tk_window": True, "tk_version": tk.TkVersion,
            "checks": checks, "resize_capture": "reports/gui/resized.png", "save_command_six_files_verified": True,
            "qualification": "Programmatic real-Tk event-loop exercise with file-dialog choices supplied; screenshots visually reviewed separately. Not a manual click-through."})
    finally:
        app.close()


if __name__ == "__main__":
    main()

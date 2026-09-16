#!/usr/bin/env python3
"""
Packages BodyPlay into ONE file your customers can double-click.

    python build_app.py

Windows  -> dist/BodyPlay.exe
macOS    -> dist/BodyPlay      (and dist/BodyPlay.app if you add --windowed)
Linux    -> dist/BodyPlay

The pose model and the settings file are baked in, so the person who buys it
downloads one thing and nothing else.
"""

import os
import platform
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "pose_landmarker_lite.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
SEP = ";" if platform.system() == "Windows" else ":"


def run(*command):
    print("+", " ".join(command))
    subprocess.check_call(command)


def main():
    run(sys.executable, "-m", "pip", "install", "--upgrade",
        "pyinstaller", "opencv-python", "mediapipe", "pynput", "numpy")

    if not os.path.exists(MODEL) or os.path.getsize(MODEL) < 100_000:
        print("Downloading the pose model so it can be bundled...")
        urllib.request.urlretrieve(MODEL_URL, MODEL)

    mediapipe_dir = os.path.dirname(__import__("mediapipe").__file__)

    args = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--onefile",
        "--name", "BodyPlay",
        "--add-data", f"{MODEL}{SEP}.",
        "--add-data", f"{os.path.join(HERE, 'body_input_settings.json')}{SEP}.",
        "--add-data", f"{os.path.join(mediapipe_dir, 'modules')}{SEP}mediapipe/modules",
        "--collect-all", "mediapipe",
        "--hidden-import", "pynput.keyboard._win32" if platform.system() == "Windows" else "pynput.keyboard._xorg",
        os.path.join(HERE, "body_input_controller.py"),
    ]
    if platform.system() == "Darwin":
        args.remove("--hidden-import")
        args.remove("pynput.keyboard._xorg")
    run(*args)

    print("\nBuilt. Your single-file app is in the 'dist' folder.")
    print("Ship dist/BodyPlay(.exe) on its own - it contains everything.")


if __name__ == "__main__":
    main()

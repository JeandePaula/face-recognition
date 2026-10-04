"""Compatibility entry point; prefer `face-camera run`."""

import sys

from face_camera.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))

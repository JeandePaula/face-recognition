import cv2
import datetime
import os

from config import OUTPUT_DIR


def save_new_faces(new_faces: set[str], original_frame):
    """Save snapshots of newly detected faces."""
    if not new_faces:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name in new_faces:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if name == "Unknown":
            prefix = "person_unknown"
            print("Status: New UNKNOWN person detected.")
        else:
            prefix = "person_known"
            print(f"Status: New detection of KNOWN person: {name}")

        safe_name = name.replace(" ", "_") if name != "Unknown" else "unknown"
        filename = os.path.join(
            OUTPUT_DIR, f"{prefix}-{safe_name}-{timestamp}.png"
        )
        try:
            cv2.imwrite(filename, original_frame)
            print(f"  >> Image saved: {filename}")
        except Exception as e:
            print(f"  [ERROR] Failed to save image {filename}: {e}")

import os
import time
import cv2

from config import (
    IP_CAMERA_URL,
    KNOWN_FACES_DATA,
    OUTPUT_DIR,
)
from face_utils import load_known_faces, process_frame
from image_saver import save_new_faces


def main():
    # Load known faces
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DATA)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Images of new detections will be saved in: '{OUTPUT_DIR}'")

    # Initialize video capture
    print(f"\nTrying to connect to IP camera: {IP_CAMERA_URL}...")
    cap = cv2.VideoCapture(IP_CAMERA_URL)

    if not cap.isOpened():
        print("-" * 40)
        print("CRITICAL ERROR: Unable to open video stream.")
        print("-" * 40)
        return

    print("Camera successfully connected! Starting recognition...")

    previous_faces_seen: set[str] = set()

    DISABLE_GUI = os.environ.get("DISABLE_GUI", "false").lower() == "true"
    if DISABLE_GUI:
        print("Headless mode activated (via DISABLE_GUI environment variable).")
    else:
        print("Press 'q' in the video window to exit.")

    while True:
        ret, original_frame = cap.read()

        if not ret or original_frame is None:
            print("Error capturing frame or stream ended.")
            print("Trying to reconnect in 5 seconds...")
            time.sleep(5)
            break

        processed_frame, current_faces_seen = process_frame(
            original_frame, known_encodings, known_names
        )

        newly_appeared_faces = current_faces_seen - previous_faces_seen
        if newly_appeared_faces:
            save_new_faces(newly_appeared_faces, original_frame)

        previous_faces_seen = current_faces_seen

        if not DISABLE_GUI:
            cv2.imshow(
                "Face Recognition (IP Camera) - Press Q to Exit", processed_frame
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Key 'q' pressed. Exiting...")
                break

    cap.release()
    if not DISABLE_GUI:
        cv2.destroyAllWindows()
    print("Resources released. Program finished.")


if __name__ == "__main__":
    main()

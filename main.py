import cv2
import numpy as np
import face_recognition
import time
import datetime  # Imported for timestamp
import os        # Imported to create directory

# --- Settings ---
IP_CAMERA_URL = "http://192.168.1.66:8080/video"  # URL of your IP camera
KNOWN_FACES_DATA = [
    ("Pessoa 1", "Morty.png"),  # Make sure this file exists
    # ("Person Name 2", "path/to/image2.jpg"),
]
RESIZE_FACTOR = 0.5
TOLERANCE = 0.6
OUTPUT_DIR = "detected_faces_output"  # Directory to save images

# --- Function to Load Known Faces ---
# (No changes needed in this function)
def load_known_faces(known_faces_data: list) -> tuple[list, list]:
    """Loads face encodings and names from image files."""
    known_face_encodings = []
    known_face_names = []
    print("Loading known faces...")
    for name, image_path in known_faces_data:
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) == 1:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"  [OK] Face of '{name}' loaded from '{image_path}'")
            elif len(encodings) > 1:
                print(f"  [WARNING] More than one face found in '{image_path}'. Using the first one.")
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
            else:
                print(f"  [ERROR] No face found in '{image_path}'. Image ignored.")
        except FileNotFoundError:
            print(f"  [ERROR] Image file not found at '{image_path}'. Check the path.")
        except Exception as e:
            print(f"  [ERROR] An error occurred while loading '{image_path}': {e}")
    if not known_face_names:
        print("\nWARNING: No known face was successfully loaded.")
    else:
        print(f"\nLoading complete. {len(known_face_names)} known face(s) ready.")
    return known_face_encodings, known_face_names

# --- Function to Process a Single Frame ---
# Modified to also return the set of detected names
def process_frame(frame: np.ndarray, known_face_encodings: list, known_face_names: list) -> tuple[np.ndarray, set]:
    """Detects and recognizes faces in a video frame and returns the annotated frame and a set of detected names."""

    detected_names_in_frame = set()  # Set to store names in this frame
    processed_frame = frame.copy()   # Work on a copy for drawing

    # 1. Resize (if needed)
    if RESIZE_FACTOR != 1.0:
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    else:
        small_frame = processed_frame  # Use the copy if not resizing

    # 2. Convert BGR -> RGB
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # 3. Find faces and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # 4. Iterate over each found face
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=TOLERANCE)
        name = "Unknown"
        color = (0, 0, 255)

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                color = (0, 255, 0)
                # print(f"Status: Recognized Person - {name}")  # Kept for debug if needed
            # else:
                # print("Status: Detected Person - Unknown")  # Kept for debug if needed

        # Add the name (known or "Unknown") to the set for the current frame
        detected_names_in_frame.add(name)

        # 5. Rescale coordinates
        top, right, bottom, left = face_location
        if RESIZE_FACTOR != 1.0:
            top = int(top / RESIZE_FACTOR)
            right = int(right / RESIZE_FACTOR)
            bottom = int(bottom / RESIZE_FACTOR)
            left = int(left / RESIZE_FACTOR)

        # 6. Draw on the *processed* frame
        cv2.rectangle(processed_frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(processed_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(processed_frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    # Return the drawn frame AND the set of names detected in it
    return processed_frame, detected_names_in_frame

# --- Main Function ---
def main():
    # Load known faces
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DATA)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Images of new detections will be saved in: '{OUTPUT_DIR}'")

    # Initialize video capture
    print(f"\nTrying to connect to IP camera: {IP_CAMERA_URL}...")
    cap = cv2.VideoCapture(IP_CAMERA_URL)

    if not cap.isOpened():
        print("-" * 40)
        print(f"CRITICAL ERROR: Unable to open video stream.")
        # ... (detailed error messages) ...
        print("-" * 40)
        return

    print("Camera successfully connected! Starting recognition...")
    # Add this line if running with GUI (if not, ignore or comment out)
    # print("Press 'q' in the video window to quit.")

    # Variable to track faces seen in the *previous frame*
    previous_faces_seen = set()

    # Check if GUI should be disabled (useful for Docker without display)
    DISABLE_GUI = os.environ.get('DISABLE_GUI', 'false').lower() == 'true'
    if DISABLE_GUI:
        print("Headless mode activated (via DISABLE_GUI environment variable).")
    else:
        print("Press 'q' in the video window to exit.")

    while True:
        # Capture frame by frame (keep the original)
        ret, original_frame = cap.read()

        if not ret or original_frame is None:
            print("Error capturing frame or stream ended.")
            # ... (reconnection logic) ...
            # (reconnection code omitted for brevity, keep your own if needed)
            print("Trying to reconnect in 5 seconds...")  # Simple example
            time.sleep(5)
            # Try to reopen here or break the loop
            # For simplicity, we'll break for now:
            break  # Exit loop if unable to read the frame

        # Process the frame (receive annotated frame and current names set)
        processed_frame, current_faces_seen = process_frame(
            original_frame, known_encodings, known_names
        )

        # --- Logic to Save Image on New Detection ---
        # Compare current set with previous to find new appearances
        newly_appeared_faces = current_faces_seen - previous_faces_seen

        if newly_appeared_faces:
            # Generate a unique timestamp for this detection moment
            # Includes microseconds to avoid collisions if multiple faces appear at the same second
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            # Iterate over each *distinct name* that just appeared
            for name in newly_appeared_faces:
                if name == "Unknown":
                    prefix = "person_unknown"
                    print(f"Status: New UNKNOWN person detected.")
                else:
                    prefix = "person_known"
                    print(f"Status: New detection of KNOWN person: {name}")

                # Build the filename
                filename = os.path.join(OUTPUT_DIR, f"{prefix}-{timestamp}.png")

                # Save the *original* frame (without annotations)
                try:
                    cv2.imwrite(filename, original_frame)
                    print(f"  >> Image saved: {filename}")
                except Exception as e:
                    print(f"  [ERROR] Failed to save image {filename}: {e}")

        # Update the set of faces seen for the next iteration
        previous_faces_seen = current_faces_seen
        # --- End of Image Saving Logic ---

        # Show the resulting frame (ONLY if GUI is not disabled)
        if not DISABLE_GUI:
            cv2.imshow('Face Recognition (IP Camera) - Press Q to Exit', processed_frame)

            # Check for 'q' key (ONLY if GUI is active)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Key 'q' pressed. Exiting...")
                break

    # Release resources
    cap.release()
    if not DISABLE_GUI:
        cv2.destroyAllWindows()
    print("Resources released. Program finished.")

# --- Entry Point ---
if __name__ == '__main__':
    main()

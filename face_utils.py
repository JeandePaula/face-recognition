import cv2
import numpy as np
import face_recognition

from config import RESIZE_FACTOR, TOLERANCE


def load_known_faces(known_faces_data: list) -> tuple[list, list]:
    """Load face encodings and names from image files."""
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
                print(
                    f"  [WARNING] More than one face found in '{image_path}'. Using the first one."
                )
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


def process_frame(
    frame: np.ndarray,
    known_face_encodings: list,
    known_face_names: list,
) -> tuple[np.ndarray, set]:
    """Detect and recognize faces in a frame."""
    detected_names_in_frame: set[str] = set()
    processed_frame = frame.copy()

    if RESIZE_FACTOR != 1.0:
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)
    else:
        small_frame = processed_frame

    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, tolerance=TOLERANCE
        )
        name = "Unknown"
        color = (0, 0, 255)

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                color = (0, 255, 0)

        detected_names_in_frame.add(name)

        top, right, bottom, left = face_location
        if RESIZE_FACTOR != 1.0:
            top = int(top / RESIZE_FACTOR)
            right = int(right / RESIZE_FACTOR)
            bottom = int(bottom / RESIZE_FACTOR)
            left = int(left / RESIZE_FACTOR)

        cv2.rectangle(processed_frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(processed_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(processed_frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    return processed_frame, detected_names_in_frame

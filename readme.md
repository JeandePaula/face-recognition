# 🔍 IP Camera Face Recognition System

This project is a real-time face recognition system that connects to an IP camera stream, detects faces in the video feed, recognizes known individuals, and automatically saves snapshots of newly detected faces. It can be run on any machine with Python installed or inside a Docker container, with or without a graphical interface (GUI).

## ✨ Features
- 🔗 Connects to an IP camera stream (e.g., via mobile IP cam apps)
- 🧠 Recognizes faces based on known image files
- 📦 Saves new face detections with timestamped image files
- 🎛️ GUI or headless (Docker-friendly) modes supported
- 🛠️ Easily extendable to output video, trigger APIs, or integrate with databases

## 📁 Project Structure
.
├── main.py                   # Main face recognition script  
├── Dockerfile                # Docker configuration  
├── requirements.txt          # Python dependencies  
├── Morty.png                 # Example known face  
└── detected_faces_output/    # Output directory for detected snapshots  

## ⚙️ Configuration
All core settings are located in **main.py**:

| Variable           | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `IP_CAMERA_URL`    | URL of the IP camera video stream                                           |
| `KNOWN_FACES_DATA` | List of tuples: (person name, image path)                                   |
| `RESIZE_FACTOR`    | Reduces frame size for faster processing (e.g., 0.5 = 50% smaller)           |
| `TOLERANCE`        | Face match sensitivity (lower = stricter)                                   |
| `OUTPUT_DIR`       | Directory where detected face images are saved                              |
| `DISABLE_GUI`      | Set this environment variable to **true** to run without OpenCV windows     |

### Example: Adding Known Faces
    KNOWN_FACES_DATA = [
        ("John", "john.jpg"),
        ("Lisa", "lisa.png"),
    ]
Each image should show **one clear face**, facing forward.

## 🚀 Running the App

### ✅ Python (Local)
1. Install Python dependencies:
    pip install -r requirements.txt
2. Run the script:
    python main.py
If a GUI is available, a window will open. Press **q** to quit.

### 🐳 Docker (Headless Option)

1. Build the Docker image:
```bash
docker build -t face-recognition-app .
```

2. Run the container:
```bash
docker run --rm \
  -e DISABLE_GUI=true \
  -v $(pwd)/detected_faces_output:/app/detected_faces_output \
  face-recognition-app
```

The `DISABLE_GUI=true` flag disables OpenCV display windows, which is useful when running on servers or containers without a graphical interface.


## 📤 Output
By default, whenever a new person appears in the video (known or unknown), a snapshot of the full original frame is saved to the `detected_faces_output/` directory with a timestamp. You can extend the logic in **main.py** to:
- Save video segments instead of images
- Trigger APIs or webhooks
- Store metadata (face name, time, location) in a database

## 🧰 Requirements
All dependencies are listed in **requirements.txt**:
    opencv-python
    numpy
    face_recognition

> `face_recognition` requires `dlib`, which is compiled automatically. The **Dockerfile** handles the system-level packages needed for this process.

## 🔧 Notes
- This system assumes your IP camera provides an MJPEG stream (e.g., via apps like IP Webcam).
- For best accuracy, face images should be well-lit, front-facing, and high-resolution.
- If the system fails to detect faces or misidentifies them frequently, adjust the `TOLERANCE` or improve the training images.

## 📄 License
This project is open-sourced software licensed under the [MIT license](LICENSE).

## 🙋 Author
Developed by **Jean de Paula**. Feel free to contribute or adapt to your own use case!

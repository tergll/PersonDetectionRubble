# YOLO People Detector (Streamlit Version)

A simple web app to detect people in images using YOLO and Streamlit.

## Features
- Drag-and-drop or select multiple images
- Detects if people are present in each image using YOLOv8
- Shows results and a summary table

## Setup

1. **Install dependencies** (in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the YOLO model weights** (if not already present):
   ```bash
   curl -L -o yolov8x.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
   ```

3. **(For Audio Setup)** *You have to do both anyways XD*

- **ffmpeg** (includes `ffprobe`, required by pydub to convert audio formats)
    ```bash
    brew install ffmpeg
    ```

- Make sure you have wget
    ```bash
    brew install wget
    ```



4. **Run the app:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to the link shown in the terminal (usually http://localhost:8501)

## Notes
- For best compatibility, use Python 3.10 or 3.11.
- If you encounter errors with PyTorch or YOLO, try downgrading your Python version.
- You can switch to `yolov8s.pt` in `app.py` for higher accuracy (but slower inference). 
- It's going to take sometime to load the PANN weights when you run app.py so be patient

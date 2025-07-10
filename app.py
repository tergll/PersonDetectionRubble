import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import torch
import cv2

# --- Audio Dependencies & Model Load ---
import torch
import librosa
import numpy as np
from pydub import AudioSegment # For audio file handling


import os, urllib.request

data_dir = os.path.expanduser('~/panns_data')
os.makedirs(data_dir, exist_ok=True)
csv_path = os.path.join(data_dir, 'class_labels_indices.csv')
if not os.path.isfile(csv_path):
    urllib.request.urlretrieve(
        "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
        csv_path
    )



from panns_inference import AudioTagging
from joblib import load
from pydub import AudioSegment
import tempfile


# Load PANNs audio tagging model
device = "cuda" if torch.cuda.is_available() else "cpu"
audio_model = AudioTagging(checkpoint_path=None, device=device)

# Loading the trained classifier 
sound_clf = load("models/classifier.pkl")  
LABELS = ['scratch', 'cough', 'crying sobbing wail', 'screaming', 'rub', 'sneeze', 'sniff', 'stone rock', 'whispering', 'whistling']

# --- Audio Helper Functions ---
def extract_embedding(path: str) -> np.ndarray:
    # Load audio, resample to mono @32kHz
    wav, sr = librosa.load(path, sr=32000, mono=True)
    # Normalize amplitude
    wav = wav / np.max(np.abs(wav))
    # Batch dimension
    tensor = torch.tensor(wav, device=device).unsqueeze(0)
    # Inference: returns (clip_output, embedding)
    _, embedding = audio_model.inference(tensor)
    return embedding.squeeze()


def predict_sound(path: str) -> str:
    emb = extract_embedding(path).reshape(1, -1)
    idx = int(sound_clf.predict(emb)[0])
    return LABELS[idx]


# Load YOLO model (used yolov8x.pt for best accuracy)
model = YOLO('yolov8x.pt')


# --- Streamlit UI ---
st.title("Multimodal Detector: Image & Audio")
st.write("Upload an image or an audio clip to classify. The app will detect if there are people in each image using YOLO and show the results ranked by confidence.")

uploaded_files = st.file_uploader(
    "Upload images", type=["png", "jpg", "jpeg", "bmp", "webp"], accept_multiple_files=True
)

results_summary = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        # Save to a temp file for YOLO
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            results = model(temp_file.name)
            result = results[0]
            # Filter boxes to only keep class 0 (person)
            if result.boxes is not None and len(result.boxes) > 0:
                person_mask = (result.boxes.cls.cpu().numpy() == 0)
                if person_mask.any():
                    BoxesClass = type(result.boxes)
                    # Stack xyxy, conf, cls (and id if present) into one tensor
                    xyxy = result.boxes.xyxy[person_mask]
                    conf = result.boxes.conf[person_mask].unsqueeze(1)
                    cls = result.boxes.cls[person_mask].unsqueeze(1)
                    if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                        id = result.boxes.id[person_mask].unsqueeze(1)
                        data = torch.cat([xyxy, conf, cls, id], dim=1)
                    else:
                        data = torch.cat([xyxy, conf, cls], dim=1)
                    # Create new Boxes object
                    new_boxes = BoxesClass(data, result.boxes.orig_shape)
                    result.boxes = new_boxes
                    max_conf = float(result.boxes.conf.max().cpu().item())
                    has_people = True
                else:
                    result.boxes = None
                    max_conf = 0.0
                    has_people = False
            else:
                max_conf = 0.0
                has_people = False
            # Render image with only person boxes
            result_img = result.plot()
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_img_rgb)
            # Save all info for later sorting
            results_summary.append({
                "filename": uploaded_file.name,
                "image": result_pil,
                "max_conf": max_conf,
                "has_people": has_people,
            })
    # Sort by max_conf descending
    results_summary.sort(key=lambda x: x["max_conf"], reverse=True)
    # Show results in containers
    st.markdown("---")
    st.subheader("Detection Results (Ranked by Confidence)")
    for i, res in enumerate(results_summary):
        with st.container():
            st.image(res["image"], caption=f"{res['filename']} (Max confidence: {res['max_conf']*100:.1f}%)", use_column_width=True)
            if res["has_people"]:
                st.success(f"People detected! Highest confidence: {res['max_conf']*100:.1f}%")
            else:
                st.warning("No people detected.")
else:
    st.info("Please upload one or more images to begin.") 


# Audio Classification Section (updated)
st.header("Audio Classification")
audio_file = st.file_uploader(
    "Upload an audio file (wav, mp3, ogg, flac)",
    type=["wav","mp3","ogg","flac"]
)
if audio_file:
    # Convert any uploaded format to WAV for consistency
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        AudioSegment.from_file(audio_file).export(tmp.name, format="wav")
        # Run prediction
        pred = predict_sound(tmp.name)
    st.success(f"Predicted sound category: **{pred}**")
else:
    st.info("Upload an audio clip to classify the sound category.")


# --- Video Classification Section ---
st.header("Video Classification")
video_file = st.file_uploader(
    "Upload a video file (mp4, avi, mov, mkv, flv)",
    type=["mp4", "avi", "mov", "mkv", "flv"]
)

if video_file:
    # 1️⃣ Save to temp file
    tfile = tempfile.NamedTemporaryFile(suffix="." + video_file.name.split(".")[-1], delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    # 2️⃣ Vision pass: scan frames for a person
    cap = cv2.VideoCapture(video_path)
    image_detected = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # run YOLO
        results = model(frame)[0]  
        # check for class 0 (person)
        for cls, conf in zip(results.boxes.cls, results.boxes.conf):
            if int(cls) == 0 and conf > 0.0:
                image_detected = True
                break
        if image_detected:
            break
    cap.release()

    # 3️⃣ Audio pass: extract and classify
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        AudioSegment.from_file(video_path).export(tmp_audio.name, format="wav")
        audio_pred = predict_sound(tmp_audio.name)
    # True if the predicted label is one of your defined classes
    audio_detected = audio_pred in LABELS  

    # 4️⃣ Decision logic
    if image_detected or audio_detected:
        st.success(
            "🔊🔍 Person likely **alive** detected " +
            (
                f"(vision={'✔️' if image_detected else '❌'}, "
                f"audio={'✔️' if audio_detected else '❌'})"
            )
        )
    elif image_detected and not audio_detected:
        st.warning("🔍 Person detected visually, but **no sound** → person might be dead")
    else:
        st.info("No person detected via vision or audio.")

else:
    st.info("Upload a video file to begin analysis.")

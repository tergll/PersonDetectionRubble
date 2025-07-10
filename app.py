import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import torch
import cv2

# --- Audio Dependencies & Model Load ---
import librosa
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

# --- Video Helper Functions ---
def extract_audio_from_video(video_path: str) -> str:
    """Extract audio from video and return path to audio file"""
    audio_path = video_path.replace('.mp4', '_audio.wav').replace('.avi', '_audio.wav').replace('.mov', '_audio.wav')
    video = AudioSegment.from_file(video_path)
    video.export(audio_path, format="wav")
    return audio_path

def extract_frames_from_video(video_path: str, interval_seconds: int = 1) -> list:
    """Extract frames from video at specified interval"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    frames = []
    frame_timestamps = []
    
    # Extract frames at 1-second intervals
    for second in range(0, int(duration), interval_seconds):
        frame_number = int(second * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_timestamps.append(second)
    
    cap.release()
    return frames, frame_timestamps

# Load YOLO model (use yolov8x.pt for best accuracy)
model = YOLO('yolov8x.pt')

# --- Streamlit UI ---
st.title("Multimodal Detector: Image, Audio & Video")
st.write("Upload images, audio files, or videos. The app will automatically detect the file type and apply the appropriate model.")

# Unified file uploader for images, audio, and video
uploaded_files = st.file_uploader(
    "Upload files (images: png, jpg, jpeg, bmp, webp | audio: wav, mp3, ogg, flac | video: mp4, avi, mov)",
    type=["png", "jpg", "jpeg", "bmp", "webp", "wav", "mp3", "ogg", "flac", "mp4", "avi", "mov"],
    accept_multiple_files=True
)

# Separate lists for different file types
image_files = []
audio_files = []
video_files = []

# Categorize uploaded files by type
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_type = uploaded_file.type
        if file_type and file_type.startswith('image/'):
            image_files.append(uploaded_file)
        elif file_type and file_type.startswith('audio/'):
            audio_files.append(uploaded_file)
        elif file_type and file_type.startswith('video/'):
            video_files.append(uploaded_file)
        else:
            # Fallback based on file extension
            file_extension = uploaded_file.name.lower().split('.')[-1]
            if file_extension in ['png', 'jpg', 'jpeg', 'bmp', 'webp']:
                image_files.append(uploaded_file)
            elif file_extension in ['wav', 'mp3', 'ogg', 'flac']:
                audio_files.append(uploaded_file)
            elif file_extension in ['mp4', 'avi', 'mov']:
                video_files.append(uploaded_file)

# Process images if any
image_results = []
image_detected = False  # Global flag for fusion logic
if image_files:
    st.header("📷 Image Analysis Results")
    for uploaded_file in image_files:
        with st.expander(f"Processing: {uploaded_file.name}", expanded=True):
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
                        image_detected = True  # Set global flag
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
                
                # Store results for later sorting
                image_results.append({
                    "filename": uploaded_file.name,
                    "has_people": has_people,
                    "max_conf": max_conf,
                    "result_image": result_pil
                })
    
    # Sort image results by confidence (highest to lowest)
    image_results.sort(key=lambda x: x["max_conf"], reverse=True)
    
    # Display sorted results
    for i, result in enumerate(image_results):
        with st.expander(f"#{i+1} - {result['filename']} (Confidence: {result['max_conf']*100:.1f}%)", expanded=True):
            st.image(result["result_image"], caption=f"Detection Result: {result['filename']}", use_container_width=True)
            if result["has_people"]:
                st.success(f"✅ People detected! Highest confidence: {result['max_conf']*100:.1f}%")
            else:
                st.warning("❌ No people detected.")

# Process videos if any
video_audio_detected = False  # Global flag for video audio detection
if video_files:
    st.header("🎬 Video Analysis Results")
    for uploaded_file in video_files:
        with st.expander(f"Processing Video: {uploaded_file.name}", expanded=True):
            # Save video to temp file
            with tempfile.NamedTemporaryFile(suffix=f".{uploaded_file.name.split('.')[-1]}", delete=False) as tmp_video:
                tmp_video.write(uploaded_file.getvalue())
                video_path = tmp_video.name
            
            # Extract audio from video
            st.write("🎵 **Extracting audio...**")
            audio_path = extract_audio_from_video(video_path)
            
            # Process audio
            pred = predict_sound(audio_path)
            # Modified audio detection logic to include "scratch" as valid
            audio_detected = (pred in LABELS)  # Include all labels including "scratch"
            if audio_detected:
                video_audio_detected = True  # Set global flag
            st.success(f"🎯 Audio prediction: **{pred}**")
            
            # Extract frames from video
            st.write("📷 **Extracting frames...**")
            frames, timestamps = extract_frames_from_video(video_path, interval_seconds=1)
            
            if frames:
                st.write(f"📊 Extracted {len(frames)} frames at 1-second intervals")
                
                # Process frames with YOLO
                frame_results = []
                video_image_detected = False  # Flag for video frame detection
                for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
                    # Convert numpy array to PIL Image
                    frame_pil = Image.fromarray(frame)
                    
                    # Save frame to temp file for YOLO
                    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
                        frame_pil.save(temp_file.name)
                        results = model(temp_file.name)
                        result = results[0]
                        
                        # Filter boxes to only keep class 0 (person)
                        if result.boxes is not None and len(result.boxes) > 0:
                            person_mask = (result.boxes.cls.cpu().numpy() == 0)
                            if person_mask.any():
                                BoxesClass = type(result.boxes)
                                xyxy = result.boxes.xyxy[person_mask]
                                conf = result.boxes.conf[person_mask].unsqueeze(1)
                                cls = result.boxes.cls[person_mask].unsqueeze(1)
                                if hasattr(result.boxes, 'id') and result.boxes.id is not None:
                                    id = result.boxes.id[person_mask].unsqueeze(1)
                                    data = torch.cat([xyxy, conf, cls, id], dim=1)
                                else:
                                    data = torch.cat([xyxy, conf, cls], dim=1)
                                new_boxes = BoxesClass(data, result.boxes.orig_shape)
                                result.boxes = new_boxes
                                max_conf = float(result.boxes.conf.max().cpu().item())
                                has_people = True
                                video_image_detected = True  # Set video frame detection flag
                            else:
                                result.boxes = None
                                max_conf = 0.0
                                has_people = False
                        else:
                            max_conf = 0.0
                            has_people = False
                        
                        # Render frame with only person boxes
                        result_img = result.plot()
                        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        result_pil = Image.fromarray(result_img_rgb)
                        
                        frame_results.append({
                            "timestamp": timestamp,
                            "has_people": has_people,
                            "max_conf": max_conf,
                            "result_image": result_pil
                        })
                
                # Sort frame results by confidence
                frame_results.sort(key=lambda x: x["max_conf"], reverse=True)
                
                # Display frame results
                st.write("🎬 **Frame Analysis Results (Ranked by Confidence):**")
                for i, result in enumerate(frame_results):
                    with st.expander(f"Frame at {result['timestamp']}s (Confidence: {result['max_conf']*100:.1f}%)", expanded=False):
                        st.image(result["result_image"], caption=f"Frame at {result['timestamp']}s", use_container_width=True)
                        if result["has_people"]:
                            st.success(f"✅ People detected at {result['timestamp']}s! Confidence: {result['max_conf']*100:.1f}%")
                        else:
                            st.warning(f"❌ No people detected at {result['timestamp']}s")
                
                # Fusion logic for video
                if video_image_detected and audio_detected:
                    st.success(
                        "✅ Person **alive** detected in video  " +
                        f"(vision={'✅' if video_image_detected else '❌'}, audio={'✅' if audio_detected else '❌'})"
                    )
                elif video_image_detected and not audio_detected:
                    st.warning(
                        "👀 Person detected visually but **no valid sound** → might be dead  " +
                        f"(vision={'✅'}, audio={'❌'})"
                    )
                elif audio_detected and not video_image_detected:
                    st.info(
                        "🔊 Valid sound detected but **no person visually** → person might be alive  " +
                        f"(vision={'❌'}, audio={'✅'})"
                    )
                else:
                    st.error("🚫 No person detected via vision or valid audio.")
            else:
                st.error("❌ Could not extract frames from video")
            
            # Clean up temp files
            os.unlink(video_path)
            os.unlink(audio_path)

# Process audio files if any
audio_results = []
audio_detected = False  # Global flag for fusion logic
if audio_files:
    st.header("🎵 Audio Analysis Results")
    for uploaded_file in audio_files:
        with st.expander(f"Processing: {uploaded_file.name}", expanded=True):
            # Display audio player
            st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
            
            # Convert any uploaded format to WAV for consistency
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                AudioSegment.from_file(uploaded_file).export(tmp.name, format="wav")
                # Run prediction
                pred = predict_sound(tmp.name)
            
            # Modified audio detection logic to include "scratch" as valid
            audio_detected = (pred in LABELS)  # Include all labels including "scratch"
            st.success(f"🎯 Predicted sound category: **{pred}**")
            audio_results.append({
                "filename": uploaded_file.name,
                "prediction": pred
            })

# Summary section
if image_results or audio_results or video_files:
    st.header("📊 Summary")
    
    if image_results:
        st.subheader("Image Analysis Summary")
        for result in image_results:
            status = "✅ People detected" if result["has_people"] else "❌ No people"
            conf_text = f" (Confidence: {result['max_conf']*100:.1f}%)" if result["has_people"] else ""
            st.write(f"**{result['filename']}**: {status}{conf_text}")
    
    if audio_results:
        st.subheader("Audio Analysis Summary")
        for result in audio_results:
            st.write(f"**{result['filename']}**: 🎵 {result['prediction']}")
    
    if video_files:
        st.subheader("Video Analysis Summary")
        for video_file in video_files:
            st.write(f"**{video_file.name}**: 🎬 Video processed (audio + frames)")

# Show file type breakdown
if uploaded_files:
    st.sidebar.header("📁 File Breakdown")
    st.sidebar.write(f"📷 Images: {len(image_files)}")
    st.sidebar.write(f"🎵 Audio: {len(audio_files)}")
    st.sidebar.write(f"📹 Videos: {len(video_files)}")
    st.sidebar.write(f"📄 Total: {len(uploaded_files)}")

else:
    st.info("👆 Please upload images, audio files, or videos to begin analysis.")
    st.write("**Supported formats:**")
    st.write("• **Images:** PNG, JPG, JPEG, BMP, WebP")
    st.write("• **Audio:** WAV, MP3, OGG, FLAC")
    st.write("• **Video:** MP4, AVI, MOV")
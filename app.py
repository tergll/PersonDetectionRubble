import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import torch
import cv2

# Load YOLO model (use yolov8x.pt for best accuracy)
model = YOLO('yolov8x.pt')

st.title("YOLO People Detector")
st.write("Upload one or more images. The app will detect if there are people in each image using YOLO and show the results ranked by confidence.")

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
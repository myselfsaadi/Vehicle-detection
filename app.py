import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import tempfile
from PIL import Image
import base64
import os
import time

# Load YOLOv5 model
@st.cache_resource
def load_model(model_path="yolov5x.pt"):
    return YOLO(model_path)

# Function to process video with preview, ETA, logs, progress
def process_video(video_path, model, confidence_threshold, output_video_path, log_callback, image_placeholder, progress_bar, show_preview=True, show_logs=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_callback("❌ Error: Unable to open video file.")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    log_callback(f"📄 Video loaded: {total_frames} frames @ {fps} FPS")
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=confidence_threshold)
        detections = results[0].boxes.data.cpu().numpy()

        obj_count = 0
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            label = f"{model.names[int(cls_id)]} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            obj_count += 1

        out.write(frame)
        frame_count += 1

        # Live preview every 5 frames
        if show_preview and frame_count % 5 == 0:
            preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_placeholder.image(preview, caption=f"Frame {frame_count}/{total_frames}", channels="RGB")

        # Progress & ETA
        elapsed = time.time() - start_time
        progress = frame_count / total_frames
        fps_now = frame_count / elapsed if elapsed > 0 else 0
        eta = (total_frames - frame_count) / fps_now if fps_now > 0 else 0
        progress_bar.progress(progress, text=f"{frame_count}/{total_frames} frames | ETA: {int(eta)}s")

        if show_logs:
            log_callback(f"✅ Frame {frame_count}/{total_frames}: {obj_count} object(s) detected")

        time.sleep(0.01)  # Throttle UI update to prevent crashes

    cap.release()
    out.release()
    log_callback("🎉 Processing complete!")
    return output_video_path

# Utility: Download link
def get_download_link(file_path, label="Download"):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    file_name = os.path.basename(file_path)
    href = f'<a href="data:video/mp4;base64,{b64}" download="{file_name}">{label}</a>'
    return href

# Streamlit UI
st.title("YOLOv5x Vehicle Detection App")
st.sidebar.title("⚙️ Settings")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
show_preview = st.sidebar.checkbox("Show Live Preview", value=True)
show_logs = st.sidebar.checkbox("Show Logs", value=True)

uploaded_video = st.file_uploader("📤 Upload a Video", type=["mp4", "avi", "mov", "mkv", "webm", "m4v"])

# Placeholders
st.subheader("Live Frame Preview")
image_placeholder = st.empty()

st.subheader("Logs")
log_expander = st.expander("Real-time Logs (click to expand)", expanded=True)
log_text = ""
log_area = log_expander.empty()

def append_log(message):
    global log_text
    log_text += message + "\n"
    log_area.text_area("Logs", log_text, height=300)

progress_bar = st.progress(0)

if uploaded_video:
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())
    st.sidebar.success("✅ Video uploaded!")

    model = load_model("yolov5x.pt")

    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    with st.spinner("⏳ Processing..."):
        processed_video_path = process_video(
            temp_video_path, model, confidence_threshold,
            output_video_path, append_log, image_placeholder, progress_bar,
            show_preview=show_preview, show_logs=show_logs
        )

    if processed_video_path:
        st.success("✅ Done! Watch your processed video below.")
        st.video(processed_video_path)
        st.markdown(get_download_link(processed_video_path, "⬇️ Download Processed Video"), unsafe_allow_html=True)

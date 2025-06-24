import time
import streamlit as st
import tempfile
import os

# Core dependencies
try:
    import cv2
    import numpy as np
    import requests
except ImportError as e:
    st.error(f"❌ Missing dependency: {e}")
    st.stop()

# FastAPI inference endpoint URL
API_URL = "http://localhost:8000/predict"

# Streamlit page config
st.set_page_config(page_title="YOLO + EasyOCR Video Demo", layout="wide")
st.title("📹 YOLO + EasyOCR Real-Time Video Processing")

# File uploader
uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded and st.button("Run Inference"):
    # Save upload to temp file
    in_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    in_file.write(uploaded.read())
    in_file.close()

    # Prepare video capture
    cap = cv2.VideoCapture(in_file.name)
    if not cap.isOpened():
        st.error("❌ Could not open uploaded video.")
        st.stop()

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # UI placeholders
    frame_placeholder = st.empty()
    progress = st.progress(0)
    status = st.empty()

    # Process each frame and display
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Encode and send to API
        _, img_jpg = cv2.imencode('.jpg', frame)
        try:
            resp = requests.post(
                API_URL,
                files={"file": ("frame.jpg", img_jpg.tobytes(), "image/jpeg")},
                timeout=5
            )
            resp.raise_for_status()
            nparr = np.frombuffer(resp.content, np.uint8)
            annotated = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            st.error(f"❌ Inference failed at frame {i+1}: {e}")
            break

        # Display annotated frame
        frame_placeholder.image(annotated, channels="BGR", use_container_width=True)

        # Update progress
        progress.progress((i+1) / frame_count)
        status.text(f"Frame {i+1} / {frame_count}")

        # Small delay to simulate playback speed
        time.sleep(1.0 / fps)

    cap.release()

    # Finalize
    st.success("✅ Video processing complete!")
    os.remove(in_file.name)

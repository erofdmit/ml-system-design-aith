import time
import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import requests

# FastAPI inference endpoint URL (можно менять через сайдбар)
API_URL = st.sidebar.text_input("API URL:", "http://localhost:8000/predict")

# Параметры отображения
display_width = st.sidebar.slider("Display width", 200, 1920, 800)
skip_frames   = st.sidebar.number_input("Process every Nth frame", 1, 10, 1)
conf_thresh   = st.sidebar.slider("Min detection confidence", 0.0, 1.0, 0.3)

st.set_page_config(page_title="YOLO + EasyOCR Video Demo", layout="wide")
st.title("📹 YOLO + EasyOCR Real-Time Video Processing")

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded and st.button("Run Inference"):
    # Сохраняем во временный файл
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded.read())
    tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    if not cap.isOpened():
        st.error("❌ Could not open uploaded video.")
        st.stop()

    orig_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps         = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_pl = st.empty()
    progress = st.progress(0)
    status   = st.empty()

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        if i % skip_frames != 0:
            continue

        # Отправляем кадр на API
        _, jpg = cv2.imencode(".jpg", frame)
        try:
            resp = requests.post(
                API_URL,
                files={"file": ("frame.jpg", jpg.tobytes(), "image/jpeg")},
                timeout=10
            )
            resp.raise_for_status()
            detections = resp.json().get("detections", [])
        except Exception as e:
            st.error(f"❌ Inference failed at frame {i+1}: {e}")
            break

        # Рисуем аннотации
        disp = cv2.resize(frame, (display_width, int(orig_h * display_width / orig_w)))
        scale = display_width / orig_w
        for det in detections:
            if det["confidence"] < conf_thresh:
                continue
            x, y, w, h = det["box"]
            x1, y1 = int(x * scale), int(y * scale)
            x2, y2 = int((x + w) * scale), int((y + h) * scale)
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_id']} {det['confidence']:.2f}: {det['text']}"
            cv2.putText(disp, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        frame_pl.image(disp, channels="BGR", use_container_width=True)
        progress.progress((i + 1) / frame_count)
        status.text(f"Frame {i + 1}/{frame_count}")

        time.sleep(1.0 / fps)

    cap.release()
    os.remove(tmp.name)
    st.success("✅ Video processing complete!")

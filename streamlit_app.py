import os
from pathlib import Path

import cv2
import tempfile
import time
from typing import Optional

import requests
import streamlit as st
from custom_inference.yolo_inference import inference, load_model

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_YOLO_MODEL_PATH = ROOT_DIR / "custom_inference" / "models" / "yolo" / "best.pt"
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", str(DEFAULT_YOLO_MODEL_PATH))


@st.cache_resource
def get_model():
    # Downloads pretrained weights on first run
    return load_model(YOLO_MODEL_PATH)


API_URL = "http://localhost:8000/predict"


def get_text(frame) -> Optional[str]:
    _, encoded = cv2.imencode(".jpg", frame)
    files = {"file": ("frame.jpg", encoded.tobytes(), "image/jpeg")}
    try:
        resp = requests.post(API_URL, files=files, timeout=5)
        if resp.status_code == 200:
            return resp.json().get("text")
    except Exception:
        return None
    return None


def draw_boxes(frame, detections, text: Optional[str] = None):
    for det in detections:
        x, y, w, h = map(int, det["box"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{det['class_id']}:{det['confidence']:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if text:
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return frame


st.title("Video Inference Demo")

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded and st.button("Run Inference"):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded.read())

    model = get_model()
    cap = cv2.VideoCapture(tfile.name)
    frame_placeholder = st.empty()
    text_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detections = inference(model, frame)
        text = get_text(frame)
        annotated = draw_boxes(frame, detections, text)
        frame_placeholder.image(annotated, channels="BGR")
        if text:
            text_placeholder.write(f"Recognized: {text}")
        time.sleep(0.03)

    cap.release()
    tfile.close()

import cv2
import tempfile
import time
import streamlit as st
from custom_inference.yolo_inference import load_model, inference


@st.cache_resource
def get_model():
    # Downloads pretrained weights on first run
    return load_model("best.pt")


def draw_boxes(frame, detections):
    for det in detections:
        x, y, w, h = map(int, det["box"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{det['class_id']}:{det['confidence']:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


st.title("Video Inference Demo")

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded and st.button("Run Inference"):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded.read())

    model = get_model()
    cap = cv2.VideoCapture(tfile.name)
    frame_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detections = inference(model, frame)
        annotated = draw_boxes(frame, detections)
        frame_placeholder.image(annotated, channels="BGR")
        time.sleep(0.03)

    cap.release()
    tfile.close()

import os

from custom_inference.ocr_inference import load_model_and_converter, predict_text
# 1) Отключаем file-watcher Streamlit
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# 2) Monkey-patch set_page_config до любых Streamlit-импортов
import streamlit.commands.page_config as _pc
_pc.set_page_config = lambda *args, **kwargs: None

import streamlit as st
# 3) Единственный вызов set_page_config
st.set_page_config(
    page_title="📹 YOLO + EasyOCR (Local) Video Processing",
    layout="wide",
)

import torch
# 4) Избегаем file-watcher у torch.classes
torch.classes.__path__ = []

# 5) Задаём устройство один раз
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("CUDA available:", torch.cuda.is_available())

import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Импорт opt, YOLO-инференс и ваши утилиты OCR
from custom_inference.config import opt
from custom_inference.yolo_inference import load_model as load_yolo, inference as yolo_inference

# Пути к весам
ROOT_DIR     = Path(__file__).resolve().parent
DEFAULT_YOLO = ROOT_DIR / "custom_inference/models/yolo/best.pt"
DEFAULT_OCR  = ROOT_DIR / "custom_inference/models/ocr/best_accuracy.pth"
YOLO_PATH    = os.getenv("YOLO_MODEL_PATH", str(DEFAULT_YOLO))
OCR_PATH     = os.getenv("OCR_MODEL_PATH",  str(DEFAULT_OCR))

@st.cache_resource
def load_models():
    # выводим, на чём работаем
    if device.type == "cuda":
        st.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("CUDA not available; using CPU.")

    yolo_model = load_yolo(YOLO_PATH)
    try:
        yolo_model.model.to(device)
    except AttributeError:
        pass

    ocr_model, ocr_converter = load_model_and_converter(opt, OCR_PATH, device=device)
    ocr_model.to(device)
    return yolo_model, ocr_model, ocr_converter

yolo_model, ocr_model, ocr_converter = load_models()

# --- UI ---
st.title("📹 YOLO + EasyOCR (Local) Video Processing")
display_width = st.sidebar.slider("Display width", 200, 1920, 800)
skip_frames   = st.sidebar.number_input("Process every Nth frame", 1, 10, 1)
conf_thresh   = st.sidebar.slider("Min detection confidence", 0.0, 1.0, 0.3)

uploaded = st.file_uploader("Upload a video", type=["mp4","avi","mov"])
if uploaded and st.button("Run Inference"):
    temp = ROOT_DIR / f"temp_{uploaded.name}"
    with open(temp, "wb") as f: f.write(uploaded.read())
    cap = cv2.VideoCapture(str(temp))
    if not cap.isOpened(): st.error("❌ Could not open video."); st.stop()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_pl = st.empty()
    progress = st.progress(0)
    status   = st.empty()

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret: break
        if i % skip_frames: continue

        disp = cv2.resize(frame, (display_width, int(h * display_width / w)))

        t0 = time.time()
        dets = yolo_inference(yolo_model, frame)
        results = []
        for det in dets:
            if det["confidence"] < conf_thresh: continue
            x,y,fw,fh = map(int, det["box"])
            crop = frame[y:y+fh, x:x+fw]
            txt  = predict_text(ocr_model, ocr_converter, crop, opt)
            print(f"Detected: {det['class_id']} {det['confidence']:.2f} - {txt}")
            results.append((det["box"], det["class_id"], det["confidence"], txt))
        t1 = time.time()

        # отрисовка
        for (x,y,fw,fh), cid, conf, txt in results:
            xs, ys = int(x*display_width/w), int(y*display_width/w)
            ws, hs = int(fw*display_width/w), int(fh*display_width/w)
            cv2.rectangle(disp, (xs,ys), (xs+ws,ys+hs), (0,255,0), 2)
            cv2.putText(disp, f"{cid} {conf:.2f}: {txt}", (xs, ys-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        frame_pl.image(disp, channels="BGR", use_container_width=True)
        progress.progress((i+1)/n_frames)
        status.text(f"Frame {i+1}/{n_frames} — {(t1-t0):.2f}s")
        time.sleep(1/fps)

    cap.release()
    os.remove(temp)
    st.success("✅ Done!")

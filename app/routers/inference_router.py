import base64
import os
import tempfile

import cv2
from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO


templates = Jinja2Templates(directory="app/templates")

inference_router = APIRouter(prefix="/inference", tags=["inference"])


@inference_router.get("/video", response_class=HTMLResponse)
async def video_form(request: Request):
    return templates.TemplateResponse("upload_video.html", {"request": request})


@inference_router.post("/video", response_class=HTMLResponse)
async def process_video(request: Request, video: UploadFile = File(...), fps: int = Form(1)):
    model = YOLO("yolov8n.pt")

    contents = await video.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(original_fps / fps))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            results = model(frame)
            annotated = results[0].plot()
            _, buffer = cv2.imencode(".jpg", annotated)
            frames.append(base64.b64encode(buffer).decode())
        idx += 1

    cap.release()
    os.remove(tmp_path)
    return templates.TemplateResponse("results.html", {"request": request, "frames": frames})

import base64
import cv2
import json
import os
import tempfile
from typing import List, Dict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert

from custom_inference.yolo_inference import load_model, inference as yolo_inference
from custom_inference.ocr_inference import (
    load_model_and_converter,
    predict_text,
    get_device,
)
from db_ops.db_create import FrameResult

YOLO_WEIGHTS = "custom_inference/models/yolo/best.pt"
OCR_WEIGHTS = "custom_inference/models/ocr/best.pth"


def encode_frame(frame: cv2.typing.MatLike) -> str:
    """Return base64-encoded JPEG image."""
    success, buffer = cv2.imencode(".jpg", frame)
    if not success:
        raise ValueError("Failed to encode frame")
    return base64.b64encode(buffer).decode()


def annotate(frame, detections, texts):
    """Draw bounding boxes and recognized text on the frame."""
    for det, text in zip(detections, texts):
        x1, y1, w, h = det["box"]
        x1, y1, w, h = int(x1), int(y1), int(w), int(h)
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            text,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return frame


async def process_video(file, fps: int, session: AsyncSession) -> List[Dict]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise ValueError("Cannot open uploaded video")

    # load models
    yolo_model = load_model(YOLO_WEIGHTS)
    device = get_device(use_cuda=False)

    # simple opt placeholder
    class Opt:
        imgH = 32
        imgW = 100
        input_channel = 3
        Prediction = "CTC"
        character = "0123456789"

    ocr_model, converter = load_model_and_converter(Opt(), OCR_WEIGHTS, device)

    results = []
    frame_idx = 0
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    step = max(int(frame_rate // fps), 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            dets = yolo_inference(yolo_model, frame)
            texts = []
            for det in dets:
                x1, y1, w, h = det["box"]
                crop = frame[int(y1) : int(y1 + h), int(x1) : int(x1 + w)]  # noqa: E203
                text = predict_text(ocr_model, converter, crop, Opt())
                texts.append(text)
            result_record = {
                "frame": frame_idx,
                "texts": texts,
            }
            stmt = insert(FrameResult).values(
                video_name=file.filename,
                frame_number=frame_idx,
                recognized_text=json.dumps(texts),
            )
            async with session.begin():
                await session.execute(stmt)
            results.append(result_record)
        frame_idx += 1
    cap.release()
    os.remove(tmp_path)
    return results


async def process_video_frames(file, fps: int, session: AsyncSession) -> List[Dict]:
    """Return annotated frames and recognized text."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        raise ValueError("Cannot open uploaded video")

    yolo_model = load_model(YOLO_WEIGHTS)
    device = get_device(use_cuda=False)

    class Opt:
        imgH = 32
        imgW = 100
        input_channel = 3
        Prediction = "CTC"
        character = "0123456789"

    ocr_model, converter = load_model_and_converter(Opt(), OCR_WEIGHTS, device)

    results = []
    frame_idx = 0
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    step = max(int(frame_rate // fps), 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            dets = yolo_inference(yolo_model, frame)
            texts = []
            for det in dets:
                x1, y1, w, h = det["box"]
                crop = frame[int(y1) : int(y1 + h), int(x1) : int(x1 + w)]  # noqa: E203
                text = predict_text(ocr_model, converter, crop, Opt())
                texts.append(text)
            annotated = annotate(frame.copy(), dets, texts)
            encoded = encode_frame(annotated)
            stmt = insert(FrameResult).values(
                video_name=file.filename,
                frame_number=frame_idx,
                recognized_text=json.dumps(texts),
            )
            async with session.begin():
                await session.execute(stmt)
            results.append({"frame": frame_idx, "texts": texts, "image": encoded})
        frame_idx += 1
    cap.release()
    os.remove(tmp_path)
    return results

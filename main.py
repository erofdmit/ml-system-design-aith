from __future__ import annotations
import os
from pathlib import Path
import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from custom_inference.config import opt
from custom_inference.yolo_inference import load_model as load_yolo, inference as yolo_inference
from custom_inference.ocr_inference import (
    get_device,
    load_model_and_converter_strip,
    predict_text_fixed
)

from database import Base, engine, get_session
from models import Recognition

ROOT_DIR = Path(__file__).resolve().parent

# —– Paths to your weights
DEFAULT_YOLO = ROOT_DIR / "custom_inference/models/yolo/best.pt"
DEFAULT_OCR  = ROOT_DIR / "custom_inference/models/ocr/best_accuracy.pth"

YOLO_PATH = os.getenv("YOLO_MODEL_PATH",  str(DEFAULT_YOLO))
OCR_PATH  = os.getenv("OCR_MODEL_PATH",   str(DEFAULT_OCR))

# —– Load YOLO one-time
yolo_model = load_yolo(YOLO_PATH)

# —– Load OCR model (strip) one-time, на нужном устройстве
ocr_model, ocr_converter = load_model_and_converter_strip(
    opt,
    str(OCR_PATH),
    device=get_device()
)

async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

def create_app() -> FastAPI:
    app = FastAPI(title="Video Inference API")

    @app.on_event("startup")
    async def on_startup():
        await init_db()

    @app.post("/predict")
    async def predict(
        file: UploadFile = File(...),
        session: AsyncSession = Depends(get_session)
    ):
        # Читаем байты и декодим в кадр OpenCV
        data = await file.read()
        img_arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image upload")

        # 1) Получаем детекции от YOLO
        detections = yolo_inference(yolo_model, frame)
        results = []

        for det in detections:
            x, y, w, h = map(int, det["box"])
            crop = frame[y:y+h, x:x+w]

            # 2) OCR только фикс–функцией
            text = predict_text_fixed(ocr_model, ocr_converter, crop, opt)
            det["text"] = text

            # 3) Сохраняем в базу
            rec = Recognition(text=text)
            session.add(rec)

            # 4) Собираем ответ
            results.append({
                "class_id":   det["class_id"],
                "confidence": float(det["confidence"]),
                "box":        [x, y, w, h],
                "text":       text
            })

        await session.commit()
        return JSONResponse(content={"detections": results})

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

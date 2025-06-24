from __future__ import annotations
import os
from pathlib import Path
import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from custom_inference.config import opt
from custom_inference.yolo_inference import load_model as load_yolo, inference as yolo_inference
from custom_inference.ocr_inference import get_device, load_model_and_converter, predict_text
from sqlalchemy.ext.asyncio import AsyncSession
from database import Base, engine, get_session
from models import Recognition

ROOT_DIR = Path(__file__).resolve().parent

# —– Paths to your weights
DEFAULT_YOLO = ROOT_DIR / "custom_inference/models/yolo/best.pt"
DEFAULT_OCR  = ROOT_DIR / "custom_inference/models/ocr/best_accuracy.pth"

YOLO_PATH = os.getenv("YOLO_MODEL_PATH",  str(DEFAULT_YOLO))
OCR_PATH  = os.getenv("OCR_MODEL_PATH",   str(DEFAULT_OCR))

# —– Load both models one-time, at import
yolo_model = load_yolo(YOLO_PATH)
ocr_model, ocr_converter = load_model_and_converter(opt, OCR_PATH, device=get_device())

async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

def create_app() -> FastAPI:
    app = FastAPI(title="Video Inference API")

    @app.on_event("startup")
    async def on_startup():
        await init_db()

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), session: AsyncSession = Depends(get_session)):
        data = await file.read()
        img_arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(400, "Invalid image upload")

        detections = yolo_inference(yolo_model, frame)
        results = []

        for det in detections:
            x, y, w, h = map(int, det["box"])
            crop = frame[y:y+h, x:x+w]
            text = predict_text(ocr_model, ocr_converter, crop, opt)
            det["text"] = text

            # persist recognized text
            rec = Recognition(text=text)
            session.add(rec)

            results.append({
                "class_id": det["class_id"],
                "confidence": float(det["confidence"]),
                "box": [x, y, w, h],
                "text": text
            })

        await session.commit()
        return JSONResponse(content={"detections": results})

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
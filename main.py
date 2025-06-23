from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from custom_inference.config import Opt
from custom_inference.ocr_inference import (
    get_device,
    load_model_and_converter,
    predict_text,
)
from database import Base, engine, get_session
from models import Recognition


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OCR_MODEL_PATH = ROOT_DIR / "custom_inference" / "models" / "ocr" / "best_accuracy.pth"

opt = Opt()
ocr_weights = os.getenv("OCR_MODEL_PATH", str(DEFAULT_OCR_MODEL_PATH))
ocr_model, ocr_converter = load_model_and_converter(opt, ocr_weights, device=get_device())


async def init_models() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def create_app() -> FastAPI:
    app = FastAPI(title="Video Inference API")

    @app.on_event("startup")
    async def on_startup() -> None:
        await init_models()

    @app.get("/ping")
    async def ping():
        return {"status": "ok"}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...), session: AsyncSession = Depends(get_session)) -> dict[str, str]:
        data = await file.read()
        np_arr = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        text = predict_text(ocr_model, ocr_converter, frame, opt)

        rec = Recognition(text=text)
        session.add(rec)
        await session.commit()

        return {"text": text}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)

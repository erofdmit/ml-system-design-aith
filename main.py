from __future__ import annotations

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from custom_inference.config import Opt
from custom_inference.ocr_inference import load_model_and_converter, predict_text, get_device
from database import Base, engine, get_session
from models import Recognition


opt = Opt()
ocr_model, ocr_converter = load_model_and_converter(opt, "ocr/best_accuracy.pth", device=get_device())


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

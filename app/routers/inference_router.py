from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from db_ops.connector import get_db_session
from ml_app.video_inference import process_video

router = APIRouter(tags=["inference"], prefix="/inference")


@router.post("/video")
async def video_inference(
    file: UploadFile = File(...),
    fps: int = Form(1),
    session: AsyncSession = Depends(get_db_session),
):
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    results = await process_video(file, fps=fps, session=session)
    return {"results": results}

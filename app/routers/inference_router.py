from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from db_ops.connector import get_db_session
from ml_app.video_inference import process_video, process_video_frames

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


@router.post("/video/frames")
async def video_inference_frames(
    file: UploadFile = File(...),
    fps: int = Form(1),
    session: AsyncSession = Depends(get_db_session),
):
    """Return annotated frames as base64 images."""
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    results = await process_video_frames(file, fps=fps, session=session)
    return {"results": results}

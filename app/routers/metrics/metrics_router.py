import asyncio

from fastapi import APIRouter, Body, Depends, HTTPException, status, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from ...db_ops.connector import get_db_session
from app.routers.metrics.metrics_methods import analyze_and_visualize_data,  fetch_and_process_trip_data
import os


metrics_router = APIRouter(tags=['metrics'], prefix='/metrics')
loop = asyncio.get_event_loop()

@metrics_router.post('/metrics_from_detector')
async def get_trip_analysis(request: Request, trip_id: int, session: AsyncSession = Depends(get_db_session)):
    # Fetch and process data
    data, error_message = await fetch_and_process_trip_data(trip_id, session)
    if error_message:
        raise HTTPException(status_code=404, detail=error_message)

    # Analyze data and generate plots
    analysis_results = await analyze_and_visualize_data(
        data['detections_data']['usavp_data'], 
        data['detections_data']['ml_data'], 
        data['detections_data']['anomaly_data'], 
        request
    )

    # Return analysis results and plot URLs
    return analysis_results
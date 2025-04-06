import asyncio
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.com_app.com_data import com_listener
from app.db_ops.connector import get_db_session
from app.db_ops.db_create import AnomalyData, MLData, Trips, USAVPdata
from app.ml_app.ml_data import ml_listener
from ..routers.models import AnomalyDataResponse, MLDataResponse, TrackDataInput, TripStartDataInput, WarningInput
from ..routers.signal_router import warning
import datetime

trip_router = APIRouter(tags=['trip'], prefix='/trip')
loop = asyncio.get_event_loop()
tasks: Dict[int, List[asyncio.Task]] = {}

@trip_router.post('/put_usavp')
async def put_usavp_data(track_data: TrackDataInput, session: AsyncSession = Depends(get_db_session)):
    new_detector_data = USAVPdata(
        trip_id=track_data.trip_id,
        kilometer_value=track_data.kilometer_value,
        picket_value=track_data.picket_value,
        time_track=datetime.datetime.now()
    )

    async with session.begin():
        session.add(new_detector_data)
        await session.commit()  
    await session.refresh(new_detector_data) 
    return new_detector_data
    

@trip_router.post('/put_ml')
async def put_ml_data(track_data: TrackDataInput, session: AsyncSession = Depends(get_db_session)):
    # Ensure that the session is not in a pending transaction state
    # This is conceptual; SQLAlchemy sessions should be managed to avoid this situation

    # Perform operations within a single managed transaction
    new_ml_data = MLData(
            trip_id=track_data.trip_id,
            kilometer_value=track_data.kilometer_value,
            picket_value=track_data.picket_value,
            time_track=datetime.datetime.now()
        )
    new_anomaly_data = None
    async with session.begin():
        
        session.add(new_ml_data)
        await session.flush()  

        last_usavp_query = select(USAVPdata).order_by(USAVPdata.time_track.desc()).limit(1)
        last_usavp_data = await session.execute(last_usavp_query)
        last_usavp_data = last_usavp_data.scalars().first()
        
        if last_usavp_data and (last_usavp_data.kilometer_value != track_data.kilometer_value or
                                last_usavp_data.picket_value != track_data.picket_value):
            new_anomaly_data = AnomalyData(
                trip_id=track_data.trip_id,
                usavp_id=last_usavp_data.usavp_id,
                ml_id=new_ml_data.ml_id
            )
            session.add(new_anomaly_data)
            await session.flush() 
        
        ml_data_response = MLDataResponse(
            ml_id=new_ml_data.ml_id,
            trip_id=new_ml_data.trip_id,
            kilometer_value=new_ml_data.kilometer_value,
            picket_value=new_ml_data.picket_value,
            time_track=new_ml_data.time_track
        )
        
        anomaly_data_response = None
        if new_anomaly_data:
            anomaly_data_response = AnomalyDataResponse(
                anomaly_id=new_anomaly_data.anomaly_id,
                trip_id=new_anomaly_data.trip_id,
                usavp_id=new_anomaly_data.usavp_id,
                ml_id=new_anomaly_data.ml_id
            )
            await warning(WarningInput(
                anomaly_id=new_anomaly_data.anomaly_id,
                kilometer_value_usavp=last_usavp_data.kilometer_value, kilometer_value_ml=track_data.kilometer_value,
                picket_value_usavp=last_usavp_data.picket_value, picket_value_ml=track_data.picket_value
            ))
    return {"ml_data": ml_data_response, "anomaly_data": anomaly_data_response}

async def start_tasks_for_trip(trip_id: int):
    """Start all tasks for a given trip."""
    tasks[trip_id] = [
        asyncio.create_task(com_listener(trip_id)),
        asyncio.create_task(ml_listener(trip_id)),
    ]

async def stop_tasks_for_trip(trip_id: int):
    """Cancel all tasks for a given trip."""
    trip_tasks = tasks.get(trip_id, [])
    for task in trip_tasks:
        task.cancel()
    if trip_tasks:
        await asyncio.gather(*trip_tasks, return_exceptions=True)
    print(f"All tasks for trip {trip_id} were cancelled.")


@trip_router.post('/start_trip')
async def start_trip(trip_data: TripStartDataInput, session: AsyncSession = Depends(get_db_session)):
    new_trip_data = Trips(
        train_id=trip_data.train_id,
        route=trip_data.route,
        start_time=datetime.datetime.now(),
        finish_time=None
    )

    async with session.begin():
        session.add(new_trip_data)
        await session.commit()
    await session.refresh(new_trip_data)
    await start_tasks_for_trip(new_trip_data.trip_id)

    return {"message": "Trip and task started", "trip_id": new_trip_data.trip_id}

@trip_router.post('/end_trip/{trip_id}')
async def end_trip(trip_id: int, session: AsyncSession = Depends(get_db_session)):
    async with session.begin():
        # Retrieve the existing trip record
        stmt = select(Trips).filter(Trips.trip_id == trip_id)
        result = await session.execute(stmt)
        trip_record = result.scalars().first()
        
        if not trip_record:
            raise HTTPException(status_code=404, detail="Trip not found")

        # Update the finish_time
        trip_record.finish_time = datetime.datetime.now()
        response = {"message": "Trip and task ended", "trip_id": trip_id, "finish_time": trip_record.finish_time}
        await stop_tasks_for_trip(trip_id)
        

    return response

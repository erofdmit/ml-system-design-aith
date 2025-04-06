import asyncio

from fastapi import APIRouter, Body, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db_ops.connector import get_db_session
from app.db_ops.db_create import Trains, Trips
from ..routers.models import TrainDataInput, TripsDataInput
import asyncio

data_router = APIRouter(tags=['data'], prefix='/data')
loop = asyncio.get_event_loop()


@data_router.get('/get_train')
async def get_train_data(session: AsyncSession = Depends(get_db_session)):
    async with session.begin(): 
        q = select(Trains)
        result = await session.execute(q)
        data = result.scalars().all()
        return data


@data_router.post('/put_train')
async def put_train_data(train_data: TrainDataInput, session: AsyncSession = Depends(get_db_session)):
    new_detector_data = Trains(
        type=train_data.type,
        start_exp_date=train_data.start_exp_date
    )

    async with session.begin():
        session.add(new_detector_data)
        await session.commit()  
    await session.refresh(new_detector_data) 
    return new_detector_data
    
@data_router.get('/get_trip')
async def get_trip_data(session: AsyncSession = Depends(get_db_session)):
    async with session.begin(): 
        q = select(Trips)
        result = await session.execute(q)
        data = result.scalars().all()
        return data


@data_router.post('/put_trip')
async def put_trip_data(trip_data: TripsDataInput, session: AsyncSession = Depends(get_db_session)):
    new_detector_data = Trips(
        train_id=trip_data.train_id,
        route=trip_data.route,
        start_time=trip_data.start_time,
        finish_time=trip_data.finish_time
    )

    async with session.begin():
        session.add(new_detector_data)
        await session.commit()  
    await session.refresh(new_detector_data) 
    return new_detector_data
    
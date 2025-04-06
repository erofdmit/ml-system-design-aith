import asyncio

from fastapi import APIRouter, Body, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db_ops.connector import get_db_session
from app.db_ops.db_create import Trains, Trips
from ..routers.models import WarningInput


signal_router = APIRouter(tags=['signal'], prefix='/signal')
loop = asyncio.get_event_loop()


@signal_router.post('/warning')
async def warning(warning_data: WarningInput):
    print('____________________________________________')
    print('WARNING')
    print(warning_data)
    print('____________________________________________')




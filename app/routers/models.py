from typing import List, Optional
from pydantic import BaseModel, HttpUrl
import datetime

class TrainDataInput(BaseModel):
    type: str
    start_exp_date: datetime.datetime
    
class TripsDataInput(BaseModel):
    train_id: int
    route: str
    start_time: datetime.datetime
    finish_time: datetime.datetime
    
class TripStartDataInput(BaseModel):
    train_id: int
    route: str
    start_time: datetime.datetime
    
class TrackDataInput(BaseModel):
    trip_id: int
    kilometer_value: int
    picket_value: int
    time_track: datetime.datetime
    
class AnomalyDataInput(BaseModel):
    trip_id: int
    usavp_id: int
    ml_id: int

class DataByID(BaseModel):
    id: int
    
class MLDataResponse(BaseModel):
    ml_id: int
    trip_id: int
    kilometer_value: int
    picket_value: int
    time_track: datetime.datetime

class AnomalyDataResponse(BaseModel):
    anomaly_id: int | None  
    trip_id: int | None
    usavp_id: int | None
    ml_id: int | None
    
class WarningInput(BaseModel):
    anomaly_id: int | None  
    kilometer_value_ml: int
    picket_value_ml: int
    kilometer_value_usavp: int
    picket_value_usavp: int
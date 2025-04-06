from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import relationship
from ..settings import settings
from sqlalchemy.sql import func

Base = declarative_base()

class Trains(Base):
    __tablename__ = 'Trains'
    train_id = Column(Integer, primary_key=True)
    type = Column(String)
    start_exp_date = Column(DateTime(timezone=True))

class Trips(Base):
    __tablename__ = 'Trips'
    trip_id = Column(Integer, primary_key=True)
    train_id = Column(Integer, ForeignKey('Trains.train_id'))
    route = Column(String)
    start_time = Column(DateTime(timezone=True))
    finish_time = Column(DateTime(timezone=True))
    train = relationship("Trains", back_populates="trips")

class USAVPdata(Base):
    __tablename__ = 'USAVPdata'
    usavp_id = Column(Integer, primary_key=True)
    trip_id = Column(Integer, ForeignKey('Trips.trip_id'))
    kilometer_value = Column(Integer)
    picket_value = Column(Integer)
    time_track = Column(DateTime(timezone=True), server_default=func.now())
    trip = relationship("Trips", back_populates="usavpdata")

class MLData(Base):
    __tablename__ = 'MLData'
    ml_id = Column(Integer, primary_key=True)
    trip_id = Column(Integer, ForeignKey('Trips.trip_id'))
    kilometer_value = Column(Integer)
    picket_value = Column(Integer)
    time_track = Column(DateTime(timezone=True), server_default=func.now())
    trip = relationship("Trips", back_populates="mldata")

class AnomalyData(Base):
    __tablename__ = 'AnomalyData'
    anomaly_id = Column(Integer, primary_key=True)
    trip_id = Column(Integer, ForeignKey('Trips.trip_id'))
    usavp_id = Column(Integer, ForeignKey('USAVPdata.usavp_id'))
    ml_id = Column(Integer, ForeignKey('MLData.ml_id'))
    trip = relationship("Trips", back_populates="anomalydata")
    usavp = relationship("USAVPdata", back_populates="anomalies")
    mldata = relationship("MLData", back_populates="anomalies")

# Adding relationships to enable reverse queries
Trains.trips = relationship("Trips", order_by=Trips.trip_id, back_populates="train")
Trips.usavpdata = relationship("USAVPdata", order_by=USAVPdata.usavp_id, back_populates="trip")
Trips.mldata = relationship("MLData", order_by=MLData.ml_id, back_populates="trip")
Trips.anomalydata = relationship("AnomalyData", order_by=AnomalyData.anomaly_id, back_populates="trip")
USAVPdata.anomalies = relationship("AnomalyData", order_by=AnomalyData.anomaly_id, back_populates="usavp")
MLData.anomalies = relationship("AnomalyData", order_by=AnomalyData.anomaly_id, back_populates="mldata")

# Connect to the database (example connection string; should be customized for your environment)
async def create_detector_data_table_async(database_url):
    engine = create_async_engine(database_url, echo=True)
    
    # AsyncSession configuration
    async with engine.begin() as conn:
        # await the creation of all tables
        await conn.run_sync(Base.metadata.create_all)
    
    print("Table created successfully (async).")
    
    
import asyncio

# Assuming create_detector_data_table_async is defined as shown before

async def main():
    database_url = settings.DB_URL
    await create_detector_data_table_async(database_url)

# This is the standard way to run the main coroutine with asyncio
if __name__ == "__main__":
    asyncio.run(main())
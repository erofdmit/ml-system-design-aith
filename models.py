from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, String, func

from database import Base


class Recognition(Base):
    __tablename__ = "recognitions"

    id = Column(Integer, primary_key=True)
    text = Column(String(256), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

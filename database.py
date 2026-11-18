"""Database setup. Default = SQLite (zero setup). For PostgreSQL set DATABASE_URL, e.g.
postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/student_risk
"""
import os
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./student_risk.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class Base(DeclarativeBase):
    pass


def now():
    return datetime.now(timezone.utc)


class Student(Base):
    """One row per student prediction (loaded from at_risk_students.csv)."""
    __tablename__ = "students"
    row_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id_student: Mapped[int] = mapped_column(Integer, index=True)
    gender: Mapped[str] = mapped_column(String(5))
    region: Mapped[str] = mapped_column(String(60), index=True)
    age_band: Mapped[str] = mapped_column(String(20))
    avg_score: Mapped[float] = mapped_column(Float)
    active_days: Mapped[float] = mapped_column(Float)
    total_clicks: Mapped[float] = mapped_column(Float)
    predicted_risk: Mapped[int] = mapped_column(Integer, index=True)
    risk_probability: Mapped[float] = mapped_column(Float)


class Intervention(Base):
    """Mentor action log for a flagged student (ethical intervention workflow)."""
    __tablename__ = "interventions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id_student: Mapped[int] = mapped_column(Integer, index=True)
    mentor: Mapped[str] = mapped_column(String(80))
    action_type: Mapped[str] = mapped_column(String(20))
    notes: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="open")
    outcome: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

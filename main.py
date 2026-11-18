"""Student Risk Early-Warning API: prediction + students + interventions + impact."""
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import String, func, select
from sqlalchemy.orm import Session

from database import Base, Intervention, Student, engine, get_db

MODEL_PATH = os.getenv("MODEL_PATH", "risk_model.pkl")
COLUMNS_PATH = os.getenv("COLUMNS_PATH", "model_columns.pkl")
ml = {"model": None, "columns": None}
ref_cache = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    if os.path.exists(MODEL_PATH) and os.path.exists(COLUMNS_PATH):
        ml["model"] = joblib.load(MODEL_PATH)
        ml["columns"] = joblib.load(COLUMNS_PATH)
    yield


app = FastAPI(title="Student Risk Early-Warning API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

ActionType = Literal["call", "meeting", "email", "counselling", "other"]
Status = Literal["open", "in_progress", "resolved"]
Outcome = Literal["pending", "improved", "no_change"]


class StudentFeatures(BaseModel):
    gender: str
    region: str
    highest_education: str
    imd_band: str
    age_band: str
    disability: str
    num_of_prev_attempts: int
    studied_credits: int
    module_presentation_length: int
    total_clicks: float
    active_days: float
    avg_score: float
    total_assessments: float


class InterventionCreate(BaseModel):
    id_student: int
    mentor: str
    action_type: ActionType
    notes: str = ""


class InterventionUpdate(BaseModel):
    status: Optional[Status] = None
    outcome: Optional[Outcome] = None
    notes: Optional[str] = None


def student_dict(s: Student):
    return {c: getattr(s, c) for c in
            ["id_student", "gender", "region", "age_band", "avg_score", "active_days",
             "total_clicks", "predicted_risk", "risk_probability"]}


def intervention_dict(i: Intervention):
    return {"id": i.id, "id_student": i.id_student, "mentor": i.mentor, "action_type": i.action_type,
            "notes": i.notes, "status": i.status, "outcome": i.outcome,
            "created_at": i.created_at.isoformat(), "updated_at": i.updated_at.isoformat()}


def reference(db: Session):
    """Average behaviour of low-risk students (used to explain why someone is flagged)."""
    if not ref_cache:
        r = db.execute(select(func.avg(Student.avg_score), func.avg(Student.active_days),
                              func.avg(Student.total_clicks)).where(Student.predicted_risk == 0)).one()
        ref_cache.update(avg_score=r[0] or 0, active_days=r[1] or 0, total_clicks=r[2] or 0)
    return ref_cache


def reasons_for(avg_score, active_days, total_clicks, probability, db):
    ref, out = reference(db), []
    labels = {"avg_score": ("Average score", avg_score), "active_days": ("Active days", active_days),
              "total_clicks": ("Total clicks", total_clicks)}
    for key, (label, value) in labels.items():
        if ref[key] and value < 0.6 * ref[key]:
            out.append(f"{label} is {value:.0f}, well below the low-risk average ({ref[key]:.0f})")
    if probability >= 0.8:
        out.append(f"Model is highly confident ({probability:.0%} risk)")
    return out or ["No single metric stands out; risk comes from a combination of factors"]


@app.get("/")
def health(db: Session = Depends(get_db)):
    return {"status": "running", "model_loaded": ml["model"] is not None,
            "students_in_db": db.scalar(select(func.count()).select_from(Student))}


@app.post("/predict")
def predict(student: StudentFeatures, db: Session = Depends(get_db)):
    if ml["model"] is None:
        raise HTTPException(503, "Model not loaded. Put risk_model.pkl and model_columns.pkl next to main.py")
    df = pd.DataFrame([student.model_dump()])
    df = pd.get_dummies(df, columns=["gender", "region", "highest_education", "imd_band",
                                     "age_band", "disability"])
    df = df.reindex(columns=ml["columns"], fill_value=0).astype(float)
    pred = int(ml["model"].predict(df)[0])
    prob = round(float(ml["model"].predict_proba(df)[0][1]), 3)
    return {"at_risk": pred, "risk_probability": prob,
            "message": "High risk - mentor support recommended" if pred else "Low risk",
            "reasons": reasons_for(student.avg_score, student.active_days, student.total_clicks, prob, db) if pred else []}


@app.get("/students/summary")
def summary(db: Session = Depends(get_db)):
    total = db.scalar(select(func.count()).select_from(Student))
    risk = db.scalar(select(func.count()).select_from(Student).where(Student.predicted_risk == 1))

    def by(col):
        rows = db.execute(select(col, func.count()).where(Student.predicted_risk == 1)
                          .group_by(col).order_by(func.count().desc())).all()
        return [{"label": r[0], "at_risk": r[1]} for r in rows]

    return {"total": total, "at_risk": risk, "low_risk": total - risk,
            "at_risk_pct": round(100 * risk / total, 1) if total else 0,
            "by_region": by(Student.region), "by_gender": by(Student.gender),
            "by_age_band": by(Student.age_band)}


@app.get("/students/at-risk")
def at_risk(region: Optional[str] = None, gender: Optional[str] = None, age_band: Optional[str] = None,
            min_probability: float = Query(0.0, ge=0, le=1), search: Optional[str] = None,
            sort: Literal["risk_probability", "avg_score", "active_days", "total_clicks"] = "risk_probability",
            order: Literal["asc", "desc"] = "desc", limit: int = Query(50, ge=1, le=500),
            offset: int = Query(0, ge=0), db: Session = Depends(get_db)):
    q = select(Student).where(Student.predicted_risk == 1, Student.risk_probability >= min_probability)
    if region:
        q = q.where(Student.region == region)
    if gender:
        q = q.where(Student.gender == gender)
    if age_band:
        q = q.where(Student.age_band == age_band)
    if search:
        q = q.where(Student.id_student.cast(String).like(f"{search}%"))
    total = db.scalar(select(func.count()).select_from(q.subquery()))
    col = getattr(Student, sort)
    rows = db.scalars(q.order_by(col.desc() if order == "desc" else col.asc())
                      .offset(offset).limit(limit)).all()
    ids = [r.id_student for r in rows]
    counts = dict(db.execute(select(Intervention.id_student, func.count()).where(
        Intervention.id_student.in_(ids)).group_by(Intervention.id_student)).all()) if ids else {}
    return {"total": total, "limit": limit, "offset": offset,
            "items": [{**student_dict(r), "interventions": counts.get(r.id_student, 0)} for r in rows]}


@app.get("/students/{id_student}")
def student_detail(id_student: int, db: Session = Depends(get_db)):
    rows = db.scalars(select(Student).where(Student.id_student == id_student)
                      .order_by(Student.risk_probability.desc())).all()
    if not rows:
        raise HTTPException(404, "Student not found")
    s = rows[0]
    ivs = db.scalars(select(Intervention).where(Intervention.id_student == id_student)
                     .order_by(Intervention.created_at.desc())).all()
    return {"student": student_dict(s), "enrollments": len(rows),
            "reasons": reasons_for(s.avg_score, s.active_days, s.total_clicks, s.risk_probability, db)
            if s.predicted_risk else [],
            "interventions": [intervention_dict(i) for i in ivs]}


@app.post("/interventions", status_code=201)
def create_intervention(body: InterventionCreate, db: Session = Depends(get_db)):
    if not db.scalar(select(func.count()).select_from(Student).where(Student.id_student == body.id_student)):
        raise HTTPException(404, "Student not found")
    i = Intervention(**body.model_dump())
    db.add(i)
    db.commit()
    db.refresh(i)
    return intervention_dict(i)


@app.get("/interventions")
def list_interventions(id_student: Optional[int] = None, status: Optional[Status] = None,
                       limit: int = Query(100, ge=1, le=500), db: Session = Depends(get_db)):
    q = select(Intervention)
    if id_student:
        q = q.where(Intervention.id_student == id_student)
    if status:
        q = q.where(Intervention.status == status)
    return [intervention_dict(i) for i in db.scalars(q.order_by(Intervention.created_at.desc()).limit(limit))]


@app.patch("/interventions/{iid}")
def update_intervention(iid: int, body: InterventionUpdate, db: Session = Depends(get_db)):
    i = db.get(Intervention, iid)
    if not i:
        raise HTTPException(404, "Intervention not found")
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(i, k, v)
    i.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(i)
    return intervention_dict(i)


@app.get("/impact")
def impact(db: Session = Depends(get_db)):
    flagged = db.scalar(select(func.count(func.distinct(Student.id_student))).where(Student.predicted_risk == 1))
    reached = db.scalar(select(func.count(func.distinct(Intervention.id_student))))
    count_by = lambda col: dict(db.execute(select(col, func.count()).group_by(col)).all())
    outcomes = count_by(Intervention.outcome)
    done = outcomes.get("improved", 0) + outcomes.get("no_change", 0)
    return {"flagged_students": flagged, "students_reached": reached,
            "coverage_pct": round(100 * reached / flagged, 1) if flagged else 0,
            "by_status": count_by(Intervention.status), "by_outcome": outcomes,
            "improvement_rate_pct": round(100 * outcomes.get("improved", 0) / done, 1) if done else None,
            "note": "Improvement rate = improved / (improved + no_change), from mentor-logged outcomes."}

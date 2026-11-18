# Student Risk Backend (Step 1)

Folder me ye 7 files ek saath rakho: main.py, database.py, load_data.py, requirements.txt,
risk_model.pkl, model_columns.pkl, at_risk_students.csv

## Run
1. pip install -r requirements.txt
2. python load_data.py            (CSV ko database me daalta hai)
3. uvicorn main:app --reload
4. Browser: http://127.0.0.1:8000/docs

## PostgreSQL use karna ho (optional, default SQLite hai)
pgAdmin me ek database `student_risk` banao, phir terminal me (run se pehle):
  Windows PowerShell:  $env:DATABASE_URL="postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/student_risk"
Phir step 2 aur 3 dobara chalao.

## Endpoints
GET  /                      health
POST /predict               risk + reasons for a student's details
GET  /students/summary      totals + at-risk by region/gender/age
GET  /students/at-risk      list (filters: region, gender, age_band, min_probability, search; sort, limit, offset)
GET  /students/{id}         student + reasons + intervention history
POST /interventions         mentor action log
GET  /interventions         list (filter: id_student, status)
PATCH /interventions/{id}   update status / outcome / notes
GET  /impact                coverage + improvement rate

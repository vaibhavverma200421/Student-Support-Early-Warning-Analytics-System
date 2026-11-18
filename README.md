# Student Support & Early-Warning Analytics System

A predictive analytics platform that identifies students who are likely to **Fail or Withdraw** so mentors can step in early. It combines a machine learning model, a REST API with a database, and a mentor web portal for tracking support actions.

## Problem

Colleges usually find out a student is struggling only after results are declared, when it is too late to help. This system uses engagement (online activity), demographic and academic data to flag at-risk students early, explain *why* they were flagged, and let mentors log and track their interventions.

## Features

- **Risk prediction:** a Random Forest model flags at-risk students with a probability score.
- **Explainable flags:** the API returns the reasons a student was flagged.
- **REST API (FastAPI):** endpoints for prediction, student lists, interventions and impact.
- **Database:** students and mentor actions stored in SQLite (PostgreSQL optional).
- **Mentor portal (HTML):** login, dashboard, at-risk list with filters and CSV export, student detail panel, "Check a student" form, mentor action tracker and impact report.

## Dataset

- **Source:** Open University Learning Analytics Dataset (OULAD), https://analyse.kmi.open.ac.uk/open-dataset
- **Tables used:** studentInfo, studentRegistration, studentAssessment, studentVle, assessments, courses, vle
- **Note:** `studentVle.csv` is very large, so it is not included in this repository. Download the dataset from the link above to re-run the EDA notebook.

## Approach

1. Merged the raw tables into one student-level dataset (demographics, clicks, active days, scores).
2. Cleaned the data: handled duplicates and missing values.
3. Created the target `at_risk` (1 = Fail/Withdrawn, 0 = Pass/Distinction).
4. Removed result-related columns to avoid data leakage.
5. One-hot encoded categorical features (gender, region, education level, etc.).
6. Split the data 80% train / 20% test.
7. Trained a **Random Forest Classifier** and evaluated it with accuracy, precision, recall and a confusion matrix.

## Results

| Metric | Value |
|---|---|
| Accuracy | 90.4% |
| Precision (at-risk class) | 0.95 |
| Recall (at-risk class) | 0.87 |

## Tech Stack

- **Language:** Python
- **Data and ML:** pandas, numpy, scikit-learn, joblib
- **Backend:** FastAPI, Uvicorn, SQLAlchemy
- **Database:** SQLite (default), PostgreSQL (optional)
- **Frontend:** HTML, CSS, JavaScript
- **Tools:** Jupyter Notebook, VS Code

## Project Structure

```
├── EDA.ipynb                            # Data exploration, cleaning, merging
├── student_risk_prediction_model.ipynb  # Model training and evaluation
├── run_api.ipynb                        # Starts the API from a notebook
├── main.py                              # FastAPI app and endpoints
├── database.py                          # Database models and connection
├── load_data.py                         # Loads the CSV into the database
├── risk_model.pkl                       # Trained model
├── model_columns.pkl                    # Feature columns used by the model
├── at_risk_students.csv                 # Student-level data with risk scores
├── student_risk_portal.html             # Mentor web portal
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Load data into the database
```bash
python load_data.py
```
Expected output: `Loaded 32548 students into sqlite:///./student_risk.db`

### 3. Start the API
```bash
uvicorn main:app --reload
```
Open http://127.0.0.1:8000/docs to see all endpoints.

*If the terminal does not find Python, open `run_api.ipynb` and run its cell. It starts the same API from inside the notebook.*

### 4. Open the portal
Double-click `student_risk_portal.html` to open it in your browser (keep the API running).

**Demo login:** any mentor name with password `mentor123`.
*This is a demo login for showing the flow, not real security.*

### Optional: use PostgreSQL
Create a database named `student_risk` in pgAdmin, then set this before steps 2 and 3 (PowerShell):
```powershell
$env:DATABASE_URL="postgresql+psycopg://postgres:YOUR_PASSWORD@localhost:5432/student_risk"
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Risk score and reasons for a student's details |
| GET | `/students/summary` | Totals and at-risk breakdown by region, gender and age |
| GET | `/students/at-risk` | At-risk list (filters: region, gender, age_band, min_probability, search; sort, limit, offset) |
| GET | `/students/{id}` | Student details, reasons and intervention history |
| POST | `/interventions` | Log a mentor action |
| GET | `/interventions` | List actions (filter by id_student, status) |
| PATCH | `/interventions/{id}` | Update status, outcome or notes |
| GET | `/impact` | Coverage and improvement rate |

## Screenshots

<!-- Add images to a screenshots/ folder and link them here -->
![Dashboard](screenshots/dashboard.png)
![At-risk students](screenshots/at_risk_list.png)
![API docs](screenshots/api_docs.png)

## Limitations

- The portal login is a demo and has no real authentication.
- The model is trained on OULAD historical data and is not validated on a real college's data.

## Future Scope

- Power BI dashboard for management reporting
- Automatic retraining with Apache Airflow
- Docker packaging and cloud deployment
- Real authentication and role-based access for mentors
- Attendance and LMS interaction trends over time

## Team

- Vaibhav Verma
- [Teammate full name]

**Guide:** [Mentor name] | **Institute:** [College name]

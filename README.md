# Student Support & Early-Warning Analytics System

## Overview
A predictive analytics project that identifies students who are academically "at risk" (likely to Fail or Withdraw) using engagement, demographic, and academic performance data. Built to help instructors intervene early and provide timely support.

## Dataset
- **Source:** UCI Open University Learning Analytics Dataset (OULAD)
- **Tables used:** studentInfo, studentRegistration, studentAssessment, studentVle, assessments, courses

## Approach
1. Merged multiple raw tables into a single student-level dataset
2. Cleaned data — handled duplicates and missing values
3. Engineered a target variable `at_risk` (1 = Fail/Withdrawn, 0 = Pass/Top Performer)
4. Encoded categorical features (gender, region, education level, etc.) using one-hot encoding
5. Split data into 80% training / 20% testing
6. Trained a **Random Forest Classifier**
7. Evaluated performance using accuracy, precision, recall, and confusion matrix

## Results
- **Accuracy:** 90.4%
- **Precision (at-risk class):** 0.95
- **Recall (at-risk class):** 0.87

## Tech Stack
- Python
- pandas, numpy
- scikit-learn
- Jupyter Notebook

## How to Run
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Open `student_risk_prediction_model.ipynb` in Jupyter
4. Run all cells

## Future Scope
- Deploy as an interactive dashboard for instructors
- Add live prediction for new/incoming student data
- Expand model to include attendance and LMS interaction trends

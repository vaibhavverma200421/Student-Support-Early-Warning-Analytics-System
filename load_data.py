"""Loads at_risk_students.csv into the database.  Usage: python load_data.py [path_to_csv]"""
import sys

import pandas as pd

from database import Base, Student, engine

path = sys.argv[1] if len(sys.argv) > 1 else "at_risk_students.csv"
df = pd.read_csv(path)
Base.metadata.create_all(engine)
with engine.begin() as conn:                      # reload = clear old students first
    conn.execute(Student.__table__.delete())
df.to_sql("students", engine, if_exists="append", index=False, chunksize=2000)
print(f"Loaded {len(df)} students into {engine.url.render_as_string(hide_password=True)}")

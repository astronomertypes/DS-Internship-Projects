import pandas as pd
from sqlalchemy import text
from src.db import get_engine

CSV = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(CSV)
# TotalCharges has 11 blank strings (customers with tenure 0) -> NaN -> 0
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
df.columns = [c.lower() for c in df.columns]
df = df.rename(columns={"customerid": "customer_id"})
df["churn"] = (df["churn"] == "Yes").astype(int)

engine = get_engine()
with engine.begin() as conn:
    conn.execute(text("DROP VIEW IF EXISTS customers_features"))  # lets us re-run safely
df.to_sql("customers_raw", engine, if_exists="replace", index=False)

with open("sql/features.sql") as f:
    sql = f.read()
with engine.begin() as conn:
    conn.execute(text(sql))
    n = conn.execute(text("SELECT COUNT(*) FROM customers_features")).scalar()
    rate = conn.execute(text("SELECT AVG(churn) FROM customers_features")).scalar()
print(f"Loaded {n} rows, churn rate = {rate:.3f}")
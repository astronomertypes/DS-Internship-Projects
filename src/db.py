import pandas as pd
from sqlalchemy import create_engine

DB_URL = "postgresql+psycopg2://churn:churn_pw@localhost:5432/churndb"

def get_engine():
    return create_engine(DB_URL)

def load_features() -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM customers_features", get_engine())
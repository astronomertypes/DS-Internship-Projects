"""Step 4: baseline forecasts - seasonal naive and Prophet."""
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from prophet import Prophet
from utils import PROC, OUT, evaluate, save_metrics

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)  # silence Prophet's chatter
logging.getLogger("prophet").setLevel(logging.ERROR)

df = pd.read_parquet(PROC / "features.parquet", columns=["id", "store_id", "date", "sales", "is_test"])
train, test = df[~df["is_test"]], df[df["is_test"]].copy()
test_dates = np.sort(test["date"].unique())

# ---------- 1) seasonal naive: repeat the last 7 known days ----------
rows = []
for k, g in train.groupby("id"):
    last7 = g.sort_values("date")["sales"].values[-7:]
    for h, d in enumerate(test_dates):
        rows.append((k, d, float(last7[h % 7])))
naive = pd.DataFrame(rows, columns=["id", "date", "pred"])
naive.to_parquet(OUT / "pred_naive.parquet", index=False)
save_metrics(evaluate(test.merge(naive, on=["id", "date"]), train, "Seasonal naive"))

# ---------- 2) Prophet, one model per series ----------
frames = []
for k, g in tqdm(train.groupby("id"), desc="Prophet"):
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, uncertainty_samples=0)
    m.add_country_holidays(country_name="US")
    m.fit(pd.DataFrame({"ds": g["date"].values, "y": g["sales"].astype(float).values}))
    fc = m.predict(pd.DataFrame({"ds": test_dates}))
    frames.append(pd.DataFrame({"id": k, "date": test_dates, "pred": fc["yhat"].clip(lower=0).values}))
prophet = pd.concat(frames, ignore_index=True)
prophet.to_parquet(OUT / "pred_prophet.parquet", index=False)
save_metrics(evaluate(test.merge(prophet, on=["id", "date"]), train, "Prophet"))
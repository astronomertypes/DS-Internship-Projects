"""Diagnostics for the TFT: bias of the total forecast and prediction interval coverage."""
import pandas as pd
from utils import PROC, OUT

df = pd.read_parquet(PROC / "features.parquet", columns=["id", "store_id", "date", "sales", "is_test"])
test = df[df["is_test"]][["id", "store_id", "date", "sales"]]

models = {
    "Seasonal naive": pd.read_parquet(OUT / "pred_naive.parquet"),
    "Prophet": pd.read_parquet(OUT / "pred_prophet.parquet"),
    "LightGBM": pd.read_parquet(OUT / "pred_lgbm.parquet"),
    "TFT": pd.read_parquet(OUT / "pred_tft.parquet"),
}

# 1) bias: total forecast divided by total actual sales (1.00 = unbiased, below 1 = under-forecasts)
rows = []
for name, p in models.items():
    m = test.merge(p[["id", "date", "pred"]], on=["id", "date"])
    by_store = m.groupby("store_id")[["pred", "sales"]].sum()
    rows.append({"model": name,
                 "forecast_over_actual": m["pred"].sum() / m["sales"].sum(),
                 "stores_under_forecast_of_10": int((by_store["pred"] < by_store["sales"]).sum())})
print(pd.DataFrame(rows).round(4).to_string(index=False))

# 2) interval coverage and average width (a wide interval covers more, so look at both)
for name in ("LightGBM", "TFT"):
    m = test.merge(models[name], on=["id", "date"])
    cover = ((m["sales"] >= m["lo"]) & (m["sales"] <= m["hi"])).mean()
    print(f"{name}: 80% interval coverage {cover:.1%}, average width {(m['hi'] - m['lo']).mean():.2f} units per day")
"""Step 9a: collect all results into app/data/ so the dashboard (and Docker) don't need raw data."""
import shutil
import pandas as pd
from utils import PROC, OUT, ROOT

APP = ROOT / "app" / "data"
APP.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(PROC / "features.parquet", columns=["id", "item_id", "store_id", "date", "sales", "is_test"])
df[df["date"] >= df["date"].max() - pd.Timedelta(days=120)].to_parquet(APP / "history.parquet", index=False)

fc = pd.read_parquet(OUT / "pred_lgbm.parquet").rename(columns={"pred": "lgbm", "lo": "lgbm_lo", "hi": "lgbm_hi"})
for name, fn in (("naive", "pred_naive"), ("prophet", "pred_prophet")):
    fc = fc.merge(pd.read_parquet(OUT / f"{fn}.parquet").rename(columns={"pred": name}), on=["id", "date"], how="left")
fc = fc.merge(pd.read_parquet(OUT / "pred_tft.parquet").rename(columns={"pred": "tft", "lo": "tft_lo", "hi": "tft_hi"}),
              on=["id", "date"], how="left")
fc = fc.merge(pd.read_parquet(OUT / "pred_lgbm_reconciled.parquet").rename(columns={"pred_rec": "lgbm_reconciled"}),
              on=["id", "date"], how="left")
fc.to_parquet(APP / "forecasts.parquet", index=False)

for f in ("metrics.csv", "inventory_results.csv", "order_recommendations.csv", "reconciliation_metrics.csv"):
    shutil.copy(OUT / f, APP / f)
print("exported to", APP)
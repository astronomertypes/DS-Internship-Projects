"""Step 2a: cut the huge M5 data down to our 500-series subset and reshape it to one row per item-store-day."""
import numpy as np
import pandas as pd
from utils import RAW, PROC

DEPT, N_ITEMS, KEEP_DAYS = "FOODS_3", 50, 900

sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
cal = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
prices = pd.read_csv(RAW / "sell_prices.csv")

# keep one department, then its 50 best-selling items (by the last 365 days)
sales = sales[sales["dept_id"] == DEPT].copy()
day_cols = [c for c in sales.columns if c.startswith("d_")]
top_items = sales.groupby("item_id")[day_cols[-365:]].sum().sum(axis=1).nlargest(N_ITEMS).index
sales = sales[sales["item_id"].isin(top_items)]

# keep only the most recent 900 days (keeps things fast, still ~2.5 years)
day_cols = day_cols[-KEEP_DAYS:]

# wide (one column per day) -> long (one row per day)
long = sales.melt(
    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    value_vars=day_cols, var_name="d", value_name="sales",
)

# attach calendar info and prices
long = long.merge(
    cal[["d", "date", "wm_yr_wk", "event_name_1", "snap_CA", "snap_TX", "snap_WI"]], on="d", how="left"
)
long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
long = long.sort_values(["id", "date"]).reset_index(drop=True)

# a missing price means "not on the shelf that week"; remember that, then fill the gap
long["on_sale"] = long["sell_price"].notna().astype(int)
long["sell_price"] = long.groupby("id")["sell_price"].transform(lambda s: s.ffill().bfill())

# SNAP flag for the row's own state only
long["snap"] = np.select(
    [long["state_id"] == "CA", long["state_id"] == "TX"], [long["snap_CA"], long["snap_TX"]],
    default=long["snap_WI"],
)
long["sales"] = long["sales"].astype("float32")
long = long.drop(columns=["snap_CA", "snap_TX", "snap_WI", "dept_id", "cat_id", "d", "wm_yr_wk"])

long.to_parquet(PROC / "long.parquet", index=False)
print("rows:", len(long), "| series:", long["id"].nunique(), "| dates:", long["date"].min(), "->", long["date"].max())
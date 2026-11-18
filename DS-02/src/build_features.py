"""Step 3: lags, rolling averages, holidays, prices, promo proxy. Marks the last 28 days as the test set."""
import pandas as pd
from utils import PROC, HORIZON

df = pd.read_parquet(PROC / "long.parquet").sort_values(["id", "date"]).reset_index(drop=True)

# --- lag features (all >= 28 days back so they are available at forecast time) ---
g = df.groupby("id")["sales"]
for lag in (28, 35, 42, 49, 56):
    df[f"lag_{lag}"] = g.shift(lag)

# --- rolling statistics computed on the 28-day-shifted series ---
shifted = g.shift(28)
for w in (7, 28, 56):
    df[f"roll_mean_{w}"] = shifted.groupby(df["id"]).transform(lambda s: s.rolling(w, min_periods=1).mean())
df["roll_std_28"] = shifted.groupby(df["id"]).transform(lambda s: s.rolling(28, min_periods=2).std())

# --- price features ---
df["price_median"] = df.groupby("id")["sell_price"].transform("median")
df["price_ratio"] = df["sell_price"] / df["price_median"]
df["promo"] = (df["price_ratio"] < 0.95).astype(int)  # promo proxy: price at least 5% below normal

# --- calendar features ---
df["dow"] = df["date"].dt.dayofweek
df["dom"] = df["date"].dt.day
df["woy"] = df["date"].dt.isocalendar().week.astype(int)
df["month"] = df["date"].dt.month
df["is_event"] = df["event_name_1"].notna().astype(int)  # holiday / special event

# --- integer codes for categorical columns (LightGBM likes these) ---
for c in ("item_id", "store_id", "state_id"):
    df[c.replace("_id", "_code")] = df[c].astype("category").cat.codes

# --- train/test flag: the last 28 days are the test period ---
df["is_test"] = df["date"] > df["date"].max() - pd.Timedelta(days=HORIZON)
df = df.drop(columns=["event_name_1"])

df.to_parquet(PROC / "features.parquet", index=False)
print(df.shape, "| test rows:", int(df["is_test"].sum()))

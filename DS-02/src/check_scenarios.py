"""Diagnostic: are the optimizer's demand scenarios as wide as the real forecast errors?"""
import numpy as np
import pandas as pd
from utils import PROC, OUT
from optimize import build_scenarios, PLAN_DAYS

df = pd.read_parquet(PROC / "features.parquet", columns=["id", "date", "sales", "is_test"])
test = df[df["is_test"]]
f = pd.read_parquet(OUT / "pred_lgbm.parquet").merge(
    pd.read_parquet(OUT / "pred_lgbm_reconciled.parquet"), on=["id", "date"])

window = sorted(f["date"].unique())[:PLAN_DAYS]
agg = build_scenarios(f[f["date"].isin(window)])
actual = test[test["date"].isin(window)].groupby("id")["sales"].sum().rename("actual")
a = agg.merge(actual, on="id")

z = (a["actual"] - a["d_mid"]) / a["sd"].replace(0, np.nan)
print("std of z (about 1.0 = scenarios have the right width):", round(z.std(), 2))
print("share of items with actual above d_hi:", round((a["actual"] > a["d_hi"]).mean(), 3), "(about 0.10 if width is right)")
print("share of items with actual below d_lo:", round((a["actual"] < a["d_lo"]).mean(), 3), "(about 0.10 if width is right)")
"""Shared paths, metrics and evaluation helpers used by every script."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, PROC, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "outputs"
for p in (RAW, PROC, OUT):
    p.mkdir(parents=True, exist_ok=True)

HORIZON = 28  # we forecast the last 28 days of the data (M5 standard)


def wape(y, yhat):
    """Weighted Absolute Percentage Error = total absolute error / total actual sales."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def rmsse(y_train, y_true, y_pred):
    """Root Mean Squared Scaled Error (M5 metric): error relative to a 'yesterday = today' naive model."""
    y_train = np.trim_zeros(np.asarray(y_train, float), "f")  # ignore days before the item existed
    if len(y_train) < 2:
        return np.nan
    scale = np.mean(np.diff(y_train) ** 2)
    if scale == 0:
        return np.nan
    err = np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)
    return float(np.sqrt(err / scale))


def evaluate(pred, train, name):
    """pred needs columns: id, store_id, date, sales (actual), pred. train needs: id, sales."""
    store = pred.groupby(["store_id", "date"], as_index=False)[["sales", "pred"]].sum()
    train_vals = {k: v["sales"].values for k, v in train.groupby("id")}
    scores = [rmsse(train_vals[k], g["sales"].values, g["pred"].values) for k, g in pred.groupby("id")]
    return {
        "model": name,
        "WAPE_item_store": wape(pred["sales"], pred["pred"]),
        "WAPE_store": wape(store["sales"], store["pred"]),
        "RMSSE": float(np.nanmean(scores)),
    }


def save_metrics(row):
    """Add/replace one model's row in outputs/metrics.csv and print the table."""
    f = OUT / "metrics.csv"
    new = pd.DataFrame([row])
    if f.exists():
        old = pd.read_csv(f)
        new = pd.concat([old[old["model"] != row["model"]], new], ignore_index=True)
    new.to_csv(f, index=False)
    print(new.round(4).to_string(index=False))


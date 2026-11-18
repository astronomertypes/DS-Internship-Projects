"""Step 7: make forecasts add up across the hierarchy: item-store -> store -> state -> total."""
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from prophet import Prophet
from darts import TimeSeries
from darts.dataprocessing.transformers import BottomUpReconciliator, MinTReconciliator
from utils import PROC, OUT, wape

logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)

df = pd.read_parquet(PROC / "features.parquet", columns=["id", "store_id", "state_id", "date", "sales", "is_test"])
lgb = pd.read_parquet(OUT / "pred_lgbm.parquet")


def make_wide(d):
    """One column per node of the hierarchy (total, 3 states, 10 stores, 500 item-stores), one row per day."""
    bottom = d.pivot(index="date", columns="id", values="sales")
    store = d.groupby(["date", "store_id"])["sales"].sum().unstack()
    state = d.groupby(["date", "state_id"])["sales"].sum().unstack()
    total = d.groupby("date")["sales"].sum().rename("total")
    w = pd.concat([total, state, store, bottom], axis=1).astype(float)
    w.columns.name = None
    return w


wide_tr, wide_te = make_wide(df[~df["is_test"]]), make_wide(df[df["is_test"]])
test_dates = wide_te.index

# ---- hierarchy: each node -> its parent ----
pairs = df[["id", "store_id", "state_id"]].drop_duplicates()
hier = {r.id: [r.store_id] for r in pairs.itertuples()}
hier.update({s: [st] for s, st in pairs[["store_id", "state_id"]].drop_duplicates().itertuples(index=False)})
states = sorted(pairs["state_id"].unique())
stores = sorted(pairs["store_id"].unique())
bottoms = sorted(pairs["id"].unique())
hier.update({st: ["total"] for st in states})


def prophet_fc(series, dates):
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, uncertainty_samples=0)
    m.add_country_holidays(country_name="US")
    m.fit(pd.DataFrame({"ds": series.index, "y": series.values}))
    return m.predict(pd.DataFrame({"ds": dates}))["yhat"].clip(lower=0).values


# ---- base forecasts: Prophet for upper levels, LightGBM for the bottom ----
cols = {c: prophet_fc(wide_tr[c], test_dates) for c in tqdm(["total"] + states + stores, desc="Prophet upper levels")}
lgb_wide = lgb.pivot(index="date", columns="id", values="pred").reindex(test_dates)
for c in bottoms:
    cols[c] = lgb_wide[c].values
base = pd.DataFrame(cols, index=test_dates)[wide_tr.columns]


def to_df(ts):  # Darts renamed this method between versions
    return ts.to_dataframe() if hasattr(ts, "to_dataframe") else ts.pd_dataframe()


train_ts = TimeSeries.from_dataframe(wide_tr, hierarchy=hier)
base_ts = TimeSeries.from_dataframe(base, hierarchy=hier)

results = {"base (unreconciled)": base}

# Bottom-up: item-store forecasts are summed upward (no fitting needed)
try:
    results["bottom_up"] = to_df(BottomUpReconciliator().transform(base_ts))[wide_tr.columns]
except Exception as e:
    print(f"[warn] bottom_up failed: {e}")

# MinT with structural weights: adjusts ALL levels so they add up, with minimal change
try:
    mint = MinTReconciliator(method="wls_struct")
    mint.fit(train_ts)
    results["mint_wls_struct"] = to_df(mint.transform(base_ts))[wide_tr.columns]
except Exception as e:
    print(f"[warn] mint_wls_struct failed: {e}")


# children of every parent node: store -> its items, state -> its stores, total -> its states
children_of = {}
for child, parents in hier.items():
    children_of.setdefault(parents[0], []).append(child)


def coherence_by_link(f):
    """Largest mismatch (units per day) for each link of the hierarchy: parent forecast minus the sum of its children."""
    links = {"items -> store": stores, "stores -> state": states, "states -> total": ["total"]}
    return {name: float(max(np.abs(f[p].values - f[children_of[p]].sum(axis=1).values).max() for p in ps))
            for name, ps in links.items()}


def coherence_gap(f):
    """Largest mismatch over all links. This is the number saved in the coherence_gap column."""
    return max(coherence_by_link(f).values())


levels = {"total": ["total"], "state": states, "store": stores, "item_store": bottoms}
rows = []
for name, f in results.items():
    gap = coherence_gap(f)
    for lvl, cs in levels.items():
        rows.append({"forecast": name, "level": lvl, "WAPE": wape(wide_te[cs].values, f[cs].values), "coherence_gap": gap})
report = pd.DataFrame(rows)
report.to_csv(OUT / "reconciliation_metrics.csv", index=False)
print(report.round(4).to_string(index=False))
print("\nCoherence gap by link (units per day):")
for name, f in results.items():
    print(f"{name:22s}", {k: round(v, 3) for k, v in coherence_by_link(f).items()})

# ---- pick the best coherent method by item-level WAPE and save its bottom level ----
cand = report[(report["level"] == "item_store") & (report["coherence_gap"] < 1e-3)]
chosen = cand.sort_values("WAPE").iloc[0]["forecast"]
print("Chosen reconciled forecast:", chosen)
rec_bottom = results[chosen][bottoms].clip(lower=0)
rec_long = rec_bottom.stack().rename("pred_rec").reset_index()
rec_long.columns = ["date", "id", "pred_rec"]
rec_long.to_parquet(OUT / "pred_lgbm_reconciled.parquet", index=False)
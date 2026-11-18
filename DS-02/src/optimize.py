"""Step 8: choose order quantities with OR-Tools (scenario-based stochastic MIP, solved per store)."""
from statistics import NormalDist
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from utils import PROC, OUT

# ---------------- assumptions (document these in the README) ----------------
PLAN_DAYS = 14          # we order for the next 14 days
COST_RATIO = 0.70       # unit cost as share of selling price
HOLD_RATE = 0.05        # holding cost per leftover unit = 5% of unit cost
BUDGET_FRACTION = 0.85  # store budget vs. the cost of ordering up to the 90th percentile of demand
ON_HAND_DAYS = 2        # assumed current stock = 2 days of expected demand

Z80 = 1.2816            # an 80% interval is +/- 1.2816 standard deviations
RHO = 0.2               # how much daily forecast errors move together (0 = independent, 1 = fully together)
SPREAD = 1.2            # widening factor, calibrated with check_scenarios.py (std of z was 1.2)
N_SCEN = 9              # number of equally likely demand scenarios per item


def build_scenarios(w):
    """w: daily forecasts with columns id, pred_rec (mean), lo, hi (80% interval).
    Returns one row per id: d_mid (expected demand over the window), sd (its standard deviation),
    and d_lo / d_hi (the 80% interval of the window total)."""
    w = w.copy()
    w["lo"] = np.minimum(w["lo"], w["pred_rec"])   # keep lo <= mean <= hi
    w["hi"] = np.maximum(w["hi"], w["pred_rec"])
    w["sd_day"] = (w["hi"] - w["lo"]) / (2 * Z80)  # daily standard deviation from the interval
    w["var_day"] = w["sd_day"] ** 2
    agg = w.groupby("id").agg(d_mid=("pred_rec", "sum"), sum_var=("var_day", "sum"),
                              sum_sd=("sd_day", "sum")).reset_index()
    # sd of the window total when daily errors are partly correlated, widened by the calibration factor
    agg["sd"] = SPREAD * np.sqrt((1 - RHO) * agg["sum_var"] + RHO * agg["sum_sd"] ** 2)
    agg["d_lo"] = np.maximum(agg["d_mid"] - Z80 * agg["sd"], 0)
    agg["d_hi"] = agg["d_mid"] + Z80 * agg["sd"]
    return agg[["id", "d_mid", "sd", "d_lo", "d_hi"]]


def add_scenarios(d, k=N_SCEN):
    """Adds k equally likely demand scenarios s_0..s_{k-1} (columns), spaced along a normal curve."""
    nd = NormalDist()
    for i in range(k):
        z = nd.inv_cdf((i + 0.5) / k)
        d[f"s_{i}"] = np.maximum(d["d_mid"] + z * d["sd"], 0)  # demand cannot be negative
    return d


def solve_store(d, budget, time_limit_ms=30000):
    """d: one row per item with columns cost, hold, short, on_hand, ub and scenario columns s_0..s_k.
    Returns the order quantity (integer) per row."""
    scen_cols = [c for c in d.columns if c.startswith("s_")]
    prob = 1.0 / len(scen_cols)
    s = pywraplp.Solver.CreateSolver("SCIP")
    s.SetTimeLimit(time_limit_ms)
    inf = s.infinity()
    q, objective = {}, []
    for i, r in d.iterrows():
        q[i] = s.IntVar(0, int(r["ub"]), f"q_{i}")                    # units to order
        for k, col in enumerate(scen_cols):
            over = s.NumVar(0, inf, f"over_{i}_{k}")                  # leftover stock in this scenario
            under = s.NumVar(0, inf, f"under_{i}_{k}")                # unmet demand in this scenario
            s.Add(float(r["on_hand"]) + q[i] - float(r[col]) == over - under)
            objective.append(prob * (float(r["hold"]) * over + float(r["short"]) * under))
    s.Add(s.Sum([float(r["cost"]) * q[i] for i, r in d.iterrows()]) <= budget)  # spending limit
    s.Minimize(s.Sum(objective))
    status = s.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError("OR-Tools found no solution")
    return pd.Series({i: q[i].solution_value() for i in q}).round().astype(int)


def realized(d, q, label):
    """Cost of an order policy against what REALLY sold in the planning window."""
    avail = d["on_hand"] + q
    over, under = np.maximum(avail - d["actual"], 0), np.maximum(d["actual"] - avail, 0)
    hold, short = (d["hold"] * over).sum(), (d["short"] * under).sum()
    return {"policy": label, "holding_cost": hold, "stockout_cost": short, "total_cost": hold + short,
            "fill_rate": np.minimum(avail, d["actual"]).sum() / d["actual"].sum(),
            "order_spend": (d["cost"] * q).sum()}


if __name__ == "__main__":
    df = pd.read_parquet(PROC / "features.parquet",
                         columns=["id", "item_id", "store_id", "date", "sales", "sell_price", "is_test"])
    test = df[df["is_test"]]
    f = pd.read_parquet(OUT / "pred_lgbm.parquet").merge(
        pd.read_parquet(OUT / "pred_lgbm_reconciled.parquet"), on=["id", "date"])

    window = sorted(f["date"].unique())[:PLAN_DAYS]
    agg = build_scenarios(f[f["date"].isin(window)])

    actual = test[test["date"].isin(window)].groupby("id")["sales"].sum().rename("actual")
    price = test.sort_values("date").groupby("id")["sell_price"].last().rename("price")
    meta = test[["id", "item_id", "store_id"]].drop_duplicates()
    d = agg.merge(actual, on="id").merge(price, on="id").merge(meta, on="id")
    d = add_scenarios(d)

    d["cost"] = COST_RATIO * d["price"]
    d["hold"] = HOLD_RATE * d["cost"]
    d["short"] = d["price"] - d["cost"]
    d["on_hand"] = np.round(ON_HAND_DAYS * d["d_mid"] / PLAN_DAYS)
    d["need_hi"] = np.maximum(d["d_hi"] - d["on_hand"], 0)   # units needed to reach the 90th percentile
    d["ub"] = np.ceil(1.5 * d["d_hi"]).astype(int)           # max units per item
    d["order_qty"] = 0

    for store, g in d.groupby("store_id"):
        budget = BUDGET_FRACTION * float((g["cost"] * g["need_hi"]).sum())
        d.loc[g.index, "order_qty"] = solve_store(g, budget)
        print(store, "solved | budget:", round(budget, 2))

    # policy 1: order the expected demand (no budget limit)
    naive_q = np.maximum(np.round(d["d_mid"] - d["on_hand"]), 0)

    # policy 2: policy 1 scaled per store to spend the same money as the optimized plan (fair comparison)
    naive_spend = (d["cost"] * naive_q).groupby(d["store_id"]).transform("sum")
    opt_spend = (d["cost"] * d["order_qty"]).groupby(d["store_id"]).transform("sum")
    scaled_q = np.round(naive_q * opt_spend / naive_spend)

    res = pd.DataFrame([
        realized(d, naive_q, "Naive (order forecast mean)"),
        realized(d, scaled_q, "Naive scaled to same spend"),
        realized(d, d["order_qty"], "Optimized (OR-Tools)"),
    ])
    res.to_csv(OUT / "inventory_results.csv", index=False)
    print(res.round(3).to_string(index=False))

    d.rename(columns={"d_lo": "forecast_lo", "d_mid": "forecast_mid", "d_hi": "forecast_hi"})[
        ["store_id", "item_id", "on_hand", "forecast_lo", "forecast_mid", "forecast_hi", "order_qty"]
    ].to_csv(OUT / "order_recommendations.csv", index=False)
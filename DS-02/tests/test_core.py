"""Step 11: unit tests for metrics, scenario building, the cost function and the optimizer."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import wape, rmsse
from optimize import build_scenarios, add_scenarios, realized, solve_store, Z80, SPREAD, N_SCEN


# ---------------- metrics ----------------
def test_wape_perfect_is_zero():
    assert wape([1, 2, 3], [1, 2, 3]) == 0


def test_wape_known_value():
    assert wape([10, 10], [8, 12]) == pytest.approx(0.2)  # total error 4 / total sales 20


def test_rmsse_known_value():
    # train diffs are all 1 -> scale 1; errors (0, 2) -> mse 2 -> sqrt(2)
    assert rmsse([1, 2, 3, 4], [5, 6], [5, 8]) == pytest.approx(np.sqrt(2))


# ---------------- scenarios ----------------
def _daily_forecast(n_days=14, mean=10.0, lo=0.0, hi=20.0):
    return pd.DataFrame({"id": ["A"] * n_days, "pred_rec": [mean] * n_days,
                         "lo": [lo] * n_days, "hi": [hi] * n_days})


def test_window_interval_is_ordered_and_nonnegative():
    s = build_scenarios(_daily_forecast()).iloc[0]
    assert s["d_mid"] == pytest.approx(140.0)
    assert 0 <= s["d_lo"] <= s["d_mid"] <= s["d_hi"]


def test_window_spread_never_exceeds_fully_correlated_bound():
    s = build_scenarios(_daily_forecast()).iloc[0]
    daily_sd = (20.0 - 0.0) / (2 * Z80)
    assert s["d_hi"] - s["d_mid"] <= Z80 * SPREAD * 14 * daily_sd + 1e-9


def test_no_spread_when_forecast_is_certain():
    s = build_scenarios(_daily_forecast(lo=10.0, hi=10.0)).iloc[0]
    assert s["d_lo"] == pytest.approx(s["d_mid"]) and s["d_hi"] == pytest.approx(s["d_mid"])


def test_scenario_columns_are_sorted_and_centered():
    d = add_scenarios(pd.DataFrame({"d_mid": [100.0], "sd": [10.0]}))
    cols = [f"s_{i}" for i in range(N_SCEN)]
    vals = d[cols].iloc[0].values
    assert len(vals) == N_SCEN
    assert (np.diff(vals) >= 0).all()          # increasing
    assert vals.mean() == pytest.approx(100.0)  # centered on the expected demand


# ---------------- cost function ----------------
def test_realized_cost_is_zero_when_order_matches_demand():
    d = pd.DataFrame({"actual": [10.0, 20.0], "on_hand": [2.0, 5.0], "hold": [0.1, 0.2],
                      "short": [0.5, 0.6], "cost": [1.0, 2.0]})
    q = pd.Series([8.0, 15.0])  # on_hand + q == actual
    r = realized(d, q, "perfect")
    assert r["total_cost"] == pytest.approx(0)
    assert r["fill_rate"] == pytest.approx(1.0)


# ---------------- optimizer ----------------
def _items():
    d = pd.DataFrame({"d_mid": [100.0, 100.0, 100.0], "sd": [10.0, 20.0, 30.0],
                      "cost": [1.0, 2.0, 3.0], "hold": [0.05, 0.1, 0.15], "short": [0.3, 0.6, 0.9],
                      "on_hand": [0.0, 0.0, 0.0], "ub": [300, 300, 300]})
    return add_scenarios(d)


def test_optimizer_respects_budget_and_bounds():
    d, budget = _items(), 400.0
    q = solve_store(d, budget)
    assert (q >= 0).all() and (q <= d["ub"]).all()
    assert (d["cost"] * q).sum() <= budget + 1e-6


def test_optimizer_orders_more_with_a_bigger_budget():
    d = _items()
    small = (d["cost"] * solve_store(d, 300.0)).sum()
    large = (d["cost"] * solve_store(d, 100000.0)).sum()
    assert large >= small


def test_optimizer_orders_above_the_mean_when_shortage_costs_more_than_leftovers():
    # critical ratio = short / (short + hold) = 0.9, so with ample budget the order should exceed expected demand
    d = add_scenarios(pd.DataFrame({"d_mid": [100.0], "sd": [10.0], "cost": [1.0], "hold": [0.1],
                                    "short": [0.9], "on_hand": [0.0], "ub": [300]}))
    q = solve_store(d, 1e9).iloc[0]
    assert 100 < q <= 120
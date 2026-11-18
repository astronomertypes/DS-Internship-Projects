import numpy as np
import pandas as pd
from src.features import clean
from src.metrics import qini_coefficient
from src.clv import expected_discounted_months

def test_clean_merges_no_service_levels():
    df = pd.DataFrame({c: ["No internet service"] for c in
        ["multiplelines","onlinesecurity","onlinebackup","deviceprotection",
         "techsupport","streamingtv","streamingmovies"]})
    assert (clean(df) == "No").all().all()

def test_no_churn_gives_full_horizon():
    S = np.ones((73, 1))
    m = expected_discounted_months(S, np.array([10]), horizon=24, annual_discount=0.0)
    assert np.isclose(m[0], 24.0)

def test_discounting_reduces_value():
    S = np.ones((73, 1))
    a = expected_discounted_months(S, np.array([5]), 24, 0.0)
    b = expected_discounted_months(S, np.array([5]), 24, 0.10)
    assert b[0] < a[0]

def test_qini_rewards_good_ranking():
    rng = np.random.default_rng(0)
    n = 5000
    t = rng.integers(0, 2, n)
    u = rng.random(n)
    y = rng.binomial(1, 0.1 + 0.4 * t * (u > 0.5))     # only u>0.5 respond to treatment
    assert qini_coefficient(y, t, u) > 0 > qini_coefficient(y, t, -u)
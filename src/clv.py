import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

MARGIN, ANNUAL_DISCOUNT, HORIZON, MAX_T = 0.30, 0.10, 24, 72

def expected_discounted_months(S, t0, horizon=HORIZON, annual_discount=0.0, max_t=MAX_T):
    """S: (max_t+1, n) survival matrix, row k = P(survive past month k), row 0 = 1.
    t0: current tenure per customer (int array). Returns expected (discounted) months
    the customer stays over the next `horizon` months, given they are here now."""
    n = S.shape[1]
    t0 = np.minimum(np.asarray(t0).astype(int), max_t)
    ks = np.arange(1, horizon + 1)[:, None]
    idx = np.minimum(t0[None, :] + ks, max_t)
    s_future = np.take_along_axis(S, idx, axis=0)
    s_now = S[t0, np.arange(n)]
    cond = s_future / s_now
    r_m = (1 + annual_discount) ** (1 / 12) - 1
    return (cond * (1 + r_m) ** (-ks)).sum(axis=0)

if __name__ == "__main__":
    from src.db import load_features, get_engine
    from src.features import clean, onehot_frame, SURV_EXCLUDE
    from src.results import save_metrics

    df = clean(load_features())
    bundle = joblib.load("models/cox.joblib")
    cph, cols = bundle["model"], bundle["columns"]
    X = onehot_frame(df, exclude=SURV_EXCLUDE).reindex(columns=cols, fill_value=0.0)

    sf = cph.predict_survival_function(X, times=np.arange(1, MAX_T + 1))
    S = np.vstack([np.ones((1, len(df))), sf.values])        # prepend S(0)=1
    t0 = df["tenure"].astype(int).to_numpy()

    df["exp_months"] = expected_discounted_months(S, t0, HORIZON, 0.0)
    disc_months = expected_discounted_months(S, t0, HORIZON, ANNUAL_DISCOUNT)
    df["clv"] = df["monthlycharges"] * MARGIN * disc_months

    # merge earlier outputs
    df = df.merge(pd.read_csv("data/processed/churn_oof.csv"), on="customer_id")
    df = df.merge(pd.read_csv("data/processed/uplift_scores.csv"), on="customer_id")

    # ---- segmentation ----
    feats = ["clv", "tenure", "monthlycharges", "churn_prob", "num_addons"]
    Z = StandardScaler().fit_transform(df[feats])
    sil = {k: silhouette_score(Z, KMeans(k, n_init=10, random_state=42).fit_predict(Z))
           for k in range(3, 7)}
    best_k = max(sil, key=sil.get)
    df["segment_id"] = KMeans(best_k, n_init=10, random_state=42).fit_predict(Z)

    prof = df.groupby("segment_id")[feats + ["churn"]].mean()
    prof["customers"] = df.groupby("segment_id").size()
    print(prof.round(2))

    def name(sid):
        hi_val = prof.loc[sid, "clv"] >= df["clv"].mean()
        hi_risk = prof.loc[sid, "churn_prob"] >= df["churn_prob"].mean()
        return f"S{sid}: {'High' if hi_val else 'Lower'}-value / {'high' if hi_risk else 'low'}-risk"
    df["segment"] = df["segment_id"].map(name)

    out = df[["customer_id", "contract", "internetservice", "tenure", "monthlycharges", "churn",
              "churn_prob", "exp_months", "clv", "uplift", "true_uplift", "segment_id", "segment"]]
    out.to_csv("data/processed/customer_scores.csv", index=False)
    out.to_sql("customer_scores", get_engine(), if_exists="replace", index=False)
    prof.round(3).to_csv("docs/segment_profiles.csv")

    save_metrics("CLV and segmentation", {
        "Margin assumption": MARGIN, "Annual discount rate": ANNUAL_DISCOUNT,
        "Horizon (months)": HORIZON, "Mean CLV": df["clv"].mean(),
        "Median CLV": df["clv"].median(), "Chosen k": best_k, "Silhouette (chosen k)": sil[best_k]})
    print("Silhouette by k:", {k: round(v, 3) for k, v in sil.items()})
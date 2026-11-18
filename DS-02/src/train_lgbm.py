"""Step 5: global LightGBM point forecast + 10%/90% quantile models for prediction intervals."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from utils import PROC, OUT, HORIZON, evaluate, save_metrics

df = pd.read_parquet(PROC / "features.parquet")
FEATURES = [c for c in df.columns if c.startswith(("lag_", "roll_"))] + [
    "sell_price", "price_ratio", "promo", "snap", "is_event", "on_sale",
    "dow", "dom", "woy", "month", "item_code", "store_code", "state_code",
]
CATS = ["item_code", "store_code", "state_code", "dow"]

train = df[~df["is_test"]].dropna(subset=["lag_56"])  # drop the first rows that have no full history
test = df[df["is_test"]].copy()

# hold out the last 28 training days to find the right number of trees (early stopping)
cut = train["date"].max() - pd.Timedelta(days=HORIZON)
tr, va = train[train["date"] <= cut], train[train["date"] > cut]

base = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)

# --- point forecast: Tweedie loss suits non-negative count data with many zeros ---
point = dict(objective="tweedie", tweedie_variance_power=1.1, **base)
m = lgb.LGBMRegressor(n_estimators=2000, **point)
m.fit(tr[FEATURES], tr["sales"], eval_set=[(va[FEATURES], va["sales"])], categorical_feature=CATS,
      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
best = m.best_iteration_
print("best number of trees:", best)

# refit on ALL training data with that number of trees
final = lgb.LGBMRegressor(n_estimators=best, **point).fit(
    train[FEATURES], train["sales"], categorical_feature=CATS)
test["pred"] = final.predict(test[FEATURES]).clip(0)

# --- interval models ---
for name, alpha in (("q10", 0.1), ("q90", 0.9)):
    q = lgb.LGBMRegressor(n_estimators=best, objective="quantile", alpha=alpha, **base).fit(
        train[FEATURES], train["sales"], categorical_feature=CATS)
    test[name] = q.predict(test[FEATURES]).clip(0)
test["lo"] = np.minimum(test["q10"], test["pred"])  # make sure lo <= pred <= hi
test["hi"] = np.maximum(test["q90"], test["pred"])

coverage = ((test["sales"] >= test["lo"]) & (test["sales"] <= test["hi"])).mean()
print(f"80% interval coverage on test: {coverage:.1%} (ideal is about 80%)")

test[["id", "date", "pred", "lo", "hi"]].to_parquet(OUT / "pred_lgbm.parquet", index=False)
save_metrics(evaluate(test, train, "LightGBM"))
pd.Series(final.feature_importances_, index=FEATURES).sort_values(ascending=False).to_csv(OUT / "lgbm_importance.csv")
with open(OUT / "interval_coverage.txt", "w") as f:
    f.write(f"LightGBM 80% interval coverage: {coverage:.4f}\n")
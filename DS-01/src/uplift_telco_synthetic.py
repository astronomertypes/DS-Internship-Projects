import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from causalml.inference.meta import BaseTClassifier

from src.db import load_features
from src.features import clean, onehot_frame
from src.metrics import qini_coefficient
from src.results import save_metrics

# ASSUMPTIONS (SYNTHETIC) - reference offer: 20% discount on the monthly bill
df = clean(load_features())
df = df.merge(pd.read_csv("data/processed/churn_oof.csv"), on="customer_id", how="left")
rng = np.random.default_rng(42)

t = rng.binomial(1, 0.5, len(df))                       # random assignment
mtm = df["contract"] == "Month-to-month"
price_sensitive = mtm & (df["monthlycharges"] > df["monthlycharges"].median())
# relative reduction in churn probability if offered
rel_red = np.where(price_sensitive, 0.45, np.where(mtm, 0.25, 0.05))

p0 = df["churn_prob"].to_numpy()
true_uplift = p0 * rel_red                              # true increase in P(customer stays)
churned = rng.binomial(1, p0 * (1 - t * rel_red))       # simulated outcome
retained = 1 - churned

X = onehot_frame(df)
lab = lambda a: np.where(a == 1, "treated", "control")
LGB = dict(n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=50,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)

# Cross-fitting: every customer's uplift comes from a model that never saw them
oof_uplift = np.zeros(len(df))
for tr, te in KFold(5, shuffle=True, random_state=42).split(X):
    m = BaseTClassifier(learner=LGBMClassifier(**LGB), control_name="control")
    m.fit(X=X.iloc[tr], treatment=lab(t[tr]), y=retained[tr])
    oof_uplift[te] = m.predict(X=X.iloc[te]).ravel()

spearman = pd.Series(oof_uplift).corr(pd.Series(true_uplift), method="spearman")
q_model = qini_coefficient(retained, t, oof_uplift)
q_risk = qini_coefficient(retained, t, p0)              # baseline: target by churn risk
q_rand = qini_coefficient(retained, t, rng.random(len(df)))
print(f"Spearman(est, true uplift)={spearman:.3f} | Qini: uplift model {q_model:.3f}, "
      f"churn-risk targeting {q_risk:.3f}, random {q_rand:.3f}")

pd.DataFrame({"customer_id": df["customer_id"], "uplift": oof_uplift,
              "true_uplift": true_uplift}).to_csv("data/processed/uplift_scores.csv", index=False)
save_metrics("Uplift - Telco (SYNTHETIC treatment)", {
    "Spearman (estimated vs true uplift)": spearman,
    "Qini coefficient - uplift model": q_model,
    "Qini coefficient - churn-risk targeting": q_risk,
    "Qini coefficient - random": q_rand,
    "Reference discount (%)": 20})
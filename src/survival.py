import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.model_selection import train_test_split

from src.db import load_features
from src.features import clean, onehot_frame, SURV_EXCLUDE
from src.results import save_metrics

df = clean(load_features())
fit_df = df[df["tenure"] > 0].copy()          # drop the 11 customers with tenure 0

# ---- Kaplan-Meier ----
kmf = KaplanMeierFitter()
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col in zip(axes, ["contract", "internetservice"]):
    for name, g in fit_df.groupby(col):
        kmf.fit(g["tenure"], g["churn"], label=name)
        kmf.plot_survival_function(ax=ax, ci_show=True)
    ax.set_title(f"Survival by {col}"); ax.set_xlabel("Months"); ax.set_ylabel("P(still a customer)")
plt.tight_layout(); plt.savefig("docs/km_curves.png", dpi=150); plt.close()

kmf.fit(fit_df["tenure"], fit_df["churn"])
median = kmf.median_survival_time_
logrank_p = multivariate_logrank_test(fit_df["tenure"], fit_df["contract"], fit_df["churn"]).p_value

# ---- Cox model ----
X = onehot_frame(fit_df, exclude=SURV_EXCLUDE)
cox_df = X.copy()
cox_df["tenure"] = fit_df["tenure"].values
cox_df["churn"] = fit_df["churn"].values

# honest evaluation: fit on 80%, score concordance on the unseen 20%
tr, te = train_test_split(cox_df, test_size=0.2, random_state=42, stratify=cox_df["churn"])
cph_h = CoxPHFitter(penalizer=0.01).fit(tr, duration_col="tenure", event_col="churn")
c_test = cph_h.score(te, scoring_method="concordance_index")

# final model on all data (used to score every customer)
cph = CoxPHFitter(penalizer=0.01).fit(cox_df, duration_col="tenure", event_col="churn")
cph.print_summary(decimals=3)
print("\nProportional-hazards assumption check:")
cph.check_assumptions(cox_df, p_value_threshold=0.01, show_plots=False)

cph.summary[["exp(coef)", "p"]].sort_values("exp(coef)").to_csv("docs/cox_hazard_ratios.csv")
joblib.dump({"model": cph, "columns": list(X.columns)}, "models/cox.joblib")

print(f"\nConcordance (train) {cph.concordance_index_:.3f} | (held-out) {c_test:.3f}")
save_metrics("Survival (Lifelines)", {
    "KM median survival (months; None = more than half never churned)":
        None if np.isinf(median) else float(median),
    "Log-rank p-value (contract)": logrank_p,
    "Cox concordance (train)": cph.concordance_index_,
    "Cox concordance (held-out)": c_test, "Customers used": len(fit_df)})
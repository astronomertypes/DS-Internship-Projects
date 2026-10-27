import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     cross_val_predict, train_test_split)

from src.db import load_features
from src.features import clean, gbm_frame
from src.results import save_metrics

PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=42, verbose=-1)

df = clean(load_features())
X, y = gbm_frame(df), df["churn"]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1) 5-fold cross-validated performance (the number to quote)
cvres = cross_validate(LGBMClassifier(**PARAMS), X, y, cv=cv,
                       scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"})
cv_auc, cv_pr = cvres["test_roc_auc"], cvres["test_pr_auc"]
print(f"CV ROC-AUC {cv_auc.mean():.3f} +/- {cv_auc.std():.3f} | "
      f"CV PR-AUC {cv_pr.mean():.3f} +/- {cv_pr.std():.3f}")

# 2) Hold-out test set (used for SHAP plots)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = LGBMClassifier(**PARAMS).fit(X_tr, y_tr)
p_te = model.predict_proba(X_te)[:, 1]
holdout = {"roc_auc": roc_auc_score(y_te, p_te),
           "pr_auc": average_precision_score(y_te, p_te),
           "brier": brier_score_loss(y_te, p_te)}
print("Hold-out:", {k: round(v, 3) for k, v in holdout.items()})

# 3) SHAP explanations
sv = shap.TreeExplainer(model).shap_values(X_te)
if isinstance(sv, list):                 # older shap versions return [class0, class1]
    sv = sv[1]
elif getattr(sv, "ndim", 2) == 3:        # some versions return (n, features, classes)
    sv = sv[:, :, 1]
shap.summary_plot(sv, X_te, show=False)
plt.tight_layout(); plt.savefig("docs/shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
shap.summary_plot(sv, X_te, plot_type="bar", show=False)
plt.tight_layout(); plt.savefig("docs/shap_importance.png", dpi=150, bbox_inches="tight"); plt.close()

# 4) Out-of-fold churn probability for EVERY customer (used by CLV, uplift, app).
#    Each customer is scored by a model that never saw them, so scores are honest.
oof = cross_val_predict(LGBMClassifier(**PARAMS), X, y, cv=cv, method="predict_proba")[:, 1]
pd.DataFrame({"customer_id": df["customer_id"], "churn_prob": oof}) \
    .to_csv("data/processed/churn_oof.csv", index=False)

joblib.dump(model, "models/churn_lgbm.joblib")
save_metrics("Churn model (LightGBM)", {
    "CV ROC-AUC mean": cv_auc.mean(), "CV ROC-AUC std": cv_auc.std(),
    "CV PR-AUC mean": cv_pr.mean(), "CV PR-AUC std": cv_pr.std(),
    "Hold-out ROC-AUC": holdout["roc_auc"], "Hold-out PR-AUC": holdout["pr_auc"],
    "Hold-out Brier score": holdout["brier"], "Positive rate": float(y.mean())})
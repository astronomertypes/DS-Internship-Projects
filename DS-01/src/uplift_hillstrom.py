import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from causalml.inference.meta import BaseTClassifier, BaseSClassifier

from src.metrics import qini_curve, qini_coefficient
from src.results import save_metrics

df = pd.read_csv("data/raw/hillstrom.csv")
print(df.columns.tolist())   # expect recency, history, mens, womens, zip_code, newbie, channel, segment, visit ...

df["treatment"] = (df["segment"] != "No E-Mail").astype(int)   # either email vs none
y, t = df["visit"].to_numpy(), df["treatment"].to_numpy()

# Randomization check: covariate means should be nearly identical across groups
print(df.groupby("treatment")[["recency", "history", "mens", "womens", "newbie"]].mean().round(3))
rates = df.groupby("treatment")["visit"].mean()
ate = rates[1] - rates[0]
print(f"Visit rate control {rates[0]:.3f}, treated {rates[1]:.3f}, average uplift {ate:.3f}")

X = pd.get_dummies(df[["recency", "history", "mens", "womens", "newbie", "zip_code", "channel"]],
                   drop_first=True, dtype=float)
strat = df["treatment"].astype(str) + df["visit"].astype(str)
X_tr, X_te, y_tr, y_te, t_tr, t_te = train_test_split(
    X, y, t, test_size=0.3, random_state=42, stratify=strat)

LGB = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=100,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
lab = lambda a: np.where(a == 1, "treated", "control")

models = {"T-learner": BaseTClassifier(learner=LGBMClassifier(**LGB), control_name="control"),
          "S-learner": BaseSClassifier(learner=LGBMClassifier(**LGB), control_name="control")}
scores = {}
for name, m in models.items():
    m.fit(X=X_tr, treatment=lab(t_tr), y=y_tr)
    scores[name] = m.predict(X=X_te).ravel()
scores["Random"] = np.random.default_rng(42).random(len(X_te))

# Qini curves
plt.style.use("default")   # undo the style that causalml sets on import
x = np.arange(1, len(y_te) + 1) / len(y_te)
colors = {"T-learner": "#1f77b4", "S-learner": "#d62728", "Random": "#7f7f7f"}
fig, ax = plt.subplots(figsize=(9, 6))
for name, s in scores.items():
    q = qini_curve(y_te, t_te, s)
    ax.plot(x, q, label=("Random scores (one draw)" if name == "Random" else name),
            color=colors[name], linewidth=2.2 if name != "Random" else 1.5,
            alpha=0.6 if name == "Random" else 1.0)
total_gain = qini_curve(y_te, t_te, scores["T-learner"])[-1]
ax.plot([0, 1], [0, total_gain], linestyle="--", color="black", linewidth=1.3,
        label="Expected random targeting")
ax.set_xlabel("Fraction of customers targeted (best-scored first)", fontsize=12)
ax.set_ylabel("Cumulative incremental website visits", fontsize=12)
ax.set_title("Qini curves on the Hillstrom randomized experiment (test set)", fontsize=13)
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=11)
fig.tight_layout()
fig.savefig("docs/qini_hillstrom.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# Decile table for the T-learner: observed uplift should fall from decile 1 to 10
d = pd.DataFrame({"y": y_te, "t": t_te, "s": scores["T-learner"]})
d["decile"] = pd.qcut(d["s"].rank(method="first", ascending=False), 10, labels=range(1, 11))
tab = d.groupby(["decile", "t"], observed=True)["y"].mean().unstack("t")
tab["observed_uplift"] = tab[1] - tab[0]
print(tab.round(4)); tab.to_csv("docs/uplift_deciles_hillstrom.csv")

res = {f"Qini coefficient - {k}": qini_coefficient(y_te, t_te, v) for k, v in scores.items()}
res["Average uplift (visit rate)"] = ate
save_metrics("Uplift - Hillstrom (real randomized experiment)", res)
print(res)
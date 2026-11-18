import pandas as pd

RECODE_COLS = ["multiplelines", "onlinesecurity", "onlinebackup", "deviceprotection",
               "techsupport", "streamingtv", "streamingmovies"]

NUM_FEATURES = ["seniorcitizen", "tenure", "monthlycharges", "num_addons",
                "no_tech_support", "charge_trend"]
CAT_FEATURES = ["gender", "partner", "dependents", "phoneservice", "multiplelines",
                "internetservice", "onlinesecurity", "onlinebackup", "deviceprotection",
                "techsupport", "streamingtv", "streamingmovies", "contract",
                "paperlessbilling", "paymentmethod"]

# Must NOT be Cox covariates: tenure is the duration itself; charge_trend is built from
# totalcharges (~ tenure x price, so it leaks duration); num_addons and no_tech_support
# are exact combinations of other columns (singular design matrix).
SURV_EXCLUDE = ["tenure", "charge_trend", "num_addons", "no_tech_support"]

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Merge 'No internet service' / 'No phone service' into 'No'."""
    df = df.copy()
    for c in RECODE_COLS:
        df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})
    return df

def gbm_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Features for LightGBM (categoricals as pandas 'category')."""
    X = df[NUM_FEATURES + CAT_FEATURES].copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype("category")
    return X

def onehot_frame(df: pd.DataFrame, exclude=()) -> pd.DataFrame:
    """One-hot features (drop_first avoids collinearity) for Cox / uplift."""
    num = [c for c in NUM_FEATURES if c not in exclude]
    X = pd.get_dummies(df[num + CAT_FEATURES], columns=CAT_FEATURES,
                       drop_first=True, dtype=float)
    return X.astype(float)
# Data dictionary

**Author:** Sayantani Manna

This file documents every table and file my project creates: what each column means, its type, and where it comes from.

**Source column key**
- **Raw:** copied from the original Kaggle CSV (only lowercased or renamed).
- **Derived (SQL):** computed in `sql/features.sql`.
- **Model output:** produced by one of my Python scripts.

---

## 1. Source datasets

| Dataset | File | Rows | Used for |
|---|---|---|---|
| IBM Telco Customer Churn (Kaggle) | `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` | 7,043 | Main dataset for every step |
| Hillstrom MineThatData E-Mail Analytics | `data/raw/hillstrom.csv` | about 64,000 | Uplift validation on a real randomized experiment |

Neither raw file is committed to Git (`data/raw/` is in `.gitignore`). The README explains where to download them.

---

## 2. PostgreSQL table `customers_raw`

Created by `src/load_data.py`. One row per customer. Column names are lowercased, and `customerID` becomes `customer_id`.

| Column | Type | Meaning | Values | Source |
|---|---|---|---|---|
| `customer_id` | text | Unique customer identifier | e.g. `7590-VHVEG` | Raw |
| `gender` | text | Customer gender | Female, Male | Raw |
| `seniorcitizen` | integer | Whether the customer is a senior citizen | 0 = no, 1 = yes | Raw |
| `partner` | text | Has a partner | Yes, No | Raw |
| `dependents` | text | Has dependents | Yes, No | Raw |
| `tenure` | integer | Months the customer has been with the company | 0 to 72 | Raw |
| `phoneservice` | text | Has phone service | Yes, No | Raw |
| `multiplelines` | text | Has multiple phone lines | Yes, No, No phone service | Raw |
| `internetservice` | text | Type of internet service | DSL, Fiber optic, No | Raw |
| `onlinesecurity` | text | Online security add-on | Yes, No, No internet service | Raw |
| `onlinebackup` | text | Online backup add-on | Yes, No, No internet service | Raw |
| `deviceprotection` | text | Device protection add-on | Yes, No, No internet service | Raw |
| `techsupport` | text | Tech support add-on | Yes, No, No internet service | Raw |
| `streamingtv` | text | Streams TV through the provider | Yes, No, No internet service | Raw |
| `streamingmovies` | text | Streams movies through the provider | Yes, No, No internet service | Raw |
| `contract` | text | Contract type | Month-to-month, One year, Two year | Raw |
| `paperlessbilling` | text | Uses paperless billing | Yes, No | Raw |
| `paymentmethod` | text | How the customer pays | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) | Raw |
| `monthlycharges` | double | Current monthly bill in dollars | about 18 to 119 | Raw |
| `totalcharges` | double | Total billed over the customer's lifetime | 0 or more. The 11 blank values (all customers with tenure 0) were set to 0 | Raw, cleaned |
| `churn` | integer | **Target.** Whether the customer left | 0 = stayed, 1 = left (rate 0.265) | Raw, converted from Yes/No |

---

## 3. PostgreSQL view `customers_features`

Created from `customers_raw` by `sql/features.sql`. It contains **all 21 columns from `customers_raw`**, plus five columns I engineered.

| Column | Type | Meaning and formula | Why I created it | Source |
|---|---|---|---|---|
| `num_addons` | integer | Count of "Yes" among onlinesecurity, onlinebackup, deviceprotection, techsupport, streamingtv, streamingmovies (0 to 6) | Proxy for how engaged a customer is | Derived (SQL) |
| `no_tech_support` | integer | 1 if the customer has internet service but no tech support, else 0 | **Proxy for complaints.** Telco has no complaint data, and customers without a support channel are more likely to be frustrated | Derived (SQL) |
| `avg_monthly_spend` | double | `totalcharges / tenure` (the monthly charge if tenure is 0) | Historical average bill | Derived (SQL) |
| `charge_trend` | double | `monthlycharges - totalcharges / tenure` (0 if tenure is 0) | **Proxy for a usage or bill trend.** Positive means the bill has grown relative to the past average | Derived (SQL) |
| `tenure_bucket` | text | Tenure group: 0-12, 13-24, 25-48, 49+ | Used in the EDA charts | Derived (SQL) |

I left `totalcharges` and `avg_monthly_spend` out of the model inputs because they are roughly tenure multiplied by price, so they are redundant and can leak duration.

### Features used by the models

Defined in `src/features.py`.

| Group | Columns |
|---|---|
| Numeric | `seniorcitizen`, `tenure`, `monthlycharges`, `num_addons`, `no_tech_support`, `charge_trend` |
| Categorical | `gender`, `partner`, `dependents`, `phoneservice`, `multiplelines`, `internetservice`, `onlinesecurity`, `onlinebackup`, `deviceprotection`, `techsupport`, `streamingtv`, `streamingmovies`, `contract`, `paperlessbilling`, `paymentmethod` |
| Cox model exclusions | `tenure` (it is the duration), `charge_trend`, `num_addons`, `no_tech_support` |

The `clean()` function merges "No internet service" and "No phone service" into "No" for the seven add-on columns.

---

## 4. Processed files in `data/processed/`

### `churn_oof.csv` (written by `src/churn_model.py`)

| Column | Type | Meaning |
|---|---|---|
| `customer_id` | text | Customer identifier |
| `churn_prob` | float, 0 to 1 | Probability of churning, from 5-fold **out-of-fold** LightGBM predictions. Each customer was scored by a model that never trained on them |

### `uplift_scores.csv` (written by `src/uplift_telco_synthetic.py`)

| Column | Type | Meaning |
|---|---|---|
| `customer_id` | text | Customer identifier |
| `uplift` | float | **Estimated** increase in the probability of staying if the customer gets the 20% reference discount (T-learner, cross-fitted) |
| `true_uplift` | float | The **simulated ground-truth** effect (`churn_prob` times a reduction factor). It exists only because the treatment is synthetic, and I used it only to check the model, never as a model input |

### `customer_scores.csv` (written by `src/clv.py`, also loaded into table `customer_scores`)

This is the final per-customer table that the Streamlit app reads. It has 7,043 rows.

| Column | Type | Meaning | Source |
|---|---|---|---|
| `customer_id` | text | Customer identifier | Raw |
| `contract` | text | Contract type | Raw |
| `internetservice` | text | Internet service type | Raw |
| `tenure` | integer | Months with the company | Raw |
| `monthlycharges` | double | Current monthly bill ($) | Raw |
| `churn` | integer | Actual churn (0/1) | Raw |
| `churn_prob` | float | Out-of-fold churn probability | Model output (LightGBM) |
| `exp_months` | float | Expected months the customer stays during the next 24 months, undiscounted, from the Cox survival curve given that they are a customer now | Model output (Cox) |
| `clv` | float | Customer lifetime value in $: `monthlycharges × margin × sum over the next 24 months of [P(still a customer) × discount factor]`. Margin 30%, discount rate 10% per year | Model output |
| `uplift` | float | Estimated uplift (synthetic treatment) | Model output (CausalML) |
| `true_uplift` | float | Simulated true uplift (for validation only) | Simulation |
| `segment_id` | integer | k-means cluster number (0, 1, 2) | Model output (k-means) |
| `segment` | text | Readable segment label | Model output |

### Segment labels

| Label in the data | Plain-language name | Customers | Mean CLV | Mean tenure | Mean churn probability |
|---|---|---|---|---|---|
| `S0: Lower-value / low-risk` | Small and steady | 2,360 | $166 | 30 months | 0.11 |
| `S1: High-value / low-risk` | Loyal big spenders | 2,492 | $508 | 53 months | 0.14 |
| `S2: High-value / high-risk` | New and at risk | 2,191 | $352 | 12 months | 0.57 |

My code labels a segment "High-value" when its mean CLV is at or above the overall mean (about $345). S2 sits only just above that line, so I describe it as mid-to-high value.

---

## 5. Saved models in `models/`

| File | Contents |
|---|---|
| `churn_lgbm.joblib` | LightGBM churn classifier trained on 80% of customers (used for the SHAP plots) |
| `cox.joblib` | Dictionary with the fitted Cox model (`model`) and its column list (`columns`) |

---

## 6. Generated reports in `docs/`

| File | Created by | Contents |
|---|---|---|
| `results.json` | `src/results.py` (called by each script) | All metrics, machine-readable |
| `results.md` | `src/make_results_md.py` | The same metrics as readable tables |
| `test_results.txt` | `pytest -v` | Unit test output (4 tests) |
| `eda_overview.png` | `notebooks/01_eda.ipynb` | Six EDA charts |
| `shap_importance.png` | `src/churn_model.py` | Mean absolute SHAP value per feature |
| `shap_summary.png` | `src/churn_model.py` | SHAP dot plot (direction of each feature's effect) |
| `km_curves.png` | `src/survival.py` | Kaplan-Meier curves by contract and by internet service |
| `cox_hazard_ratios.csv` | `src/survival.py` | Hazard ratio and p-value per feature |
| `qini_hillstrom.png` | `src/uplift_hillstrom.py` | Qini curves on the real Hillstrom experiment |
| `uplift_deciles_hillstrom.csv` | `src/uplift_hillstrom.py` | Observed uplift per predicted-uplift decile |
| `segment_profiles.csv` | `src/clv.py` | Average values per customer segment |
| `architecture.md`, `architecture.mmd`, `architecture.png` | written by me | System diagram and explanation |
| `data_dictionary.md` | written by me | This file |

---

## 7. Hillstrom dataset columns (used only in `src/uplift_hillstrom.py`)

| Column | Meaning |
|---|---|
| `recency` | Months since last purchase |
| `history_segment` | Bucketed past spending (not used; I used `history` instead) |
| `history` | Dollars spent in the past year |
| `mens`, `womens` | 1 if the customer bought men's or women's merchandise in the past year |
| `zip_code` | Urban, Suburban, or Rural |
| `newbie` | 1 if the customer is new in the past year |
| `channel` | Channel used for past purchases (Phone, Web, Multichannel) |
| `segment` | Experiment arm: Mens E-Mail, Womens E-Mail, or No E-Mail |
| `visit` | 1 if the customer visited the website in the following two weeks (the outcome) |
| `conversion`, `spend` | Purchase outcomes (not used) |

In my script, `treatment = 1` for either email arm and `0` for No E-Mail.

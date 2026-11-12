# Results

All results in this document come from the runs on my machine (Windows, PowerShell, CPU only). The short version is in the [README](../README.md), the design is in [ARCHITECTURE.md](ARCHITECTURE.md) and the files are described in [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## Contents

1. [How the results were measured](#1-how-the-results-were-measured)
2. [Forecast accuracy](#2-forecast-accuracy)
3. [The TFT epoch experiment](#3-the-tft-epoch-experiment)
4. [Prediction intervals](#4-prediction-intervals)
5. [Hierarchical reconciliation](#5-hierarchical-reconciliation)
6. [Inventory optimization](#6-inventory-optimization)
7. [How the optimizer changed during the project](#7-how-the-optimizer-changed-during-the-project)
8. [Diagnostics](#8-diagnostics)
9. [Sanity checks on saved files](#9-sanity-checks-on-saved-files)
10. [Reproducibility](#10-reproducibility)
11. [What these results do not show](#11-what-these-results-do-not-show)

---

## 1. How the results were measured

- **Data:** 500 series (50 FOODS_3 items × 10 stores), the last 900 days of the M5 data.
- **Test period:** the last 28 days (25 April to 22 May 2016). No model saw them in training. All lag features are shifted by 28 days or more, so there is no leakage.
- **Metrics (lower is better):**
  - **WAPE item-store:** total absolute error ÷ total actual sales over all 500 series and 28 days.
  - **WAPE store:** the same after adding item forecasts and actuals up to store × day.
  - **RMSSE:** the mean over the 500 series of the forecast RMSE ÷ the RMSE of a one-step naive forecast on the training data. This is a simple mean, not the official M5 WRMSSE, so it is not comparable with the Kaggle leaderboard.
- **Inventory results:** a 14-day order plan (the first 14 test days), scored against what really sold.

---

## 2. Forecast accuracy

| Model | WAPE item-store | WAPE store | RMSSE |
|---|---|---|---|
| Seasonal naive | 0.4666 | 0.1163 | 0.9273 |
| Prophet | 0.4626 | 0.1276 | 0.8529 |
| LightGBM | 0.4097 | **0.1004** | 0.7941 |
| TFT | **0.3919** | 0.1293 | **0.7853** |

Change against the seasonal naive baseline:

| Model | WAPE item-store | WAPE store |
|---|---|---|
| Prophet | 0.8% lower | 9.7% higher |
| LightGBM | 12.2% lower | 13.7% lower |
| TFT | 16.0% lower | 11.2% higher |

What I take from this:

- **The TFT is the best model for individual item-store series** (lowest item-level WAPE and RMSSE).
- **LightGBM is the best once forecasts are added up to the store level**, and it is the only model that beats the naive baseline at both levels.
- **Prophet barely beats the naive baseline at item level** and is worse at store level. It uses no price information, which is a likely reason (not tested).
- The TFT's store-level error is higher than LightGBM's. One possible reason is that its point forecast is the median, which tends to sit below the mean for skewed count data, so store totals come out slightly low. I have not tested this. [IDEA: check whether the TFT's store totals are systematically below the actual totals.]
- All numbers come from one 28-day window, and I did not test whether the gap between the TFT and LightGBM (0.3919 versus 0.4097) is statistically meaningful.

### LightGBM training facts

- 22 features, about 408,000 training rows.
- Early stopping on the last 28 training days chose **147 trees** (validation Tweedie loss 112.757). The final model was refit on all training rows with that number of trees.
- Parameters: learning rate 0.05, 63 leaves, minimum 50 samples per leaf, 80% row and column sampling, Tweedie variance power 1.1, random state 42.

### TFT training facts

- 25,858 trainable parameters (hidden size 16, 2 attention heads, dropout 0.1, hidden continuous size 8).
- 56-day look-back, 28-day horizon, quantile loss at 10%, 50% and 90%.
- Batch size 128, 400 batches per epoch (51,200 windows), at most 10 epochs, early stopping after 3 epochs without improvement.
- About 76 seconds per epoch on the CPU, so about 13 minutes for 10 epochs. The 10-epoch run ended at the epoch limit with a validation loss of 3.640.

---

## 3. The TFT epoch experiment

I tested whether more training helps by raising the epoch limit from 10 to 20.

| Setting | Epochs actually run | WAPE item-store | WAPE store | RMSSE |
|---|---|---|---|---|
| `max_epochs = 10` | 10 (stopped at the limit) | **0.3919** | **0.1293** | **0.7853** |
| `max_epochs = 20` | 15 (stopped early) | 0.3941 | 0.1446 | 0.7948 |

More epochs did not help. Early stopping ended the longer run after 15 epochs, and its result was slightly worse, especially at store level. I kept the 10-epoch model. I then reran the 10-epoch setting and got exactly the same metrics, because the seed is fixed (42).

---

## 4. Prediction intervals

| Item | Value |
|---|---|
| LightGBM interval | 10% and 90% quantile models, so an 80% interval |
| Share of real test sales inside the interval | **82.3%** (target 80%) |

The coverage is close to the target, so the interval is reasonably calibrated at the daily item-store level. The TFT also outputs 10% and 90% quantiles (`lo` and `hi` in `pred_tft.parquet`), but I did not evaluate their coverage. [IDEA: compute the TFT's interval coverage and add a row to this table.]

---

## 5. Hierarchical reconciliation

The hierarchy has 514 nodes (1 total, 3 states, 10 stores, 500 item-store series). The bottom level comes from LightGBM and the 14 upper nodes come from separate Prophet models, so the base forecasts disagree. Numbers are WAPE per level (lower is better). The coherence gap is the largest difference, in units per day, between a forecast and the sum of its parts, taken over all three links of the hierarchy (items against their store, stores against their state, and states against the total).

| Forecast | Total | State | Store | Item-store | Coherence gap |
|---|---|---|---|---|---|
| Base (unreconciled) | 0.0817 | 0.1063 | 0.1309 | 0.4097 | 241.04 |
| Bottom-up | **0.0584** | **0.0767** | **0.1004** | **0.4097** | 0 |
| MinT (`wls_struct`) | 0.0748 | 0.0953 | 0.1167 | 0.4166 | 0 |

The coherence gap for each link, in units per day (largest over the 28 days):

| Forecast | Items → store | Stores → state | States → total |
|---|---|---|---|
| Base (unreconciled) | 241.043 | 25.289 | 15.795 |
| Bottom-up | 0 | 0 | 0 |
| MinT (`wls_struct`) | 0 | 0 | 0 |

- Both reconciliation methods remove the disagreement (gap 0 on every link) and improve the upper levels compared with the Prophet base forecasts.
- **Bottom-up is the best at every level**, so my selection rule (best item-level WAPE among coherent methods) picked it.
- MinT blends in the weaker Prophet forecasts, so its item-level error is slightly higher (0.4166 versus 0.4097).
- With bottom-up, the reconciled item-store forecast is identical to LightGBM's. The benefit of reconciliation in this project is coherent and more accurate upper levels.
- The "store" column of the base row (0.1309) is a direct Prophet forecast of each store's total. This is different from the `WAPE_store` of Prophet in the accuracy table (0.1276), which adds up Prophet's item-level forecasts.

---

## 6. Inventory optimization

Policies are scored against the real sales of the 14-day plan window. Costs use the assumptions listed in the README (unit cost 70% of price, lost margin per missed unit, holding cost 5% of unit cost, 2 days of stock on hand).

| Policy | Holding cost | Stockout cost | Total cost | Fill rate | Order spend |
|---|---|---|---|---|---|
| Naive (order forecast mean, no budget) | 656 | 6,697 | 7,353 | 87.8% | 102,135 |
| Naive scaled to the same spend | 1,572 | 3,051 | 4,623 | 94.6% | 128,966 |
| **Optimized (OR-Tools)** | **1,527** | **2,670** | **4,197** | **95.7%** | 128,956 |

**The fair comparison is the last two rows**, because they spend the same money. At equal spend, the optimizer has:

- **9.2% lower total cost** (4,197 versus 4,623),
- **12.5% lower stockout cost** (2,670 versus 3,051),
- a fill rate **1.1 percentage points higher** (95.7% versus 94.6%).

The first row is a weaker baseline because it has no budget and spends about 21% less than the other two. It is shown to give a sense of scale, not as the comparison.

### Store budgets in the final run

| Store | Budget |
|---|---|
| CA_1 | 14,320.27 |
| CA_2 | 11,221.06 |
| CA_3 | 21,417.80 |
| CA_4 | 6,489.46 |
| TX_1 | 11,658.53 |
| TX_2 | 15,446.45 |
| TX_3 | 13,468.31 |
| WI_1 | 7,571.36 |
| WI_2 | 12,796.10 |
| WI_3 | 14,567.33 |
| **Total** | **128,956.67** |

The optimizer spent 128,955.51 in total, so the budget was used almost fully in every store.

---

## 7. How the optimizer changed during the project

The optimizer first looked good, then lost to a fair baseline, and I fixed it step by step. This table is the history.

| Version | Optimized total cost | Naive at the same spend | What happened |
|---|---|---|---|
| 1. Three scenarios built from summed daily quantiles | 6,285 (spend 153,625) | not computed (the plain naive policy had cost 7,353 and spend 102,135) | Looked 14.5% cheaper, but it simply ordered about 50% more than the naive policy. The "high" scenario was far too wide (for one item, 199 units against an expected 77) |
| 2. Three scenarios with a correlation assumption (`RHO = 0.2`), and the equal-spend baseline added | 5,832 | 4,477 | The optimizer lost to the equal-spend baseline |
| 3. Same, with scenarios widened by `SPREAD = 1.2` | 6,105 | 4,423 | Calibration was correct, but the gap did not close. It got slightly worse |
| 4. **Nine scenarios** | **4,197** | **4,623** | The optimizer won: 9.2% lower total cost at equal spend |

What I learned from this history:

- A result can look good only because it spends more. Comparing at equal spend is essential.
- I checked hypotheses with small diagnostics (next section) before changing code, and the first two guesses (forecast bias, scenarios too narrow) were not the main cause.

---

## 8. Diagnostics

### 8.1 Forecast bias check

Is the forecast too low or too high overall in the 14-day plan window?

| Total actual sales | Total forecast | Forecast ÷ actual |
|---|---|---|
| 93,282 | 93,263 | 0.9998 |

The forecast is essentially unbiased, so bias was not the reason the optimizer lost in version 2.

### 8.2 Scenario calibration (`src/check_scenarios.py`)

The script standardizes each series' real 14-day demand using the scenario model's mean and standard deviation. If the scenarios are the right width, the standard deviation of these standardized errors (z) is about 1.0, and the share of series outside the scenario range matches the expected tails.

| Version | Std of z | Actual above the top scenario | Actual below the bottom scenario | Expected for each tail |
|---|---|---|---|---|
| Three scenarios, `RHO = 0.2`, no `SPREAD` | 1.2 | 6.6% | 6.0% | about 4% |
| Three scenarios, `SPREAD = 1.2` | 1.0 | 3.8% | 3.0% | about 4% |
| Final version (P10 to P90 range, `SPREAD = 1.2`) | 1.0 | 7.6% | 10.0% | about 10% |

The first row showed that the scenarios were about 20% too narrow, which `SPREAD` fixed. The final version uses the P10 and P90 as its range, so each tail should be about 10%, and it is.

`SPREAD` was chosen on the same window that is used for scoring, which flatters the optimization result slightly.

### 8.3 Scenario clustering check

With three scenarios the solver had only three "stops" for each item. I measured where each item's available stock (`on_hand + order_qty`) ended up.

| Version | Items at the high scenario | Items at the mid scenario | Items in between |
|---|---|---|---|
| Three scenarios | 42% | 31% | 26% |
| Nine scenarios | 0% | 0% | 100% |

With three scenarios 73% of items sat exactly on a scenario value. That is the pattern I expected if stock is topped up in large steps, which spreads a limited budget badly. With nine scenarios the stock is spread smoothly. This check is not perfect proof, because items with very low volume fall within one unit of a scenario by chance, but together with the improvement in cost it supports the explanation.

---

## 9. Sanity checks on saved files

| Check | Result |
|---|---|
| `pred_prophet.parquet` has 14,000 rows, 500 series, no missing values, no negative forecasts | Passed (shape `(14000, 3)`, 500 ids, 0 missing, minimum 0.0) |
| `metrics.csv` contains the model rows and matches the printed tables | Passed |
| `order_recommendations.csv` has 500 rows, no missing values and no negative orders | Passed (shape `(500, 7)`) |
| Shape and column names of every processed, output and dashboard file match the data dictionary | Passed (commands in `DATA_DICTIONARY.md`, section 7). Prediction files have 14,000 rows: `(14000, 5)` for LightGBM and TFT, `(14000, 3)` for naive, Prophet and the reconciled forecast |
| `pred_lgbm.parquet`, `pred_tft.parquet` and `pred_lgbm_reconciled.parquet` have no missing values and 500 series, LightGBM has `lo ≤ pred ≤ hi`, and the reconciled forecast is never negative | Passed |
| Unit tests | 11 tests in `tests/test_core.py` passed. The full `pytest -q` result is in the README, section 9 |


---

## 10. Reproducibility

- Random seeds are fixed to 42 for LightGBM and for the TFT (PyTorch Lightning `seed_everything`).
- Rerunning `train_tft.py` with the 10-epoch setting reproduced the same metrics exactly.
- The versions of the libraries are in `requirements.txt` (made with `pip freeze`). Darts was 0.48.1. The main versions are listed in the README, section 6.
- Runtime on my CPU-only machine: the TFT takes about 13 minutes. The run times of the other scripts are in the README, section 6.3.

---

## 11. What these results do not show

- **They come from one 28-day test window** and one 14-day order window. A different window could give a different ranking or a different cost difference.
- **The 9.2% cost difference is encouraging, not proof.** `SPREAD` was tuned on the scoring window, and `RHO`, the cost parameters and the stock on hand are assumptions, not data.
- **Only 500 series** from one department were used.
- **The RMSSE is not the official M5 metric**, so these numbers cannot be compared with the competition leaderboard.
- **Order spend is not part of the cost totals**, so policies are only comparable at equal spend.

[IDEA: the strongest improvement to this document would be a rolling-origin backtest over several 28-day windows, with `SPREAD` and `RHO` tuned on earlier windows and scored on later ones. Add it to Next steps if you do not have time to run it.]

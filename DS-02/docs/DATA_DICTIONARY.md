# Data dictionary

This document describes every data file in my project: what it is, which script creates it, and what each column means. The overview is in the [README](../README.md) (section 3.1) and the design is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Contents

1. [Conventions](#1-conventions)
2. [Raw data (`data/raw/`)](#2-raw-data-dataraw)
3. [Processed data (`data/processed/`)](#3-processed-data-dataprocessed)
4. [Results (`outputs/`)](#4-results-outputs)
5. [Dashboard data (`app/data/`)](#5-dashboard-data-appdata)
6. [Hierarchy node names](#6-hierarchy-node-names)
7. [How to check this dictionary](#7-how-to-check-this-dictionary)

---

## 1. Conventions

| Term | Meaning |
|---|---|
| Series | One item in one store, for example item `FOODS_3_007` in store `CA_1`. There are 500 series (50 items × 10 stores) |
| `id` | The series id: item id + `_` + store id + `_evaluation`, for example `FOODS_3_007_CA_1_evaluation` |
| `item_id` | Product, for example `FOODS_3_007`. All 50 items are in department `FOODS_3` |
| `store_id` | One of `CA_1` to `CA_4`, `TX_1` to `TX_3`, `WI_1` to `WI_3` |
| `state_id` | `CA`, `TX` or `WI` |
| `sales` | Units sold that day (a count, 0 or more) |
| `date` | Calendar date (`datetime64[ns]`) |
| Training period | all days except the last 28 |
| Test period | the last 28 days of the data (d_1914 to d_1941, 25 April to 22 May 2016) |
| `pred` | A point forecast of daily units, never below 0 |
| `lo`, `hi` | The 10% and 90% quantile forecasts (the ends of an 80% interval) |
| Plan window | The first 14 days of the test period, used by the optimizer |

The data runs from **5 December 2013 to 22 May 2016** (900 days). This is the output of `python src/prepare_data.py`: `rows: 450000 | series: 500 | dates: 2013-12-05 00:00:00 -> 2016-05-22 00:00:00`.

---

## 2. Raw data (`data/raw/`)

Downloaded from the Kaggle competition *M5 Forecasting - Accuracy* (Walmart). Not committed to the repository.

The row counts and shapes below were confirmed with `pd.read_csv(...).shape`: `calendar.csv` is (1969, 14), `sell_prices.csv` is (6841121, 4) and `sales_train_evaluation.csv` is (30490, 1947).

### `calendar.csv` (1,969 rows, one per day)

| Column | Meaning | Used by me |
|---|---|---|
| `date` | calendar date | yes |
| `wm_yr_wk` | Walmart week number, the key for prices | yes |
| `weekday`, `wday`, `month`, `year` | weekday name and number, month, year | no (I compute my own from `date`) |
| `d` | day id such as `d_1`, the key for the sales file | yes |
| `event_name_1`, `event_type_1`, `event_name_2`, `event_type_2` | holiday or special event names and types | only `event_name_1` |
| `snap_CA`, `snap_TX`, `snap_WI` | 1 if SNAP food-assistance purchases are allowed that day in that state | yes |

### `sell_prices.csv` (6,841,121 rows, one per store, item and week)

| Column | Meaning |
|---|---|
| `store_id` | store |
| `item_id` | item |
| `wm_yr_wk` | week |
| `sell_price` | selling price (USD). Missing weeks mean the item was not sold that week |

### `sales_train_evaluation.csv` (30,490 rows, one per item and store)

| Column | Meaning |
|---|---|
| `id` | series id |
| `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` | the hierarchy of the series |
| `d_1` to `d_1941` | units sold on each day (wide format) |

---

## 3. Processed data (`data/processed/`)

Not committed. Recreated by running the scripts.

### `long.parquet`

Created by `src/prepare_data.py`. **450,000 rows × 10 columns** (500 series × 900 days).

| Column | Type | Meaning |
|---|---|---|
| `id` | string | series id |
| `item_id` | string | item |
| `store_id` | string | store |
| `state_id` | string | state |
| `sales` | float32 | units sold that day |
| `date` | datetime | calendar date |
| `event_name_1` | string, may be empty | name of the holiday or event on that day |
| `sell_price` | float | price of the item in that store that week. Weeks with no price are filled from the nearest known price in the same series |
| `on_sale` | 0 or 1 | 1 if the item had a price that week (it was on the shelf), 0 if the price was missing before filling |
| `snap` | 0 or 1 | 1 if SNAP purchases were allowed that day in the series' own state |

### `features.parquet`

Created by `src/build_features.py`. **450,000 rows × 30 columns.** It has the same rows as `long.parquet`, without `event_name_1`, plus the features below.

| Column | Type | Meaning | LightGBM | TFT |
|---|---|---|---|---|
| `id`, `item_id`, `store_id`, `state_id` | string | identifiers | | yes (item, store, state as static inputs) |
| `sales` | float32 | units sold (the target) | | yes |
| `date` | datetime | calendar date | | |
| `sell_price` | float | price | yes | yes |
| `on_sale` | 0 or 1 | item on the shelf that week | yes | |
| `snap` | 0 or 1 | SNAP day in the series' state | yes | yes |
| `lag_28`, `lag_35`, `lag_42`, `lag_49`, `lag_56` | float | sales 28, 35, 42, 49 and 56 days earlier. Empty at the start of each series | yes | |
| `roll_mean_7`, `roll_mean_28`, `roll_mean_56` | float | mean sales over the 7, 28 or 56 days that end 28 days before the row's date | yes | |
| `roll_std_28` | float | standard deviation of sales over the 28 days that end 28 days before the row's date | yes | |
| `price_median` | float | median price of the series over the whole 900 days | | |
| `price_ratio` | float | `sell_price / price_median` | yes | yes |
| `promo` | 0 or 1 | promotion proxy: 1 when `price_ratio < 0.95` (price at least 5% below normal) | yes | |
| `dow` | integer 0 to 6 | day of the week, Monday = 0 | yes | yes (as a category) |
| `dom` | integer 1 to 31 | day of the month | yes | |
| `woy` | integer | ISO week of the year | yes | |
| `month` | integer 1 to 12 | month | yes | yes (as a category) |
| `is_event` | 0 or 1 | 1 if there is a holiday or event that day | yes | yes |
| `item_code`, `store_code`, `state_code` | small integer | numeric code of the item, store or state (alphabetical order) | yes | |
| `is_test` | boolean | `True` for the last 28 days of every series (14,000 rows). The other 436,000 rows are training data | | |

Notes:

- LightGBM uses 22 features: the 9 lag and rolling columns, `sell_price`, `price_ratio`, `promo`, `snap`, `is_event`, `on_sale`, `dow`, `dom`, `woy`, `month` and the three codes. The categorical ones are `item_code`, `store_code`, `state_code` and `dow`.
- The TFT also uses `time_idx` (days since the first date), which it creates itself and which is not stored in the file.
- `price_median` is calculated over the whole history including the test days. Prices are known in advance, so I consider this mild, but a stricter setup would calculate it on training data only.
- `promo` is a **proxy** based on price. M5 has no promotion column.

---

## 4. Results (`outputs/`)

Created by the model scripts. All prediction files cover the 28 test days: **500 series × 28 days = 14,000 rows**.

### Prediction files

| File | Created by | Columns | Meaning |
|---|---|---|---|
| `pred_naive.parquet` | `baselines.py` | `id`, `date`, `pred` | Seasonal naive: repeats the last 7 training days |
| `pred_prophet.parquet` | `baselines.py` | `id`, `date`, `pred` | Prophet forecast, one model per series |
| `pred_lgbm.parquet` | `train_lgbm.py` | `id`, `date`, `pred`, `lo`, `hi` | LightGBM point forecast and the 10% (`lo`) and 90% (`hi`) quantile forecasts, with `lo ≤ pred ≤ hi` enforced |
| `pred_tft.parquet` | `train_tft.py` | `id`, `date`, `lo`, `pred`, `hi` | TFT 10% quantile (`lo`), median (`pred`) and 90% quantile (`hi`), in this column order |
| `pred_lgbm_reconciled.parquet` | `reconcile.py` | `date`, `id`, `pred_rec` | Item-store forecast of the chosen reconciliation method (bottom-up), never below 0. Equal to the LightGBM `pred` |

### `metrics.csv` (4 rows)

Created by `evaluate` and `save_metrics` in `src/utils.py`. One row per model. Lower is better in every column.

| Column | Meaning |
|---|---|
| `model` | `Seasonal naive`, `Prophet`, `LightGBM` or `TFT` |
| `WAPE_item_store` | total absolute error divided by total actual sales, over all 500 series and 28 days |
| `WAPE_store` | the same after adding the item-store forecasts and actuals up to store × day |
| `RMSSE` | mean over the 500 series of the forecast RMSE divided by the RMSE of a one-step naive forecast on the training data |

Current content:

| model | WAPE_item_store | WAPE_store | RMSSE |
|---|---|---|---|
| Seasonal naive | 0.4666 | 0.1163 | 0.9273 |
| Prophet | 0.4626 | 0.1276 | 0.8529 |
| LightGBM | 0.4097 | 0.1004 | 0.7941 |
| TFT | 0.3919 | 0.1293 | 0.7853 |

### `reconciliation_metrics.csv` (12 rows)

Created by `reconcile.py`. One row per forecast and hierarchy level.

| Column | Meaning |
|---|---|
| `forecast` | `base (unreconciled)`, `bottom_up` or `mint_wls_struct` |
| `level` | `total`, `state`, `store` or `item_store` |
| `WAPE` | WAPE of that forecast at that level |
| `coherence_gap` | largest absolute difference (units per day) between a forecast and the sum of its parts, taken over all three links: items against their store, stores against their state and states against the total. It is the same on all four rows of a forecast. The per-link values are printed by `reconcile.py` but not saved |

Current content:

| forecast | level | WAPE | coherence_gap |
|---|---|---|---|
| base (unreconciled) | total | 0.0817 | 241.043 |
| base (unreconciled) | state | 0.1063 | 241.043 |
| base (unreconciled) | store | 0.1309 | 241.043 |
| base (unreconciled) | item_store | 0.4097 | 241.043 |
| bottom_up | total | 0.0584 | 0 |
| bottom_up | state | 0.0767 | 0 |
| bottom_up | store | 0.1004 | 0 |
| bottom_up | item_store | 0.4097 | 0 |
| mint_wls_struct | total | 0.0748 | 0 |
| mint_wls_struct | state | 0.0953 | 0 |
| mint_wls_struct | store | 0.1167 | 0 |
| mint_wls_struct | item_store | 0.4166 | 0 |

Per-link gaps printed by `reconcile.py` (units per day, largest over the 28 days):

| Forecast | Items → store | Stores → state | States → total |
|---|---|---|---|
| Base (unreconciled) | 241.043 | 25.289 | 15.795 |
| Bottom-up | 0 | 0 | 0 |
| MinT (`wls_struct`) | 0 | 0 | 0 |

### `inventory_results.csv` (3 rows)

Created by `optimize.py`. Each policy is scored against the real sales of the 14-day plan window.

| Column | Meaning |
|---|---|
| `policy` | `Naive (order forecast mean)`, `Naive scaled to same spend` or `Optimized (OR-Tools)` |
| `holding_cost` | cost of leftover units (holding cost per unit × units left over) |
| `stockout_cost` | cost of unmet demand (lost margin per unit × units short) |
| `total_cost` | `holding_cost + stockout_cost`. Purchase spend is not included |
| `fill_rate` | units sold divided by units demanded (0 to 1) |
| `order_spend` | total unit cost of everything ordered |

Current content:

| policy | holding_cost | stockout_cost | total_cost | fill_rate | order_spend |
|---|---|---|---|---|---|
| Naive (order forecast mean) | 656.028 | 6696.633 | 7352.661 | 0.878 | 102134.508 |
| Naive scaled to same spend | 1572.206 | 3050.529 | 4622.735 | 0.946 | 128965.648 |
| Optimized (OR-Tools) | 1527.323 | 2670.162 | 4197.485 | 0.957 | 128955.512 |

### `order_recommendations.csv` (500 rows)

Created by `optimize.py`. One row per item and store: the order plan for the 14-day window.

| Column | Meaning |
|---|---|
| `store_id` | store |
| `item_id` | item |
| `on_hand` | assumed current stock, 2 days of expected demand, in units |
| `forecast_lo` | 10th percentile of demand over the 14 days (units) |
| `forecast_mid` | expected demand over the 14 days (sum of the reconciled daily forecasts) |
| `forecast_hi` | 90th percentile of demand over the 14 days |
| `order_qty` | recommended order in units (integer, 0 or more) |

### Other files

| File | Created by | Columns and meaning |
|---|---|---|
| `lgbm_importance.csv` | `train_lgbm.py` | 22 rows. The first column has no header name (pandas shows it as `Unnamed: 0`) and holds the feature name; the column `0` is the importance as the number of times the feature was used in a split. Sorted from most to least important |
| `interval_coverage.txt` | `train_lgbm.py` | One line with the share of real test sales that fell between `lo` and `hi` (about 0.823) |
| `tft_ckpt/tft-best.ckpt`, `tft-best-v1.ckpt`, … | `train_tft.py` | Saved TFT models. Every run adds a new file with a version suffix. Not committed |
| `lightning_logs/` (project root) | `train_tft.py` | PyTorch Lightning training logs. Not committed |

---

## 5. Dashboard data (`app/data/`)

Created by `src/export_app_data.py`. These small files are committed because the Docker image needs them.

| File | Rows | Columns |
|---|---|---|
| `history.parquet` | 60,500 (500 series × 121 days) | `id`, `item_id`, `store_id`, `date`, `sales`, `is_test`. The last 121 days of actual sales, including the 28 test days |
| `forecasts.parquet` | 14,000 | `id`, `date`, `lgbm`, `lgbm_lo`, `lgbm_hi` (LightGBM and its interval), `naive`, `prophet`, `tft_lo`, `tft`, `tft_hi` (TFT 10%, median and 90%), `lgbm_reconciled` |
| `metrics.csv` | 4 | copy of `outputs/metrics.csv` |
| `inventory_results.csv` | 3 | copy of `outputs/inventory_results.csv` |
| `order_recommendations.csv` | 500 | copy of `outputs/order_recommendations.csv` |
| `reconciliation_metrics.csv` | 12 | copy of `outputs/reconciliation_metrics.csv` |

The dashboard also calculates two numbers when it starts: the **interval coverage** (the share of test-day sales between `lgbm_lo` and `lgbm_hi`) and the **cost change against the equal-spend baseline** (`total_cost` of the optimized policy divided by `total_cost` of the equal-spend naive policy, minus 1).

---

## 6. Hierarchy node names

These are the column names of the wide tables that `reconcile.py` builds (514 columns, one per node).

| Level | Count | Node names |
|---|---|---|
| Total | 1 | `total` |
| State | 3 | `CA`, `TX`, `WI` |
| Store | 10 | `CA_1`, `CA_2`, `CA_3`, `CA_4`, `TX_1`, `TX_2`, `TX_3`, `WI_1`, `WI_2`, `WI_3` |
| Item-store | 500 | the series `id`, for example `FOODS_3_007_CA_1_evaluation` |

---

## 7. How to check this dictionary

I ran the commands below after the final pipeline run. The shapes and columns of every processed, output and dashboard file match the tables in this document, and the shapes and column names of the three raw files match section 2.

```powershell
python -c "import pandas as pd, glob; [print(f, pd.read_parquet(f).shape, list(pd.read_parquet(f).columns)) for f in sorted(glob.glob('data/processed/*.parquet') + glob.glob('outputs/*.parquet') + glob.glob('app/data/*.parquet'))]"
```

```powershell
python -c "import pandas as pd, glob; [print(f, pd.read_csv(f).shape, list(pd.read_csv(f).columns)) for f in sorted(glob.glob('outputs/*.csv') + glob.glob('app/data/*.csv'))]"
```

To check the raw files (shapes):

```powershell
python -c "import pandas as pd; [print(f, pd.read_csv('data/raw/'+f).shape) for f in ['calendar.csv','sell_prices.csv','sales_train_evaluation.csv']]"
```

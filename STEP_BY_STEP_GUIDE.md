# Step-by-Step Guide: Freight Rate Assignment

Follow in order. Total time: about 2–3 hours including the Loom.

---

## Step 1 — Set up the project folder

```
freight-rate/
├── data/
│   ├── train_test.csv
│   ├── validation.csv
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv
├── score.py
├── requirements.txt
├── train.py          <- you will write this
└── README.md
```

Note: the uploaded files used dashes (`train-test.csv`). The code expects underscores (`train_test.csv`). Rename them.

Add `scikit-learn>=1.4,<2` to `requirements.txt`, then:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Step 2 — Understand the task (5 min)

- Train on `train_test.csv` (Jan–Oct 2025, with `posted_rate`).
- Predict `validation.csv` (Nov–Dec 2025, no rate). 12,000 rows.
- The December chart rows have no `market_index` or `quote_signal`.

**Key insight: the test set is the future.** So you must validate forward in time, not with a random split.

---

## Step 3 — Explore the data (20 min)

Open a notebook or Python shell and run these checks. Write down what you find; this is your Loom material.

```python
import pandas as pd, numpy as np
tr  = pd.read_csv("data/train_test.csv", parse_dates=["date"])
val = pd.read_csv("data/validation.csv", parse_dates=["date"])

print(tr.isna().sum())                          # missing: weight, market_index
print((tr.weight < 0).sum())                    # negative weights
print(tr.date.min(), tr.date.max(), val.date.min(), val.date.max())
print((set(val.pickup) | set(val.delivery)) - (set(tr.pickup) | set(tr.delivery)))   # unseen cities
print(tr.groupby(tr.date.dt.to_period("M")).market_index.mean())
print((tr.posted_rate / tr.distance).describe())
```

The unseen-city check should return 8 cities.

**What you should find:**

| Finding | Number |
|---|---|
| Train dates / validation dates | Jan–Oct / Nov–Dec |
| Negative weights | 292 (sign errors) |
| Missing weight / market_index | 300 / 374 |
| Cities in validation not in train | 8 |
| `market_index` is date-level (small spread within a day) | std ≈ 0.025 |
| Rate is multiplicative in distance | log-log is near-linear |
| Spiked labels (×2–5 or ×0.2–0.4) | about 1.4% of rows |
| `market_index` range shift | train 0.68–1.47, validation 0.72–1.10 |

---

## Step 4 — Decide the cleaning (10 min)

| Problem | Fix |
|---|---|
| Negative weight | `abs(weight)` |
| Missing weight | training median |
| Missing market_index | that date's median market_index |
| Spiked labels | keep them; use a robust loss instead |
| Unseen cities | do not use city names or lat/lon as features |

Compute the date-level medians from **feature columns only** (train + validation). Never from `posted_rate`.

---

## Step 5 — Engineer features (10 min)

Use log scale because rate is multiplicative:

- `ld` = log(distance)
- `lw` = log(abs(weight))
- `lmi` = log(market_index)
- `lq` = log(quote_signal)
- `reefer`, `flatbed` = 0/1 flags (Dry Van is the baseline)

I tested day-of-week, day-of-year and lat/lon differences. They made forward validation worse, so I left them out. You can repeat that test to show you tried it.

---

## Step 6 — Set up forward-in-time validation (15 min)

Each fold trains on everything before a cutoff and tests on the next 2 months:

| Fold | Train | Test |
|---|---|---|
| 1 | Jan – Apr | May – Jun |
| 2 | Jan – Jun | Jul – Aug |
| 3 | Jan – Aug | Sep – Oct |

Report MAE, MAPE, median APE and RMSE in dollars. The scoring metric isn't disclosed, so report several.

---

## Step 7 — Build the models (20 min)

1. **Baseline:** `HuberRegressor` on `log(rate)`.
2. **Main model:** `HistGradientBoostingRegressor(loss="absolute_error")` on `log(rate)`, then `exp()` the prediction.
3. **Drift correction:** take the median of (actual log rate − predicted log rate) over the last 14 days before the cutoff, and add it to every prediction.

Why absolute-error loss: the spiked labels would drag a squared-error model. L1 ignores them.

Expected result (3-fold mean):

| Model | MAE | MedAPE |
|---|---|---|
| Huber baseline | ~139 | ~3.4% |
| HGB | ~125 | ~2.5% |
| HGB + drift correction | ~118 | ~2.15% |

---

## Step 7b — Use the complete code

`train.py` (attached) does steps 4–8 in one run. Read it top to bottom before running it. You need to be able to explain every function.

| Function | What it does |
|---|---|
| `date_signals` | daily median market_index and quote_signal |
| `build_features` | cleaning + log features |
| `make_model` | the gradient boosting model |
| `level_offset` | the 14-day drift correction |
| `metrics` | MAE, MAPE, MedAPE, RMSE |
| `main` | CV loop, final fit, writes both prediction files |

---

## Step 8 — Run and score

```bash
python train.py
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

You should see:
```
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
```

Troubleshooting:
- `FileNotFoundError` → files still have dashes; rename to underscores.
- `exactly two columns` error → the output must be `load_id,predicted_rate` only.
- December error → the file must keep all 7 original columns.

---

## Step 9 — GitHub repo (10 min)

```bash
git init
printf '.venv/\n__pycache__/\nscorer_results/\n' > .gitignore
git add .
git commit -m "Freight rate prediction solution"
git branch -M main
git remote add origin https://github.com/<you>/freight-rate.git
git push -u origin main
```

Make the repo **public** (or share access). The assignment says "accessible". Include `README.md` with the run commands.

---

## Step 10 — Write the report (30 min)

Word or PDF. Sections:

1. **Data and split:** train is Jan–Oct, validation is Nov–Dec, so the split is forward in time. Say why random splits would leak.
2. **Validation approach:** the three folds, the metrics, the table of results.
3. **Data-quality issues:** the table from Step 4.
4. **Model choice:** baseline vs HGB vs HGB + offset, and why L1 loss on log target.
5. **December chart:** insert `scorer_results/candidate_december.png` and add 2 sentences explaining it. Predictions vary by only about 1.2% across the month. The y-axis is zoomed.
6. **Limitations:** only 10 months of data, so no annual seasonality; the drift offset was tuned on one year.

I included a finished version, `Freight_Rate_Report.docx`, to use as a template. Rewrite it in your own words.

---

## Step 11 — Loom script (2–3 min)

| Time | Say |
|---|---|
| 0:00–0:30 | **Findings:** train is Jan–Oct, test is Nov–Dec; rate is roughly linear in log distance; Reefer about +12%, Flatbed about +8%. |
| 0:30–1:00 | **Data quality:** negative weights → abs; missing weight/market_index → median; about 1.4% spiked labels → robust loss; 8 unseen cities → no city features. |
| 1:00–1:30 | **Model:** gradient boosting handles the non-linear distance curve; L1 loss on log rate ignores spikes; beat the linear baseline by about 15% MAE. |
| 1:30–2:00 | **Validation:** three forward folds; why not random split. |
| 2:00–2:45 | **Code:** show `build_features`, `make_model`, the CV loop, `level_offset`. |
| 2:45–3:00 | Mention limitations and show the December chart. |

Tip: rehearse once, then record. Keep `train.py` open on screen.

---

## Step 12 — Final checklist

- [ ] `validation_predictions.csv` has exactly `load_id,predicted_rate`, 12,000 rows
- [ ] `score.py` runs with no errors
- [ ] GitHub repo is accessible, with code, `requirements.txt`, README
- [ ] Report contains the validation approach and `candidate_december.png`
- [ ] Loom is 2–3 minutes and covers all five required topics

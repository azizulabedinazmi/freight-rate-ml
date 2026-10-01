# Freight Rate Prediction Challenge — Solution

Predicts `posted_rate` for 12,000 held-out loads (Nov–Dec 2025) from 48,000 labelled loads (Jan–Oct 2025).

## Run

```bash
python -m pip install -r requirements.txt
python train.py            # validates (time-forward CV), fits, writes predictions
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`train.py` writes `validation_predictions.csv` (`load_id,predicted_rate`), fills `data/december_chart_inputs.csv`,
and writes `report/cv_metrics.csv`. The scorer creates `scorer_results/candidate_december.png`.
Runtime: under a minute on a laptop; fully deterministic (`random_state=42`).

## Approach (summary)

| Step | Decision | Why |
|---|---|---|
| Split | Expanding-window, **forward-in-time** folds (train < cutoff, predict next 2 months) | Validation set is Nov–Dec; train is Jan–Oct. A random split would leak time/level and over-state accuracy. |
| Cleaning | `abs(weight)`; missing weight → train median; missing `market_index` → that date's median | 292 negative weights are sign errors; `market_index` is a date-level signal. |
| Labels | Not dropped; robust loss (absolute error) on `log(rate)` | ~1.4% of labels are random ×2–5 / ×0.2–0.4 spikes unrelated to any feature. |
| Features | log distance, log weight, equipment, `market_index`, `quote_signal` | Rolling CV showed date-of-year, day-of-week and coordinates did not help (and coordinates would not generalise to 8 cities unseen in training). |
| Model | `HistGradientBoostingRegressor(loss="absolute_error")` on `log(rate)` | HGB alone beats the Huber log-linear baseline by ~10% MAE; with the drift step below the final model is ~15% better on MAE and ~37% on median APE (3-fold mean). |
| Drift | Add median log-residual of the last 14 training days to all predictions | Rates drift slowly in a way `market_index` does not explain; carry-forward cut CV MAE a further ~6%. |
| December chart | Rows lack `market_index`/`quote_signal` → use the date-level medians from validation feature columns (no labels) | Keeps the model's inputs consistent with training. |

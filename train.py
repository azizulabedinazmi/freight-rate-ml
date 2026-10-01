import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor

SEED = 42
LEVEL_WINDOW_DAYS = 14
FEATURES = ["ld", "lw", "lmi", "lq", "reefer", "flatbed"]
FOLDS = [("2025-05-01", "2025-07-01"), ("2025-07-01", "2025-09-01"), ("2025-09-01", "2025-11-01")]


def date_signals(*frames):
    x = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames if "market_index" in f])
    return x.groupby("date").agg(mi_day=("market_index", "median"), qs_day=("quote_signal", "mean"))


def build_features(df, day, weight_median):
    out = pd.DataFrame(index=df.index)
    out["ld"] = np.log(df["distance"])
    out["lw"] = np.log(df["weight"].abs().fillna(weight_median))
    out["lmi"] = np.log(df["market_index"].fillna(df["date"].map(day["mi_day"])))
    out["lq"] = np.log(df["quote_signal"].fillna(df["date"].map(day["qs_day"])))
    out["reefer"] = (df["equipment"] == "Reefer").astype(int)
    out["flatbed"] = (df["equipment"] == "Flatbed").astype(int)
    return out


def make_model():
    return HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=400, learning_rate=0.05,
        max_leaf_nodes=15, random_state=SEED,
    )


def level_offset(model, X, y, dates, cutoff):
    recent = (dates >= cutoff - pd.Timedelta(days=LEVEL_WINDOW_DAYS)) & (dates < cutoff)
    return float(np.median(y[recent] - model.predict(X.loc[recent, FEATURES])))


def metrics(pred, true):
    err = np.abs(pred - true)
    return {"MAE": err.mean(), "MAPE_%": (err / true).mean() * 100,
            "MedAPE_%": np.median(err / true) * 100, "RMSE": np.sqrt((err ** 2).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="validation_predictions.csv")
    args = ap.parse_args()
    d = Path(args.data_dir)

    train = pd.read_csv(d / "train_test.csv", parse_dates=["date"])
    val = pd.read_csv(d / "validation.csv", parse_dates=["date"])
    dec = pd.read_csv(d / "december_chart_inputs.csv", parse_dates=["date"])
    template = pd.read_csv(d / "validation_predictions_template.csv")

    day = date_signals(train, val)
    wmed = train["weight"].abs().median()
    X = build_features(train, day, wmed)
    y = np.log(train["posted_rate"])

    rows = []
    for s, e in FOLDS:
        a = train["date"] < s
        b = (train["date"] >= s) & (train["date"] < e)
        t = train.loc[b, "posted_rate"].values
        m = make_model().fit(X.loc[a, FEATURES], y[a])
        off = level_offset(m, X.loc[a], y[a], train.loc[a, "date"], pd.Timestamp(s))
        lin = HuberRegressor(max_iter=1000).fit(X.loc[a, FEATURES], y[a])
        preds = {
            "HGB + level offset (final)": np.exp(m.predict(X.loc[b, FEATURES]) + off),
            "HGB no offset": np.exp(m.predict(X.loc[b, FEATURES])),
            "Huber log-linear": np.exp(lin.predict(X.loc[b, FEATURES])),
        }
        for name, p in preds.items():
            rows.append({"fold": f"{s}..{e}", "model": name, **metrics(p, t)})
    cv = pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True)
    cv.to_csv("report/cv_metrics.csv", index=False)
    print(cv.groupby("model")[["MAE", "MAPE_%", "MedAPE_%", "RMSE"]].mean().round(2))

    model = make_model().fit(X[FEATURES], y)
    offset = level_offset(model, X, y, train["date"], train["date"].max() + pd.Timedelta(days=1))

    Xv = build_features(val, day, wmed)
    pred = pd.DataFrame({"load_id": val["load_id"],
                         "predicted_rate": np.exp(model.predict(Xv[FEATURES]) + offset).round(2)})
    sub = template[["load_id"]].merge(pred, on="load_id", how="left")
    assert len(sub) == 12_000 and sub["predicted_rate"].notna().all()
    sub.to_csv(args.out, index=False)

    dec_in = dec.drop(columns=["predicted_rate"]).copy()
    dec_in["market_index"] = dec_in["date"].map(day["mi_day"])
    dec_in["quote_signal"] = dec_in["date"].map(day["qs_day"])
    Xd = build_features(dec_in, day, wmed)
    dec["predicted_rate"] = np.exp(model.predict(Xd[FEATURES]) + offset).round(2)
    dec["date"] = dec["date"].dt.strftime("%Y-%m-%d")
    dec.to_csv(d / "december_chart_inputs.csv", index=False)
    print(f"wrote {args.out} ({len(sub)} rows) and December predictions")


if __name__ == "__main__":
    main()

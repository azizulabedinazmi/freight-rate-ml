"""Builds figures and numeric macros for report.tex. Run from the repo root:
    python train.py && python report/make_assets.py
"""
import sys
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

sys.path.insert(0, ".")
import train as T

OUT = Path("report")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
chart = Path("scorer_results/candidate_december.png")
if not chart.is_file():
    raise FileNotFoundError(
        "Missing scorer_results/candidate_december.png; run score.py before make_assets.py"
    )
shutil.copy2(chart, FIG / chart.name)

TEAL, MID, LIGHT, CORAL, GREY = "#064A56", "#2A9D8F", "#CFE3E6", "#E4572E", "#8A9BA0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": "#25363A", "xtick.color": "#25363A", "ytick.color": "#25363A",
    "axes.grid": True, "grid.color": "#E3EBED", "grid.linewidth": 0.7, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10,
})
EQ_COL = {"Dry Van": TEAL, "Reefer": CORAL, "Flatbed": "#E9A23B"}

train = pd.read_csv("data/train_test.csv", parse_dates=["date"])
val = pd.read_csv("data/validation.csv", parse_dates=["date"])
cv = pd.read_csv("report/cv_metrics.csv")
dec = pd.read_csv("data/december_chart_inputs.csv", parse_dates=["date"])

day = T.date_signals(train, val)
wmed = train["weight"].abs().median()
X = T.build_features(train, day, wmed)
y = np.log(train["posted_rate"])

M = {}
def n(x): return f"{int(x):,}".replace(",", "{,}")
def f(x, d=1): return f"{x:.{d}f}"
M["NTrain"], M["NVal"] = n(len(train)), n(len(val))
M["NegWeightTrain"], M["NegWeightVal"] = n((train.weight < 0).sum()), n((val.weight < 0).sum())
M["MissWeightTrain"], M["MissWeightVal"] = n(train.weight.isna().sum()), n(val.weight.isna().sum())
M["MissMITrain"], M["MissMIVal"] = n(train.market_index.isna().sum()), n(val.market_index.isna().sum())
unseen = (set(val.pickup) | set(val.delivery)) - (set(train.pickup) | set(train.delivery))
M["NUnseen"] = str(len(unseen))
M["UnseenRows"] = n((~val.pickup.isin(set(train.pickup) | set(train.delivery)) |
                     ~val.delivery.isin(set(train.pickup) | set(train.delivery))).sum())
M["UnseenList"] = ", ".join(sorted(unseen))
M["MIWithinStd"] = f(train.groupby("date").market_index.std().mean(), 3)
M["MITrainMin"], M["MITrainMax"] = f(train.market_index.min(), 2), f(train.market_index.max(), 2)
M["MIValMin"], M["MIValMax"] = f(val.market_index.min(), 2), f(val.market_index.max(), 2)
M["MIValMean"] = f(val.market_index.mean(), 2)
M["DistFloor"] = f(train.distance.min(), 0)
M["NCities"] = str(len(set(train.pickup) | set(train.delivery)))

# --- structure of the price (robust linear fit on log scale) ---
hub = HuberRegressor(max_iter=1000).fit(X[T.FEATURES], y)
co = dict(zip(T.FEATURES, hub.coef_))
M["ElastDist"] = f(co["ld"], 2)
M["ReeferPct"] = f((np.exp(co["reefer"]) - 1) * 100, 0)
M["FlatbedPct"] = f((np.exp(co["flatbed"]) - 1) * 100, 0)
res = y - hub.predict(X[T.FEATURES])
core_sigma = 1.4826 * np.median(np.abs(res - np.median(res)))
spike = res.abs() > 0.5
M["CoreSigma"] = f(core_sigma * 100, 1)
M["SpikePct"] = f(spike.mean() * 100, 1)
M["SpikeHiMult"] = f"{np.exp(res[res > .5]).min():.1f}--{np.exp(res[res > .5]).max():.1f}"
M["SpikeLoMult"] = f"{np.exp(res[res < -.5]).min():.2f}--{np.exp(res[res < -.5]).max():.2f}"

# --- final-model offset ---
final = T.make_model().fit(X[T.FEATURES], y)
off = T.level_offset(final, X, y, train.date, train.date.max() + pd.Timedelta(days=1))
M["Offset"] = f"{(np.exp(off) - 1) * 100:+.1f}\\%"

# --- cv tables ---
metrics = ["MAE", "MAPE_%", "MedAPE_%", "RMSE"]
mean = cv.groupby("model")[metrics].mean()
order = ["Huber log-linear", "HGB no offset", "HGB + level offset (final)"]
names = {"Huber log-linear": "Huber log-linear (baseline)", "HGB no offset": "Gradient boosting, L1 loss",
         "HGB + level offset (final)": r"\textbf{Boosting + level offset (final)}"}
rows = []
for k in order:
    r = mean.loc[k]
    cells = [f"{r['MAE']:.1f}", f"{r['MAPE_%']:.2f}", f"{r['MedAPE_%']:.2f}", f"{r['RMSE']:.1f}"]
    if k.endswith("(final)"): cells = [rf"\textbf{{{c}}}" for c in cells]
    rows.append(names[k] + " & " + " & ".join(cells))
M["CVMeanRows"] = " \\\\\n".join(rows)
fr = []
for fold, g in cv.groupby("fold"):
    s, e = fold.split("..")
    lab = pd.Timestamp(s).strftime("%b") + "--" + (pd.Timestamp(e) - pd.Timedelta(days=1)).strftime("%b")
    gi = g.set_index("model")
    fr.append(lab + " & " + " & ".join(f"{gi.loc[k, 'MAE']:.1f}" for k in order))
M["CVFoldRows"] = " \\\\\n".join(fr)
b, f2 = mean.loc["Huber log-linear"], mean.loc["HGB + level offset (final)"]
M["GainMAE"] = f((1 - f2["MAE"] / b["MAE"]) * 100, 0)
M["GainMed"] = f((1 - f2["MedAPE_%"] / b["MedAPE_%"]) * 100, 0)
M["FinalMAE"], M["FinalMed"], M["FinalMAPE"] = f(f2["MAE"], 1), f(f2["MedAPE_%"], 2), f(f2["MAPE_%"], 2)
M["OffsetGainMAE"] = f(mean.loc["HGB no offset", "MAE"] - f2["MAE"], 1)

# --- fig: cv bars ---
fig, ax = plt.subplots(figsize=(6.4, 2.05))
folds = sorted(cv.fold.unique()); w = 0.26
for i, (k, c) in enumerate(zip(order, [GREY, MID, TEAL])):
    vals = [cv[(cv.fold == fo) & (cv.model == k)].MAE.iloc[0] for fo in folds]
    bars = ax.bar(np.arange(3) + (i - 1) * w, vals, w, color=c, label=names[k].replace(r"\textbf{", "").replace("}", ""))
    for bx, v in zip(bars, vals): ax.text(bx.get_x() + w / 2, v + 2, f"{v:.0f}", ha="center", fontsize=7.5)
ax.set_xticks(range(3)); ax.set_xticklabels(["Test May--Jun", "Test Jul--Aug", "Test Sep--Oct"])
ax.set_ylabel("MAE (\\$)"); ax.set_ylim(0, 190); ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax.grid(axis="x", visible=False)
fig.savefig(FIG / "cv_mae.pdf"); plt.close(fig)

# --- fig: eda log-log ---
smp = train.sample(7000, random_state=0)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
for eq in ["Flatbed", "Reefer", "Dry Van"]:
    g = smp[smp.equipment == eq]
    ax.scatter(g.distance, g.posted_rate, s=3, alpha=.35, color=EQ_COL[eq], label=eq, rasterized=True, linewidths=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Distance (miles, log scale)"); ax.set_ylabel("Posted rate (\\$, log scale)")
lg = ax.legend(frameon=False, markerscale=4, fontsize=8, loc="upper left")
for h in lg.legend_handles: h.set_alpha(1)
fig.savefig(FIG / "eda_loglog.pdf"); plt.close(fig)

# --- fig: residual histogram with label spikes ---
fig, ax = plt.subplots(figsize=(6.4, 2.6))
bins = np.linspace(-2, 2, 161)
ax.hist(res, bins=bins, color=TEAL, edgecolor="none")
ax.hist(res[spike], bins=bins, color=CORAL, edgecolor="none", label=f"|residual| > 0.5 ({spike.mean()*100:.1f}% of rows)")
ax.set_yscale("log"); ax.set_xlabel("log(actual) $-$ log(robust linear prediction)"); ax.set_ylabel("Loads (log scale)")
ax.legend(frameon=False, fontsize=8)
fig.savefig(FIG / "label_spikes.pdf"); plt.close(fig)

# --- fig: market index drift ---
allx = pd.concat([train.assign(s="train"), val.assign(s="val")])
d = allx.groupby("date").market_index.mean()
fig, ax = plt.subplots(figsize=(6.4, 2.5))
sm = d.rolling(7, center=True, min_periods=4).mean()
ax.plot(d, color=GREY, lw=0.5, alpha=.35, label="Daily mean")
ax.plot(sm[sm.index < "2025-11-01"], color=TEAL, lw=2, label="7-day average, training")
ax.plot(sm[sm.index >= "2025-11-01"], color=CORAL, lw=2, label="7-day average, prediction period")
ax.axvspan(pd.Timestamp("2025-11-01"), d.index.max(), color=CORAL, alpha=.07)
ax.set_ylabel("market_index"); ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=1)
fig.autofmt_xdate(rotation=0, ha="center")
fig.savefig(FIG / "market_index.pdf"); plt.close(fig)

# --- last fold: predictions, errors, importance ---
s = pd.Timestamp("2025-09-01"); e = pd.Timestamp("2025-11-01")
a = train.date < s; bm = (train.date >= s) & (train.date < e)
m3 = T.make_model().fit(X.loc[a, T.FEATURES], y[a])
o3 = T.level_offset(m3, X.loc[a], y[a], train.loc[a, "date"], s)
pl = m3.predict(X.loc[bm, T.FEATURES]) + o3
act = y[bm].values
fig, ax = plt.subplots(figsize=(3.4, 3.4))
hb = ax.hexbin(np.exp(act), np.exp(pl), xscale="log", yscale="log", gridsize=55, bins="log", cmap="GnBu", mincnt=1)
lim = [min(np.exp(act).min(), np.exp(pl).min()) * .9, max(np.exp(act).max(), np.exp(pl).max()) * 1.1]
ax.plot(lim, lim, color=CORAL, lw=1, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual rate (\\$)"); ax.set_ylabel("Predicted rate (\\$)")
fig.savefig(FIG / "pred_vs_actual.pdf"); plt.close(fig)

dist = train.loc[bm, "distance"].values
ape = np.abs(np.exp(pl) - np.exp(act)) / np.exp(act) * 100
bins_d = [0, 250, 500, 1000, 1500, 2500, 4000]
labs = ["<250", "250--500", "500--1k", "1k--1.5k", "1.5k--2.5k", ">2.5k"]
cat = pd.cut(dist, bins_d, labels=labs)
medape = pd.Series(ape).groupby(cat, observed=True).median()
fig, ax = plt.subplots(figsize=(3.4, 3.4))
ax.bar(range(len(medape)), medape.values, color=TEAL)
for i, v in enumerate(medape.values): ax.text(i, v + .05, f"{v:.1f}", ha="center", fontsize=7.5)
ax.set_xticks(range(len(medape))); ax.set_xticklabels(labs, rotation=35, ha="right", fontsize=7.5)
ax.set_ylabel("Median abs. % error"); ax.set_xlabel("Distance bucket (miles)"); ax.grid(axis="x", visible=False)
fig.savefig(FIG / "error_by_distance.pdf"); plt.close(fig)

rng = np.random.RandomState(0)
Xb = X.loc[bm, T.FEATURES].reset_index(drop=True)
base_err = np.median(np.abs(m3.predict(Xb) + o3 - act))
imp = {}
nice = {"ld": "log distance", "lw": "log weight", "lmi": "market_index", "lq": "quote_signal",
        "reefer": "Reefer flag", "flatbed": "Flatbed flag"}
for c in T.FEATURES:
    vals = []
    for _ in range(5):
        Xp = Xb.copy(); Xp[c] = rng.permutation(Xp[c].values)
        vals.append(np.median(np.abs(m3.predict(Xp) + o3 - act)) - base_err)
    imp[nice[c]] = np.mean(vals)
imp = pd.Series(imp).sort_values()
fig, ax = plt.subplots(figsize=(6.4, 2.3))
ax.barh(imp.index, imp.values * 100, color=[TEAL if v > imp.values.max() * .5 else MID for v in imp.values])
ax.set_xlabel("Increase in median abs. log error when shuffled (percentage points)"); ax.grid(axis="y", visible=False)
fig.savefig(FIG / "importance.pdf"); plt.close(fig)
M["TopFeature"] = imp.index[-1]
M["MedAPEFold"] = f(np.median(ape), 2)
M["MedAPEShort"] = f(medape.iloc[0], 1)
M["MedAPELong"] = f(medape.iloc[-1], 1)

# --- December ---
p = dec.predicted_rate
M["DecMin"], M["DecMax"], M["DecMean"] = f(p.min(), 0), f(p.max(), 0), f(p.mean(), 0)
M["DecSpreadPct"] = f((p.max() - p.min()) / p.mean() * 100, 1)

with open(OUT / "generated.tex", "w") as fh:
    fh.write("% auto-generated by make_assets.py - do not edit\n")
    for k, v in M.items():
        fh.write("\\newcommand{\\" + "x" + k + "}{" + v + "}\n" if "\n" not in v else "")
    for k in ("CVMeanRows", "CVFoldRows"):
        Path(OUT / f"{k}.tex").write_text(M[k] + "\n")
print("assets written:", len(M), "macros;", len(list(FIG.glob('*.pdf'))), "figures")
for k in ["GainMAE", "GainMed", "FinalMAE", "Offset", "SpikePct", "ElastDist", "ReeferPct", "FlatbedPct", "TopFeature", "DecMin", "DecMax", "DecSpreadPct", "UnseenRows", "SpikeHiMult", "SpikeLoMult", "CoreSigma"]:
    print(k, M[k])

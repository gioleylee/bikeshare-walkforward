"""Backtest one-day-ahead Capital Bikeshare forecasts in chronological order."""

import argparse
import csv
import json
from datetime import date
from math import ceil
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt


START = date(2012, 1, 1)
TEST_START = date(2012, 7, 1)
REFIT_EVERY = 14
CALIBRATION = 60
LAGS = (1, 2, 7, 14, 28)


def read_series(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    dates = [date.fromisoformat(row["dteday"]) for row in rows]
    if not dates or any((b - a).days != 1 for a, b in zip(dates, dates[1:])):
        raise ValueError("expected a complete series in chronological order")
    counts = np.array([int(row["cnt"]) for row in rows])
    holidays = np.array([int(row["holiday"]) for row in rows])
    if np.any(counts < 0) or np.any((holidays != 0) & (holidays != 1)):
        raise ValueError("invalid counts or holiday indicators")
    return dates, counts, holidays


def features(dates, counts, holidays):
    rows = []
    for t in range(max(LAGS), len(dates)):
        weekday = [int(dates[t].weekday() == day) for day in range(7)]
        angle = 2 * np.pi * (dates[t].timetuple().tm_yday - 1) / 365.25
        rows.append([
            t, np.sin(angle), np.cos(angle), holidays[t], *weekday,
            *(counts[t - lag] for lag in LAGS),
            counts[t - 7:t].mean(), counts[t - 28:t].mean(),
        ])
    return np.asarray(rows, dtype=float)


def rolling_predict(model, x, y, first, cadence):
    estimates = []
    for index in range(first, len(y), cadence):
        model.fit(x[:index], y[:index])
        window = x[index:index + cadence]
        estimates.extend(np.maximum(model.predict(window), 0))
    return np.asarray(estimates)


def intervals(predictions, truth, window=CALIBRATION, coverage=0.9):
    residuals = np.abs(truth - predictions)
    rank = min(window - 1, ceil((window + 1) * coverage) - 1)
    radii = np.full(len(predictions), np.nan)
    for index in range(window, len(predictions)):
        radii[index] = np.partition(residuals[index - window:index], rank)[rank]
    return np.maximum(predictions - radii, 0), predictions + radii


def metrics(pred, truth, lower, upper):
    error = truth - pred
    return {
        "days": len(truth),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "interval_coverage": float(np.mean((truth >= lower) & (truth <= upper))),
        "mean_interval_width": float(np.mean(upper - lower)),
    }


def run(source, output):
    dates, counts, holidays = read_series(source)
    first, test_start = dates.index(START), dates.index(TEST_START)
    lag = max(LAGS)
    if first <= lag or test_start - first < CALIBRATION:
        raise ValueError("dataset does not cover the required training and calibration dates")
    x = features(dates, counts, holidays)
    y = counts[lag:]
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "boosting": HistGradientBoostingRegressor(
            learning_rate=0.07, max_iter=120, max_leaf_nodes=15, l2_regularization=1,
            early_stopping=False, random_state=42,
        ),
    }
    all_predictions = {"yesterday": counts[first - 1:-1], "last_week": counts[first - 7:-7]}
    with threadpool_limits(limits=1):
        for name, model in models.items():
            all_predictions[name] = rolling_predict(model, x, y, first - lag, REFIT_EVERY)
    observed = counts[first:]
    offset = test_start - first
    dates = dates[test_start:]
    truth = observed[offset:]
    table = {"date": dates, "actual": truth}
    report = {"test_start": str(TEST_START), "test_end": str(dates[-1]), "models": {}}
    figure, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True, layout="constrained")
    for axis in axes:
        axis.plot(dates, truth, color="#172554", label="Observed", linewidth=1.1)
        axis.set_ylabel("Daily rentals")
        axis.grid(alpha=0.2)
    for name, predictions in all_predictions.items():
        lower, upper = intervals(predictions, observed)
        pred, lo, hi = predictions[offset:], lower[offset:], upper[offset:]
        table.update({name: pred, f"{name}_lower": lo, f"{name}_upper": hi})
        report["models"][name] = metrics(pred, truth, lo, hi)
        if name in ("last_week", "ridge"):
            axes[0].plot(dates, pred, label=name.replace("_", " ").title(), linewidth=1, alpha=0.75)
        if name == "ridge":
            axes[1].fill_between(dates, lo, hi, color="#0284c7", alpha=0.2, label="Ridge 90% rolling band")
    for axis in axes:
        axis.legend(loc="upper right")
    axes[0].set_title("One-day-ahead predictions and observations")
    axes[1].set_title("Rolling residual band for ridge")
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / "backtest.png", dpi=160)
    plt.close(figure)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(table)
        writer.writerows(zip(*table.values()))
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/day.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.source, args.output)

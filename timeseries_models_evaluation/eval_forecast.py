# eval_forecast.py
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from utils.forecast import fetch_weather_data
from prophet import Prophet
import logging
logging.getLogger("cmdstanpy").disabled = True


# Config
COUNTRIES = {
    "Singapore": "singapore",
    "United States": "america",
    "India": "india",
}
# days
PARAMS = ["precipitation", "temp_max", "temp_min", "windspeed", "soil_moisture"]
# use 2y of history if available
HORIZONS = [3, 7, 14]  
HISTORY_YEARS = 2.0    
# rolling window size for each backtest
TRAIN_WINDOW_DAYS = 365

# Metrics
def mae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape_safe(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = np.abs(y_true) > 1e-9
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100.0)

def r2_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

# Backtesting helpers
def _fit_prophet_single_series(df: pd.DataFrame, column: str) -> Prophet:
    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=False,
        yearly_seasonality=(len(df) >= 365),
        changepoint_prior_scale=0.01,
    )
    m.fit(df.rename(columns={column: "y"})[["ds", "y"]])
    return m

def backtest_prophet(df: pd.DataFrame, column: str, horizon_days: int, train_window_days: int) -> Dict[str, float]:
    series = df.dropna(subset=[column]).sort_values("ds").reset_index(drop=True)
    if len(series) < train_window_days + horizon_days + 10:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE%": float("nan"), "R2": float("nan"), "n_tests": 0}

    errors = []
    for cutoff in range(train_window_days, len(series) - horizon_days):
        train = series.iloc[cutoff - train_window_days:cutoff][["ds", column]]
        future_truth = series.iloc[cutoff:cutoff + horizon_days][["ds", column]]

        try:
            m = _fit_prophet_single_series(train, column)
            future = m.make_future_dataframe(periods=horizon_days, freq="D")
            fc = m.predict(future).tail(horizon_days)
            y_pred = fc["yhat"].values
            y_true = future_truth[column].values
        except Exception:
            continue

        errors.append((
            mae(y_true, y_pred),
            rmse(y_true, y_pred),
            mape_safe(y_true, y_pred),
            r2_score(y_true, y_pred),
        ))

    if not errors:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE%": float("nan"), "R2": float("nan"), "n_tests": 0}

    arr = np.array(errors)
    return {
        "MAE": float(np.mean(arr[:, 0])),
        "RMSE": float(np.mean(arr[:, 1])),
        "MAPE%": float(np.nanmean(arr[:, 2])),
        "R2": float(np.nanmean(arr[:, 3])),
        "n_tests": int(len(errors)),
    }

def backtest_naive_persistence(df: pd.DataFrame, column: str, horizon_days: int, train_window_days: int) -> Dict[str, float]:
    series = df.dropna(subset=[column]).sort_values("ds").reset_index(drop=True)
    if len(series) < train_window_days + horizon_days + 1:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE%": float("nan"), "R2": float("nan"), "n_tests": 0}

    errors = []
    for cutoff in range(train_window_days, len(series) - horizon_days):
        last_obs = series.iloc[cutoff - 1][column]
        y_pred = np.array([last_obs] * horizon_days)
        y_true = series.iloc[cutoff:cutoff + horizon_days][column].values
        errors.append((
            mae(y_true, y_pred),
            rmse(y_true, y_pred),
            mape_safe(y_true, y_pred),
            r2_score(y_true, y_pred),
        ))

    if not errors:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE%": float("nan"), "R2": float("nan"), "n_tests": 0}

    arr = np.array(errors)
    return {
        "MAE": float(np.mean(arr[:, 0])),
        "RMSE": float(np.mean(arr[:, 1])),
        "MAPE%": float(np.nanmean(arr[:, 2])),
        "R2": float(np.nanmean(arr[:, 3])),
        "n_tests": int(len(errors)),
    }

# Orchestrates evaluation over all countries, horizons, and parameters.
# Produces a tidy DataFrame of metrics for Prophet vs a NaivePersistence baseline.
def evaluate_all() -> pd.DataFrame:
    # Collect per-evaluation metric dicts here; converted to a DataFrame at the end.
    rows = []
    # Total loop count to size the global progress bar (countries × horizons × params).
    total_iterations = len(COUNTRIES) * len(HORIZONS) * len(PARAMS)
    
    # Single overall progress bar for the nested loops below.
    with tqdm(total=total_iterations, desc="Overall Progress", ncols=100) as pbar:
        # Iterate each configured country: label is for reporting; key is used to fetch data.
        for label, key in COUNTRIES.items():
            # Pull historical weather (optionally with soil moisture) for this country once.
            df = fetch_weather_data(country=key, history_years=HISTORY_YEARS, include_soil_moisture=True)

            # Sweep through each forecast horizon (e.g., 3/7/14).
            for horizon in HORIZONS:
                # Evaluate each weather parameter (e.g., precipitation, temperature, wind, soil moisture).
                for param in PARAMS:
                    # Guard: if the entire column is NaN, record NaN metrics and skip modeling to save time.
                    if df[param].isna().all():
                        # Insert a placeholder metrics row for Prophet to keep the output table complete.
                        rows.append({
                            "country": label,
                            "param": param,
                            "horizon_days": horizon,
                            "model": "Prophet",
                            "MAE": float("nan"), "RMSE": float("nan"),
                            "MAPE%": float("nan"), "R2": float("nan"), "n_tests": 0
                        })
                        # Insert a placeholder metrics row for the naive baseline under the same identifiers.
                        rows.append({
                            "country": label,
                            "param": param,
                            "horizon_days": horizon,
                            "model": "NaivePersistence",
                            "MAE": float("nan"), "RMSE": float("nan"),
                            "MAPE%": float("nan"), "R2": float("nan"), "n_tests": 0
                        })
                        # Advance the progress bar for this (country, horizon, param) triplet and continue.
                        pbar.update(1)
                        continue

                    # Run Prophet backtest with a rolling train/test setup at the requested horizon.
                    m_prophet = backtest_prophet(
                        df,
                        param,
                        horizon_days=horizon,
                        train_window_days=TRAIN_WINDOW_DAYS
                    )
                    # Always compute a naive persistence baseline for apples-to-apples comparison.
                    m_naive = backtest_naive_persistence(
                        df,
                        param,
                        horizon_days=horizon,
                        train_window_days=TRAIN_WINDOW_DAYS
                    )

                    # Record Prophet metrics along with identifying metadata columns.
                    rows.append({
                        "country": label,
                        "param": param,
                        "horizon_days": horizon,
                        "model": "Prophet",
                        **m_prophet
                    })
                    # Record naive baseline metrics using the same identifiers to ease later grouping/plots.
                    rows.append({
                        "country": label,
                        "param": param,
                        "horizon_days": horizon,
                        "model": "NaivePersistence",
                        **m_naive
                    })

                    # Mark this combination complete on the progress bar.
                    pbar.update(1)

    # Convert accumulated results to a tidy frame, sort for readability, reset index.
    results = pd.DataFrame(rows).sort_values(
        ["country", "param", "horizon_days", "model"]
    ).reset_index(drop=True)
    # Return the consolidated evaluation table to the caller.
    return results

if __name__ == "__main__":
    out_path = Path("forecast_eval_results.csv")
    results_df = evaluate_all()
    pd.set_option("display.float_format", lambda v: f"{v:0.3f}" if isinstance(v, float) and not math.isnan(v) else str(v))

    print("\n=== Summary (first 25 rows) ===")
    print(results_df.head(25))

    results_df.to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path.resolve()}")

# eval_forecast_darts.py
# Run: python eval_forecast_darts.py
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from utils.forecast import fetch_weather_data
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.metrics import mae, rmse, mape, r2_score
from darts.utils.missing_values import fill_missing_values
import torch

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# -----------------------------
# Config
# -----------------------------
COUNTRIES = {
    "Singapore": "singapore",
    "United States": "america", 
    "India": "india",
}

PARAMS = ["precipitation", "temp_max", "temp_min", "windspeed", "soil_moisture"]
HORIZONS = [3, 7, 14]  # days
HISTORY_YEARS = 2.0     # use 2y of history if available
TRAIN_WINDOW_DAYS = 365 # rolling window size for each backtest

# N-BEATS specific parameters - simplified for stability
NBEATS_CONFIG = {
    "input_chunk_length": 14,  # reduced lookback window
    "output_chunk_length": 14, # max forecast horizon
    "num_stacks": 3,           # reduced complexity
    "num_blocks": 2,
    "num_layers": 2,           # reduced layers
    "layer_widths": 64,        # reduced width
    "expansion_coefficient_dim": 5,
    "trend_polynomial_degree": 2,
    "dropout": 0.1,
    "activation": "ReLU",
    "n_epochs": 50,            # reduced epochs for faster training
    "batch_size": 16,          # smaller batch size
    "optimizer_kwargs": {"lr": 1e-3},
    "random_state": 42,
    "pl_trainer_kwargs": {
        "accelerator": "cpu",
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "enable_checkpointing": False,
        "logger": False,
    },
    "save_checkpoints": False,
    "force_reset": True,
}

# -----------------------------
# Darts-compatible metrics
# -----------------------------
def safe_mape(y_true: TimeSeries, y_pred: TimeSeries) -> float:
    """Safe MAPE calculation that handles zero values"""
    try:
        return float(mape(y_true, y_pred))
    except:
        return float("nan")

def safe_r2(y_true: TimeSeries, y_pred: TimeSeries) -> float:
    """Safe R2 calculation"""
    try:
        return float(r2_score(y_true, y_pred))
    except:
        return float("nan")

# -----------------------------
# Backtesting helpers
# -----------------------------
def _fit_nbeats_single_series(train_series: TimeSeries, horizon_days: int) -> NBEATSModel:
    """Fit N-BEATS model to a single time series with robust error handling"""
    # Adjust output_chunk_length if horizon is smaller
    output_chunk_length = min(NBEATS_CONFIG["output_chunk_length"], horizon_days)

    config = NBEATS_CONFIG.copy()
    config["output_chunk_length"] = output_chunk_length

    # Check for NaN values
    values = train_series.values().flatten()
    has_nan_values = np.isnan(values).any()

    # Fill missing values if any
    if has_nan_values:
        train_series = fill_missing_values(train_series, method='linear')
        values = train_series.values().flatten()

    # Check if series has sufficient variation
    if np.std(values) < 1e-6:
        raise ValueError("Time series has insufficient variation")

    # Normalize for stability
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    normalized_values = (values - mean_val) / (std_val + 1e-8)

    # ✅ Use times + values API (works across Darts versions)
    normalized_series = TimeSeries.from_times_and_values(
        times=train_series.time_index,
        values=normalized_values.reshape(-1, 1),
        columns=None  # single component
    )

    model = NBEATSModel(**config)
    model.fit(normalized_series)

    # Store normalization params for inverse transform
    model._mean_val = mean_val
    model._std_val = std_val

    return model


def backtest_nbeats(df: pd.DataFrame, column: str, horizon_days: int, train_window_days: int) -> Dict[str, float]:
    """Backtest N-BEATS model on a single parameter with improved robustness"""
    series_df = df.dropna(subset=[column]).sort_values("ds").reset_index(drop=True)

    min_required_length = train_window_days + horizon_days + NBEATS_CONFIG["input_chunk_length"] + 10
    if len(series_df) < min_required_length:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE%": float("nan"), "R2": float("nan"), "n_tests": 0}

    # Build a dense daily series
    try:
        series_df['ds'] = pd.to_datetime(series_df['ds'])
        series_df = series_df.set_index('ds').sort_index()

        # Create continuous daily index
        full_date_range = pd.date_range(
            start=series_df.index.min(),
            end=series_df.index.max(),
            freq='D'
        )
        series_df = series_df.reindex(full_date_range)

        # Forward/backward fill to avoid gaps
        series_df[column] = series_df[column].ffill().bfill()

        full_series = TimeSeries.from_dataframe(
            series_df.reset_index(),
            time_col="index",
            value_cols=[column],
            freq="D"
        )
    except Exception as e:
        print(f"TimeSeries conversion error: {e}")
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE%": float("nan"), "R2": float("nan"), "n_tests": 0}

    errors = []
    min_cutoff = train_window_days + NBEATS_CONFIG["input_chunk_length"]
    max_cutoff = len(full_series) - horizon_days
    step_size = max(5, (max_cutoff - min_cutoff) // 20)

    for cutoff in range(min_cutoff, max_cutoff, step_size):
        try:
            train_series = full_series[:cutoff]
            test_series = full_series[cutoff:cutoff + horizon_days]

            # Keep only recent window
            if len(train_series) > train_window_days:
                train_series = train_series[-train_window_days:]

            if len(train_series) < NBEATS_CONFIG["input_chunk_length"] + 10:
                continue

            model = _fit_nbeats_single_series(train_series, horizon_days)
            pred_series = model.predict(n=horizon_days)

            # ✅ Inverse-transform using times + values API
            pred_values = pred_series.values().flatten()
            pred_values = pred_values * model._std_val + model._mean_val
            pred_series = TimeSeries.from_times_and_values(
                times=pred_series.time_index,
                values=pred_values.reshape(-1, 1)
            )

            # Metrics
            mae_val = float(mae(test_series, pred_series))
            rmse_val = float(rmse(test_series, pred_series))
            mape_val = safe_mape(test_series, pred_series)
            r2_val = safe_r2(test_series, pred_series)

            if not (np.isfinite(mae_val) and np.isfinite(rmse_val)):
                continue

            errors.append((mae_val, rmse_val, mape_val, r2_val))

        except Exception as e:
            if len(errors) == 0:
                print(f"N-BEATS training error for {column}: {str(e)[:100]}")
            continue

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
    """Backtest naive persistence model (same as original)"""
    series = df.dropna(subset=[column]).sort_values("ds").reset_index(drop=True)
    if len(series) < train_window_days + horizon_days + 1:
        return {"MAE": float("nan"), "RMSE": float("nan"), "MAPE%": float("nan"), "R2": float("nan"), "n_tests": 0}

    errors = []
    for cutoff in range(train_window_days, len(series) - horizon_days, 10):  # Step by 10
        last_obs = series.iloc[cutoff - 1][column]
        y_pred = np.array([last_obs] * horizon_days)
        y_true = series.iloc[cutoff:cutoff + horizon_days][column].values
        
        # Calculate metrics manually for consistency
        mae_val = float(np.mean(np.abs(y_true - y_pred)))
        rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        
        # MAPE
        mask = np.abs(y_true) > 1e-9
        mape_val = float(np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100.0) if np.any(mask) else float("nan")
        
        # R2
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2_val = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        
        errors.append((mae_val, rmse_val, mape_val, r2_val))

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

# -----------------------------
# Runner with progress bars
# -----------------------------
def evaluate_all() -> pd.DataFrame:
    rows = []
    total_iterations = len(COUNTRIES) * len(HORIZONS) * len(PARAMS)
    
    with tqdm(total=total_iterations, desc="Overall Progress", ncols=100) as pbar:
        for label, key in COUNTRIES.items():
            print(f"\nProcessing {label}...")
            df = fetch_weather_data(country=key, history_years=HISTORY_YEARS, include_soil_moisture=True)

            for horizon in HORIZONS:
                for param in PARAMS:
                    pbar.set_description(f"{label} - {param} - {horizon}d")
                    
                    # Skip if all NaN
                    if df[param].isna().all():
                        rows.append({
                            "country": label,
                            "param": param,
                            "horizon_days": horizon,
                            "model": "N-BEATS",
                            "MAE": float("nan"), "RMSE": float("nan"),
                            "MAPE%": float("nan"), "R2": float("nan"), "n_tests": 0
                        })
                        rows.append({
                            "country": label,
                            "param": param,
                            "horizon_days": horizon,
                            "model": "NaivePersistence",
                            "MAE": float("nan"), "RMSE": float("nan"),
                            "MAPE%": float("nan"), "R2": float("nan"), "n_tests": 0
                        })
                        pbar.update(1)
                        continue

                    # Run N-BEATS model
                    try:
                        m_nbeats = backtest_nbeats(df, param, horizon_days=horizon, train_window_days=TRAIN_WINDOW_DAYS)
                    except Exception as e:
                        print(f"Error with N-BEATS for {label}-{param}-{horizon}d: {e}")
                        m_nbeats = {"MAE": float("nan"), "RMSE": float("nan"), "MAPE%": float("nan"), "R2": float("nan"), "n_tests": 0}
                    
                    # Run naive baseline
                    m_naive = backtest_naive_persistence(df, param, horizon_days=horizon, train_window_days=TRAIN_WINDOW_DAYS)

                    rows.append({"country": label, "param": param, "horizon_days": horizon, "model": "N-BEATS", **m_nbeats})
                    rows.append({"country": label, "param": param, "horizon_days": horizon, "model": "NaivePersistence", **m_naive})

                    pbar.update(1)
    
    results = pd.DataFrame(rows).sort_values(["country", "param", "horizon_days", "model"]).reset_index(drop=True)
    return results

def print_summary_table(results_df: pd.DataFrame):
    """Print a nice summary table comparing models"""
    print("\n=== Model Comparison Summary ===")
    
    summary = results_df.groupby(['model', 'horizon_days']).agg({
        'MAE': 'mean',
        'RMSE': 'mean', 
        'MAPE%': 'mean',
        'R2': 'mean',
        'n_tests': 'sum'
    }).round(3)
    
    print(summary)
    
    # Best model per horizon
    print("\n=== Best Model by Horizon (lowest MAE) ===")
    for horizon in HORIZONS:
        horizon_data = results_df[results_df['horizon_days'] == horizon]
        best_mae = horizon_data.groupby('model')['MAE'].mean().idxmin()
        best_mae_val = horizon_data.groupby('model')['MAE'].mean().min()
        print(f"{horizon}-day horizon: {best_mae} (MAE: {best_mae_val:.3f})")

if __name__ == "__main__":
    out_path = Path("forecast_eval_results_nbeats.csv")
    
    print("Starting N-BEATS weather forecast evaluation...")
    print(f"Countries: {list(COUNTRIES.keys())}")
    print(f"Parameters: {PARAMS}")
    print(f"Horizons: {HORIZONS} days")
    print(f"N-BEATS config: input_length={NBEATS_CONFIG['input_chunk_length']}, epochs={NBEATS_CONFIG['n_epochs']}")
    
    results_df = evaluate_all()
    
    # Set display options
    pd.set_option("display.float_format", lambda v: f"{v:0.3f}" if isinstance(v, float) and not math.isnan(v) else str(v))
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    print_summary_table(results_df)

    print(f"\n=== Detailed Results (first 25 rows) ===")
    print(results_df.head(25))

    results_df.to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path.resolve()}")
    
    # Print some final stats
    total_tests = results_df['n_tests'].sum()
    successful_tests = results_df[results_df['n_tests'] > 0].shape[0]
    print(f"\nTotal backtests run: {total_tests}")
    print(f"Successful model evaluations: {successful_tests}/{len(results_df)}")
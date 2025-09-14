import os
import json
import requests
import pandas as pd
import numpy as np
from functools import lru_cache
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

# Quieter Lightning output (useful in Streamlit)
os.environ.setdefault("PYTORCH_LIGHTNING_DISABLE_PROGRESS_BAR", "1")

from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST  # noqa: F401

#  Debug helpers 
import traceback
DEBUG = os.getenv("PATCHTST_DEBUG", "0") in ("1", "true", "True", "yes")
def _dprint(*args, **kwargs):
    if DEBUG:
        print("[PATCHTST-DBG]", *args, **kwargs)

# Parameters for forecast 
PARAMS: List[str] = [
    "temp_max",
    "temp_min",
    "precipitation",
    "windspeed",
    "soil_moisture",
]

PARAMS_DICT: Dict[str, str] = {
    "temp_max": "Maximum Temperature (°C)",
    "temp_min": "Minimum Temperature (°C)",
    "precipitation": "Precipitation (mm)",
    "windspeed": "Wind Speed (m/s)",
    "soil_moisture": "Soil Moisture (%)",
}

# All file paths 
MODELS_BASE_DIR = "./artifacts/patchtst"
MANIFEST_PATH = os.path.join(MODELS_BASE_DIR, "MANIFEST.json")

# Legacy Pickle Shims
def _shim_neuralforecast_missing_symbols():
    try:
        import neuralforecast.losses.pytorch as nfl
        if not hasattr(nfl, "student_domain_map") or not callable(getattr(nfl, "student_domain_map", None)):
            def _noop_student_domain_map(*args, **kwargs):
                return {}
            nfl.student_domain_map = _noop_student_domain_map
        if hasattr(nfl, "DistributionLoss") and not hasattr(nfl.DistributionLoss, "has_predicted"):
            nfl.DistributionLoss.has_predicted = False
    except Exception:
        pass

def _shim_pl_trainer_kwargs():
    try:
        import pytorch_lightning as pl
    except Exception:
        return
    Trainer = pl.Trainer
    if hasattr(Trainer.__init__, "_ags_shimmed"):
        return
    deprecated_keys = {
        "num_workers_loader","reload_dataloaders_every_n_epochs","progress_bar_refresh_rate",
        "weights_summary","gpus","amp_backend","auto_lr_find","auto_scale_batch_size","terminate_on_nan",
    }
    orig_init = Trainer.__init__
    def patched_init(self, *args, **kwargs):
        for k in list(kwargs.keys()):
            if k in deprecated_keys:
                kwargs.pop(k, None)
        tk = kwargs.get("trainer_kwargs")
        if isinstance(tk, dict):
            for k in list(tk.keys()):
                if k in deprecated_keys:
                    tk.pop(k, None)
        return orig_init(self, *args, **kwargs)
    patched_init._ags_shimmed = True
    Trainer.__init__ = patched_init

def _shim_nf_tsloader_kwargs():
    try:
        import neuralforecast.tsdataset as nfts
        import inspect
        from torch.utils.data import DataLoader as TorchDL
        TLS = getattr(nfts, "TimeSeriesLoader", None)
        if TLS is None or getattr(TLS, "_ags_shimmed", False):
            return
        orig_init = TLS.__init__
        dl_sig = inspect.signature(TorchDL.__init__)
        allowed = set(dl_sig.parameters.keys())
        def patched_init(self, *args, **kwargs):
            scrubbed = {}
            for k, v in kwargs.items():
                if k == "h":
                    continue
                if k in allowed:
                    scrubbed[k] = v
            return orig_init(self, *args, **scrubbed)
        TLS.__init__ = patched_init
        TLS._ags_shimmed = True
    except Exception:
        pass

def _shim_nf_student_scale_decouple():
    try:
        import neuralforecast.losses.pytorch as nfl
        import torch
        def _coerce_like(ref, x, default=None):
            if x is None:
                x = default
            if isinstance(ref, torch.Tensor):
                if isinstance(x, torch.Tensor):
                    return x.to(dtype=ref.dtype, device=ref.device)
                return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)
            return torch.as_tensor(x)
        def _safe_student_scale_decouple(*args, **kwargs):
            if ("loc" in kwargs) or ("scale" in kwargs) or ("df" in kwargs) or ("tscale" in kwargs):
                loc   = kwargs.get("loc",   None)
                scale = kwargs.get("scale", None)
                df    = kwargs.get("df",    kwargs.get("tscale", None))
                if loc is not None and scale is None:
                    try: scale = torch.ones_like(loc)
                    except Exception: scale = 1.0
                loc_t   = _coerce_like(loc,   loc,   default=0.0)
                scale_t = _coerce_like(loc_t, scale, default=1.0)
                df_t    = _coerce_like(loc_t, df,    default=3.0)
                if hasattr(scale_t, "clamp_min"):
                    scale_t = scale_t.clamp_min(1e-6)
                return df_t, loc_t, scale_t
            output = args[0] if len(args) else None
            if isinstance(output, (list, tuple)):
                if len(output) >= 3:
                    df, loc, scale = output[0], output[1], output[2]
                elif len(output) == 2:
                    loc, scale = output
                    df = None
                elif len(output) == 1:
                    output = output[0]
                    df, loc, scale = None, output, None
                else:
                    df, loc, scale = None, None, None
            else:
                df, loc, scale = None, output, None
            loc_t   = _coerce_like(loc,   loc,   default=0.0)
            scale_t = _coerce_like(loc_t, scale, default=1.0)
            df_t    = _coerce_like(loc_t, df,    default=3.0)
            if hasattr(scale_t, "clamp_min"):
                scale_t = scale_t.clamp_min(1e-6)
            return df_t, loc_t, scale_t
        nfl.student_scale_decouple = _safe_student_scale_decouple
    except Exception:
        pass

#  Geocoding helper 
def get_coordinates(location_name: str, count: int = 1, language: str = "en") -> Optional[Tuple[float, float]]:
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": location_name, "count": count, "language": language}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "results" in data and data["results"]:
            result = data["results"][0]
            return float(result["latitude"]), float(result["longitude"])
        return None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None

#  Data fetcher (Open-Meteo ERA5) 
def fetch_weather_data(country: str, history_years: float = 3.5, include_soil_moisture: bool = True, latitude: float = None, longitude: float = None) -> pd.DataFrame:
    import pathlib, json
    OFFLINE = os.getenv("OFFLINE", "0") in ("1", "true", "True", "yes")
    CACHE_DIR = pathlib.Path("./data/cache/open_meteo")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    COUNTRY_LL = {
        "singapore": (1.3521, 103.8198),
        "america":   (41.8780, -93.0977),
        "india":     (30.7333, 76.7794),
    }
    key = (country or "").strip().lower()

    if latitude is not None and longitude is not None:
        lat, lon = latitude, longitude
        key = f"custom_{lat:.2f}_{lon:.2f}"
    elif key in COUNTRY_LL:
        lat, lon = COUNTRY_LL[key]
    else:
        raise ValueError(f"Unknown country key '{country}' or missing coordinates.")

    cache_path = CACHE_DIR / f"{key}.json"

    if OFFLINE and cache_path.exists():
        with open(cache_path, "r") as f:
            data = json.load(f)
    else:
        end_dt = date.today()
        start_dt = end_dt - timedelta(days=int(history_years * 365.25))
        daily_vars = [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
        ]
        sm_var = "soil_moisture_0_to_7cm_mean"
        if include_soil_moisture:
            daily_vars.append(sm_var)

        url = "https://archive-api.open-meteo.com/v1/era5"
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": start_dt.isoformat(), "end_date": end_dt.isoformat(),
            "daily": ",".join(daily_vars), "timezone": "UTC",
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            with open(cache_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            if cache_path.exists():
                with open(cache_path, "r") as f:
                    data = json.load(f)
            else:
                raise RuntimeError(f"Failed to fetch Open-Meteo data and no cache present: {e}")

    if "daily" not in data or "time" not in data["daily"]:
        raise RuntimeError("Open-Meteo response missing 'daily' data")

    daily = data["daily"]
    n = len(daily["time"])
    if n == 0:
        raise RuntimeError("Open-Meteo returned no daily records")

    df = pd.DataFrame({
        "ds": pd.to_datetime(daily["time"]),
        "temp_max": daily.get("temperature_2m_max", [np.nan] * n),
        "temp_min": daily.get("temperature_2m_min", [np.nan] * n),
        "precipitation": daily.get("precipitation_sum", [np.nan] * n),
        "windspeed": daily.get("wind_speed_10m_max", [np.nan] * n),
    })
    if include_soil_moisture:
        df["soil_moisture"] = daily.get("soil_moisture_0_to_7cm_mean", [np.nan] * n)

    df = df.sort_values("ds").reset_index(drop=True)
    full_range = pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")
    df = df.set_index("ds").reindex(full_range)
    df.index.name = "ds"
    df = df.ffill().bfill().reset_index()
    df.attrs["start"] = str(df["ds"].min().date())
    df.attrs["end"] = str(df["ds"].max().date())
    return df

#  Pretrained PatchTST loader
class PretrainedPatchTSTPredictor:
    def __init__(self, models_dir: str = MODELS_BASE_DIR):
        self.models_dir = models_dir
        self.manifest_path = os.path.join(models_dir, "MANIFEST.json")
        self.manifest = self._load_manifest()
        self._cache: Dict[str, Dict] = {}

    def _load_manifest(self) -> dict:
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Manifest not found at {self.manifest_path}")
        with open(self.manifest_path, "r") as f:
            return json.load(f)

    def _candidate_bases(self) -> List[str]:
        bases = [self.models_dir]
        extra = os.environ.get("PATCHTST_BASES", "")
        for b in [p.strip() for p in extra.split(",") if p.strip()]:
            if b not in bases:
                bases.append(b)
        known_old = ["/content/drive/MyDrive/Models/patchtst", "/content/patchtst"]
        for kb in known_old:
            if kb not in bases:
                bases.append(kb)
        return bases

    def _rebase_path(self, stored_path: str) -> str:
        if os.path.exists(stored_path):
            return stored_path
        if not os.path.isabs(stored_path):
            guess = os.path.join(self.models_dir, stored_path)
            if os.path.exists(guess):
                return guess
        parts = os.path.normpath(stored_path).split(os.sep)
        countries = {"america", "india", "singapore"}
        start_idx = 0
        for i, seg in enumerate(parts):
            if seg in countries:
                start_idx = i
                break
        tail = os.path.join(*parts[start_idx:]) if start_idx < len(parts) else stored_path
        for base in self._candidate_bases():
            guess = os.path.join(base, tail)
            if os.path.exists(guess):
                return guess
        return os.path.join(self.models_dir, tail)

    def _get_model_path(self, country_key: str, param: str, horizon_days: int) -> str:
        try:
            stored = self.manifest[country_key][param][str(horizon_days)]
        except KeyError:
            available = []
            for c in self.manifest:
                for p in self.manifest[c]:
                    for h in self.manifest[c][p]:
                        available.append(f"{c}/{p}/h{h}")
            raise ValueError(
                f"No model found for {country_key}/{param}/h{horizon_days}. "
                f"Available: {', '.join(available)}"
            )
        return self._rebase_path(stored)

    def _post_load_shims(self, nf):
        try:
            _shim_neuralforecast_missing_symbols()
            _shim_nf_student_scale_decouple()
        except Exception:
            pass
        try:
            import neuralforecast.losses.pytorch as nfl
            for m in getattr(nf, "models", []):
                loss = getattr(m, "loss", None)
                if loss is None:
                    continue
                if not hasattr(loss, "has_predicted"):
                    setattr(loss, "has_predicted", False)
                if not hasattr(loss, "student_domain_map") or not callable(getattr(loss, "student_domain_map")):
                    setattr(loss, "student_domain_map", lambda *a, **k: {})
                try:
                    loss.scale_decouple = nfl.student_scale_decouple
                except Exception:
                    pass
        except Exception:
            pass

    def _load_model(self, country_key: str, param: str, horizon_days: int):
        cache_key = f"{country_key}_{param}_{horizon_days}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        model_path = self._get_model_path(country_key, param, horizon_days)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        _shim_neuralforecast_missing_symbols()
        _shim_pl_trainer_kwargs()
        _shim_nf_tsloader_kwargs()
        _shim_nf_student_scale_decouple()

        nf = NeuralForecast.load(model_path)
        _dprint(f"Loaded model at {model_path}")
        self._post_load_shims(nf)

        meta_path = os.path.join(model_path, "meta.json")
        metadata = {}
        trained_uid = None
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    metadata = json.load(f)
                trained_uid = (
                    metadata.get("unique_id")
                    or metadata.get("uid")
                    or (metadata.get("dataset", {}) if isinstance(metadata.get("dataset", {}), dict) else {}).get("unique_id")
                    or (metadata.get("data", {}) if isinstance(metadata.get("data", {}), dict) else {}).get("unique_id")
                )
            except Exception as e:
                _dprint("Failed to read meta.json:", repr(e))

        # try to infer input_size
        model = nf.models[0] if getattr(nf, "models", []) else None
        input_size = getattr(model, "input_size", metadata.get("input_size", 365))

        info = {
            "forecaster": nf,
            "metadata": metadata,
            "input_size": input_size,
            "trained_unique_id": trained_uid,
        }
        self._cache[cache_key] = info
        return info

    def _prepare_data(self, raw_data: pd.DataFrame, param: str, input_size: int, horizon_days: int = None) -> pd.DataFrame:
        if param not in raw_data.columns:
            raise ValueError(f"Parameter '{param}' not found in data")

        df = raw_data[["ds", param]].dropna().copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.set_index("ds").sort_index()

        full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
        df = df.reindex(full_range)
        df[param] = df[param].interpolate(method='linear', limit_direction='both')
        df[param] = df[param].ffill().bfill()

        df = df.reset_index().rename(columns={"index": "ds", param: "y"})
        df["unique_id"] = "series1"

        margin = max(60, (horizon_days or 7) * 2)
        need = input_size + margin
        if len(df) < need:
            raise ValueError(f"Insufficient history: have {len(df)} rows, require >= {need}")
        df = df.iloc[-need:].copy()

        return df

    def predict(self, country_key: str, param: str, horizon_days: int, raw_data: Optional[pd.DataFrame] = None) -> dict:
        info = self._load_model(country_key, param, horizon_days)
        nf = info["forecaster"]
        input_size = info["input_size"]

        if raw_data is None:
            if country_key.startswith("custom_"):
                _, lat_str, lon_str = country_key.split("_")
                lat, lon = float(lat_str), float(lon_str)
                raw_data = fetch_weather_data(country=country_key, history_years=3.5, include_soil_moisture=True, latitude=lat, longitude=lon)
            else:
                raw_data = fetch_weather_data(country=country_key, history_years=3.5, include_soil_moisture=True)

        df = self._prepare_data(raw_data, param, input_size, horizon_days)

        # predict
        forecast = nf.predict(df=df, h=horizon_days)
        if not isinstance(forecast, pd.DataFrame) or forecast.empty:
            raise ValueError("NeuralForecast returned no predictions")

        yhat = None
        for col_name in forecast.columns:
            if "PatchTST" in col_name:
                yhat = forecast[col_name].values
                break
        if yhat is None:
            raise ValueError(f"No PatchTST predictions in output columns: {forecast.columns.tolist()}")

        last_date = pd.to_datetime(df["ds"].iloc[-1])
        forecast_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(len(yhat))]

        # a bit of recent context (optional to show)
        hist_days = min(horizon_days * 2, 30)
        historical_df = df[df["ds"] >= (last_date - pd.Timedelta(days=hist_days))].copy()
        historical_df = historical_df.rename(columns={"y": "yhat"})
        historical_dates = historical_df["ds"].tolist()
        historical_values = historical_df["yhat"].tolist()

        return {
            "predictions": yhat.tolist(),
            "dates": [d.strftime("%Y-%m-%d %H:%M:%S") for d in forecast_dates],
            "metadata": info["metadata"],
            "parameter": param,
            "country": country_key,
            "horizon_days": horizon_days,
            "historical_dates": historical_dates,
            "historical_values": historical_values,
        }

    def get_available_models(self) -> List[Dict]:
        out = []
        for c in self.manifest:
            for p in self.manifest[c]:
                for h in self.manifest[c][p]:
                    stored = self.manifest[c][p][h]
                    model_path = self._rebase_path(stored)
                    meta_path = os.path.join(model_path, "meta.json")
                    metrics, saved_at = {}, "unknown"
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, "r") as f:
                                meta = json.load(f)
                            metrics = meta.get("backtest_metrics", {})
                            saved_at = meta.get("saved_at", "unknown")
                        except Exception:
                            pass
                    out.append({
                        "country": c, "param": p, "horizon_days": int(h),
                        "metrics": metrics, "trained_at": saved_at,
                    })
        return out

#  Cached accessor (rerun-safe) 
@lru_cache(maxsize=1)
def get_predictor() -> PretrainedPatchTSTPredictor:
    return PretrainedPatchTSTPredictor()

# Convenience API used by app.py 
def forecast_with_pretrained_patchtst(country, param, horizon_days, raw_data=None):
    country_map = {
        "Singapore": "singapore",
        "United States": "america",
        "India": "india",
        "singapore": "singapore",
        "america": "america",
        "india": "india",
    }
    country_key = country_map.get(country, (country or "").lower())
    predictor = get_predictor()
    return predictor.predict(country_key, param, horizon_days, raw_data)

# Post Processing helpers
def rankdata(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a)) + 1

def quantile_map(forecast: np.ndarray, historical: np.ndarray) -> np.ndarray:
    hist = np.asarray(historical, dtype=float)
    fcst = np.asarray(forecast, dtype=float)
    hist = hist[~np.isnan(hist)]
    fcst = fcst[~np.isnan(fcst)]
    if hist.size < 30 or fcst.size == 0:
        return forecast
    r = rankdata(fcst)
    pct = (r - 0.5) / fcst.size
    mapped = np.percentile(hist, pct * 100)
    return mapped

def _ema(vals, alpha=0.25):
    v = np.array(vals, dtype=float)
    if v.size <= 1:
        return v
    out = np.empty_like(v)
    out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = alpha * v[i] + (1 - alpha) * out[i-1]
    return out

def _limit_step(vals, max_step):
    v = np.array(vals, dtype=float)
    if v.size <= 1:
        return v
    for i in range(1, len(v)):
        delta = v[i] - v[i-1]
        if delta > max_step:   v[i] = v[i-1] + max_step
        if delta < -max_step:  v[i] = v[i-1] - max_step
    return v

def _recent_stats(series, days=60):
    h = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    h = h.tail(days) if len(h) > days else h
    mu = float(h.mean()) if len(h) else 0.0
    sd = float(max(h.std(ddof=0), 1e-6)) if len(h) else 1.0
    return h, mu, sd

def _is_flat_or_low_var(preds, hist, param) -> bool:
    """Trigger only when model output is nearly flat / too low variance."""
    p = np.asarray(preds, dtype=float)
    h = pd.to_numeric(pd.Series(hist), errors="coerce").dropna().tail(60).to_numpy(dtype=float)
    if p.size < 3 or h.size < 10:
        return False
    std_p = np.std(p)
    std_h = np.std(h)
    cv_ratio = std_p / (std_h + 1e-6)

    uniq_frac = len(np.unique(np.round(p, 3))) / p.size

    thresholds = {
        "temp_max": 0.40,   
        "temp_min": 0.45,
        "windspeed": 0.35,
        "soil_moisture": 0.30,
    }
    uniq_min = {
        "temp_max": 0.5,
        "temp_min": 0.5,
        "windspeed": 0.5,
        "soil_moisture": 0.6,
    }
    return (cv_ratio < thresholds.get(param, 0.40)) or (uniq_frac < uniq_min.get(param, 0.5))

def refine_forecast(
    preds: List[float],
    hist_series: pd.Series,
    param: str,
    horizon_days: int,
    context: Optional[dict] = None,
) -> List[float]:
    """
    History-calibrated refinement that ONLY kicks in when predictions are flat/low variance.
    Parameter-specific intensity & constraints.
    """
    p = np.array(pd.to_numeric(pd.Series(preds), errors="coerce"), dtype=float)
    if p.size == 0:
        return preds

    # If not flat/low variance, keep PatchTST shape (do nothing).
    if not _is_flat_or_low_var(p, hist_series, param):
        return p.tolist()

    # --- recent history stats
    h_recent, mu_h, sd_h = _recent_stats(hist_series, days=60)
    if h_recent.size == 0:
        return p.tolist()

    #  base: align mean/std to recent history (bias & scale)
    mu_p = float(np.mean(p))
    sd_p = float(max(np.std(p), 1e-6))
    p_cal = (p - mu_p) * (sd_h / sd_p) + mu_h

    # parameter-specific anomaly injection & noise intensity
    cfg = {
        # min temp often biased low → a tad more anomaly, smaller steps, tighter lower clamp
        "temp_max":     {"anom_len": 14, "anom_scale": 0.55, "ar_rho": 0.55, "ar_scale": 0.35, "step": 2.5, "clip_slack": 0.12},
        "temp_min":     {"anom_len": 14, "anom_scale": 0.60, "ar_rho": 0.55, "ar_scale": 0.30, "step": 1.6, "clip_slack": 0.08},
        # wind is too high → shrink variance, smaller steps, much tighter upper clamp
        "windspeed":    {"anom_len": 10, "anom_scale": 0.45, "ar_rho": 0.65, "ar_scale": 0.40, "step": 2.0, "clip_slack": 0.10},
        # unchanged
        "soil_moisture":{"anom_len": 10, "anom_scale": 0.20, "ar_rho": 0.50, "ar_scale": 0.10, "step": 0.04, "clip_slack": 0.08},
    }[param]


    # recent anomalies pattern
    h_vals = h_recent.to_numpy(dtype=float)
    anom_base = h_vals - _ema(h_vals, alpha=0.25)
    anom = anom_base[-cfg["anom_len"]:] if anom_base.size >= cfg["anom_len"] else anom_base
    if anom.size == 0:
        anom = np.zeros(cfg["anom_len"])
    # tile anomalies across horizon (with decay)
    anom_tiled = np.resize(anom, p.size)
    decay = np.linspace(1.0, 0.6, p.size)
    anom_inject = cfg["anom_scale"] * anom_tiled * decay

    # AR(1) noise sized to history
    z = np.random.normal(0.0, sd_h * cfg["ar_scale"], size=p.size)
    noise = np.zeros(p.size)
    noise[0] = z[0]
    for i in range(1, p.size):
        noise[i] = cfg["ar_rho"] * noise[i-1] + np.sqrt(max(1e-6, 1 - cfg["ar_rho"]**2)) * z[i]

    refined = 0.70 * p_cal + 0.20 * (mu_h + anom_inject) + 0.10 * noise

    # cross-effects (lightweight)
    if context:
        precip = np.asarray(context.get("precip", []), dtype=float)
        if precip.size >= refined.size:
            if param == "temp_max":
                refined -= np.clip(0.04 * precip, 0.0, 1.2)
            elif param == "temp_min":
                refined += np.clip(0.03 * precip, 0.0, 0.9)
            elif param == "windspeed":
                refined += np.clip(0.02 * precip, 0.0, 0.6)
            elif param == "soil_moisture":
            
                alpha = 0.0018  # fraction per mm
                evap_base = 0.006  # daily evaporation fraction
                add = np.clip(alpha * precip[:refined.size], 0.0, 0.04)
                sub = evap_base * np.ones(refined.size)
                sm = np.empty_like(refined)
                # start from last history value
                sm0 = float(h_recent.iloc[-1])
                sm[0] = sm0 + add[0] - sub[0]
                for i in range(1, refined.size):
                    sm[i] = sm[i-1] + add[i] - sub[i]
                refined = 0.6 * refined + 0.4 * sm

    # day-to-day step cap
    refined = _limit_step(refined, cfg["step"])
    refined = _ema(refined, alpha=0.25)

    # quantile-map to historical distribution (keeps realism)
    refined = quantile_map(refined, h_recent.tail(365).values)

    # final envelope / physical constraints
    q05, q95 = float(np.quantile(h_recent, 0.05)), float(np.quantile(h_recent, 0.95))
    span = max(q95 - q05, 1e-6)
    lower = q05 - cfg["clip_slack"] * span
    upper = q95 + cfg["clip_slack"] * span
    refined = np.clip(refined, lower, upper)
    if param == "windspeed":
        refined = np.maximum(refined, 0.0)
    if param == "soil_moisture":
        refined = np.clip(refined, 0.0, 1.0)

    return refined.tolist()

#  wrapper that tries pretrained first 
def enhanced_forecast_function(country, param, days_ahead=7):
    try:
        result = forecast_with_pretrained_patchtst(country, param, days_ahead)
        result["method"] = "pretrained_patchtst"
        return result
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Pre-trained model not available: {e}")
        print("Falling back to your existing forecast method…")
        return None

if __name__ == "__main__":
    try:
        os.environ["PATCHTST_DEBUG"] = "1"
        predictor = get_predictor()
        available = predictor.get_available_models()
        print(f"Found {len(available)} pre-trained models.")
        for m in available[:5]:
            mae = m["metrics"].get("MAE", "N/A")
            print(f"  {m['country']}/{m['param']}/h{m['horizon_days']} (MAE: {mae})")
        if available:
            m = available[0]
            raw = fetch_weather_data(m["country"], history_years=3.5, include_soil_moisture=True)
            res = predictor.predict(m["country"], m["param"], m["horizon_days"], raw)
            print(f"Sample {m['country']}/{m['param']} next 3: {res['predictions'][:3]}")
    except Exception as e:
        print("Error testing predictor:", e)
        traceback.print_exc()

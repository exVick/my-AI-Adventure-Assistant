from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

_OPEN_METEO_URL    = "https://api.open-meteo.com/v1/forecast"
_WIND_THRESHOLD_MS = 7.5
_FORECAST_DAYS     = 14
_REQUEST_TIMEOUT_S = 30   # multi-model responses are larger
_MAX_RETRIES       = 4
_BACKOFF_BASE_S    = 1.0
_BACKOFF_MAX_S     = 20.0
_CACHE_DIR         = os.path.join(os.path.dirname(__file__), ".cache", "open_meteo")

# ── WG-Mix ensemble (7 NWP models, resolution × preference weighted) ──────────
# Weight formula: W = (1 / resolution_km) ^ R_SENS × preference
# Higher resolution and higher preference → larger weight in the blend.
_R_SENS = 1.2
_MODEL_CONFIG: dict[str, dict] = {
    "ecmwf_ifs04":    {"res": 9.0,  "pref": 1.0},
    "gfs_seamless":   {"res": 13.0, "pref": 1.0},
    "icon_seamless":  {"res": 7.0,  "pref": 1.0},
    "icon_global":    {"res": 13.0, "pref": 0.9},
    "gem_seamless":   {"res": 15.0, "pref": 0.7},
    "arome_seamless": {"res": 2.0,  "pref": 1.0},   # sub-regional; NaN outside Europe
    "arpege_europe":  {"res": 11.0, "pref": 1.0},   # sub-regional; NaN outside Europe
}
_BASE_WEIGHTS: dict[str, float] = {
    mod: ((1.0 / cfg["res"]) ** _R_SENS) * cfg["pref"]
    for mod, cfg in _MODEL_CONFIG.items()
}

_HOURLY_VARS = [
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "temperature_2m",
    "precipitation",
    "cloud_cover",
]


# ── Blending helpers ──────────────────────────────────────────────────────────

def _blend_variable(
    df: pd.DataFrame, var_name: str, weights: dict[str, float]
) -> pd.Series:
    """
    Weighted average of a scalar variable across all models, ignoring NaN.
    Column pattern: <var_name>_<model_id>  (Open-Meteo multi-model format).
    """
    cols = [c for c in df.columns if c.startswith(var_name)]
    df_var = df[cols].astype(float)

    col_weights = [weights.get(col.replace(f"{var_name}_", "", 1), 0.0) for col in cols]
    w = pd.Series(col_weights, index=cols)

    valid_w   = df_var.notna() * w
    total_w   = valid_w.sum(axis=1).replace(0, np.nan)
    blended   = (df_var.fillna(0.0) * valid_w).sum(axis=1) / total_w
    return blended


def _blend_wind_direction(
    df: pd.DataFrame, var_name: str, weights: dict[str, float]
) -> pd.Series:
    """
    Vector-average wind direction across all models.
    Decomposes each bearing into (U, V) unit vectors, weights them, then
    converts the resultant vector back to degrees — handles 0°/360° wrap
    correctly.
    """
    cols = [c for c in df.columns if c.startswith(var_name)]
    df_var = df[cols].astype(float)

    col_weights = [weights.get(col.replace(f"{var_name}_", "", 1), 0.0) for col in cols]
    w = pd.Series(col_weights, index=cols)

    valid_w = df_var.notna() * w
    total_w = valid_w.sum(axis=1).replace(0, np.nan)

    rad = np.radians(df_var)
    u   = (np.sin(rad).fillna(0.0) * valid_w).sum(axis=1) / total_w
    v   = (np.cos(rad).fillna(0.0) * valid_w).sum(axis=1) / total_w

    return (np.degrees(np.arctan2(u, v)) + 360) % 360


# ── Per-spot forecast fetch ───────────────────────────────────────────────────

def _fetch_spot_forecast(spot: dict) -> tuple[str, dict[str, Any] | None]:
    """
    Fetch a 14-day multi-model ensemble forecast for one spot, blend it with
    the WG-Mix weighting scheme, and return a structured forecast dict.

    Filtering: spots are dropped when the peak BLENDED WIND SPEED (not gusts)
    is below _WIND_THRESHOLD_MS.
    """
    name = spot["name"]
    try:
        cached = False
        data = _load_cached_forecast(spot)
        if data is None:
            resp = _request_forecast(spot)
            if resp is None:
                return name, None
            data = resp.json()
            _save_cached_forecast(spot, data)
        else:
            cached = True

        # ── Build DataFrame ───────────────────────────────────────────────────
        df = pd.DataFrame(data["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)

        # ── WG-Mix blend ──────────────────────────────────────────────────────
        wg = pd.DataFrame(index=df.index)
        wg["wind_ms"]   = _blend_variable(df, "wind_speed_10m",  _BASE_WEIGHTS).round(1)
        wg["gust_ms"]   = _blend_variable(df, "wind_gusts_10m",  _BASE_WEIGHTS).round(1)
        wg["dir_deg"]   = _blend_wind_direction(df, "wind_direction_10m", _BASE_WEIGHTS).round(0)
        wg["temp_c"]    = _blend_variable(df, "temperature_2m",  _BASE_WEIGHTS).round(1)
        wg["precip_mm"] = _blend_variable(df, "precipitation",   _BASE_WEIGHTS).round(2)
        wg["cloud_pct"] = _blend_variable(df, "cloud_cover",     _BASE_WEIGHTS).round(0)

        # ── Threshold check on blended wind speed only ────────────────────────
        peak_ms = float(wg["wind_ms"].max(skipna=True))
        cache_tag = " (cached)" if cached else ""
        if np.isnan(peak_ms) or peak_ms < _WIND_THRESHOLD_MS:
            print(f"  [weather] {name:<35} peak {peak_ms:.1f} m/s — below threshold, skipped.{cache_tag}")
            return name, None

        # ── Viable hours (wind_ms ≥ threshold) ───────────────────────────────
        viable = wg[wg["wind_ms"] >= _WIND_THRESHOLD_MS]

        def _safe_float(val) -> float | None:
            return float(val) if pd.notna(val) else None

        viable_hours = [
            {
                "time":          ts.isoformat(),
                "wind_ms":       _safe_float(row["wind_ms"]),
                "gust_ms":       _safe_float(row["gust_ms"]),
                "direction_deg": _safe_float(row["dir_deg"]),
                "temp_c":        _safe_float(row["temp_c"]),
                "precip_mm":     _safe_float(row["precip_mm"]),
                "cloud_pct":     _safe_float(row["cloud_pct"]),
            }
            for ts, row in viable.iterrows()
        ]

        # ── Summary stats ─────────────────────────────────────────────────────
        # avg_ms and avg_temp are over viable hours only (when the athlete kites).
        # avg_precip and avg_cloud are over the whole window (trip conditions).
        avg_ms   = round(float(viable["wind_ms"].mean()), 1)
        avg_temp = round(float(viable["temp_c"].mean()),  1) if not viable["temp_c"].isna().all() else None
        # precipitation: sum all hourly mm and divide by days → daily average
        avg_precip_per_day = round(float(wg["precip_mm"].sum()) / _FORECAST_DAYS, 2)
        avg_cloud          = round(float(wg["cloud_pct"].mean()), 0) if not wg["cloud_pct"].isna().all() else None

        forecast = {
            "spot_name":             name,
            "latitude":              spot["latitude"],
            "longitude":             spot["longitude"],
            "distance_km":           spot.get("distance_km"),
            "wind_info":             spot.get("wind_info", {}),
            "peak_ms":               round(peak_ms, 1),
            "avg_ms":                avg_ms,
            "avg_temp_c":            avg_temp,
            "avg_precip_per_day_mm": avg_precip_per_day,
            "avg_cloud_pct":         avg_cloud,
            "viable_hours":          viable_hours,
            "unit":                  "m/s",
        }

        print(
            f"  [weather] {name:<35} peak {peak_ms:.1f} m/s "
            f"avg {avg_ms} m/s  {len(viable_hours)} hrs ≥ threshold  "
            f"temp {avg_temp}°C  precip {avg_precip_per_day} mm/day{cache_tag}"
        )
        return name, forecast

    except Exception as exc:
        print(f"  [weather] {name:<35} ERROR: {exc}")
        return name, None


def _request_forecast(spot: dict) -> requests.Response | None:
    params = {
        "latitude":        spot["latitude"],
        "longitude":       spot["longitude"],
        "hourly":          ",".join(_HOURLY_VARS),
        "models":          ",".join(_MODEL_CONFIG.keys()),
        "wind_speed_unit": "ms",
        "timezone":        "auto",
        "forecast_days":   _FORECAST_DAYS,
    }

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = requests.get(
                _OPEN_METEO_URL,
                params=params,
                timeout=_REQUEST_TIMEOUT_S,
            )

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = min(float(retry_after), _BACKOFF_MAX_S)
                else:
                    delay = min(_BACKOFF_BASE_S * (2 ** attempt), _BACKOFF_MAX_S)
                time.sleep(delay + random.random() * 0.5)
                continue

            resp.raise_for_status()
            return resp
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt >= _MAX_RETRIES:
                raise exc
            delay = min(_BACKOFF_BASE_S * (2 ** attempt), _BACKOFF_MAX_S)
            time.sleep(delay + random.random() * 0.5)

    return None


def _cache_key(spot: dict) -> str:
    date_key = datetime.now(timezone.utc).date().isoformat()
    payload = {
        "date": date_key,
        "lat": round(float(spot["latitude"]), 5),
        "lon": round(float(spot["longitude"]), 5),
        "hourly": _HOURLY_VARS,
        "models": list(_MODEL_CONFIG.keys()),
        "forecast_days": _FORECAST_DAYS,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(spot: dict) -> str:
    return os.path.join(_CACHE_DIR, f"{_cache_key(spot)}.json")


def _load_cached_forecast(spot: dict) -> dict[str, Any] | None:
    path = _cache_path(spot)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _save_cached_forecast(spot: dict, data: dict[str, Any]) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = _cache_path(spot)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=True, separators=(",", ":"))
    except Exception:
        pass


# ── Public API ────────────────────────────────────────────────────────────────

def check_wind_forecasts(
    candidate_spots: list[dict],
    wind_threshold_ms: float = _WIND_THRESHOLD_MS,
    max_workers: int = 6,
) -> dict[str, dict[str, Any]]:
    """
    Fetch 14-day WG-Mix ensemble forecasts for all candidate spots in parallel.
    Returns only spots where the peak blended wind speed meets wind_threshold_ms.

    Args:
        candidate_spots:   List of spot dicts (output of filter_spots).
        wind_threshold_ms: Minimum peak blended wind speed to consider a spot
                           viable (default 7.5 m/s ≈ 15 kn).
        max_workers:       Thread-pool size for parallel HTTP calls (default 6).

    Returns:
        Dict keyed by spot name → forecast dict for each viable spot.
    """
    global _WIND_THRESHOLD_MS
    _WIND_THRESHOLD_MS = wind_threshold_ms

    print(
        f"\n[check_wind_forecasts] Querying {len(candidate_spots)} spots "
        f"(threshold: {wind_threshold_ms} m/s, {_FORECAST_DAYS}-day window, "
        f"workers: {max_workers}) ..."
    )

    viable: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_spot_forecast, spot): spot for spot in candidate_spots}
        for future in as_completed(futures):
            name, forecast = future.result()
            if forecast is not None:
                viable[name] = forecast

    print(f"[check_wind_forecasts] {len(viable)}/{len(candidate_spots)} spots viable.\n")
    return viable

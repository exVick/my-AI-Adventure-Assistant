import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_WIND_THRESHOLD_KN = 15.0
_MS_TO_KN = 1.94384          # Open-Meteo returns m/s by default; we convert here
_FORECAST_DAYS = 3
_REQUEST_TIMEOUT_S = 10


def _fetch_spot_forecast(spot: dict) -> tuple[str, dict[str, Any] | None]:
    """
    Fetch hourly wind speed for a single spot from Open-Meteo.

    Returns (spot_name, forecast_dict) where forecast_dict is None if the
    peak wind across the forecast window is below the viability threshold.
    """
    name = spot["name"]
    try:
        resp = requests.get(
            _OPEN_METEO_URL,
            params={
                "latitude":       spot["latitude"],
                "longitude":      spot["longitude"],
                "hourly":         "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
                "forecast_days":  _FORECAST_DAYS,
                "timezone":       "auto",
            },
            timeout=_REQUEST_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        speeds_ms: list[float] = hourly.get("wind_speed_10m", [])
        times: list[str] = hourly.get("time", [])
        directions: list[int] = hourly.get("wind_direction_10m", [])
        gusts_ms: list[float] = hourly.get("wind_gusts_10m", [])

        if not speeds_ms:
            return name, None

        speeds_kn = [round(s * _MS_TO_KN, 1) for s in speeds_ms]
        gusts_kn  = [round(g * _MS_TO_KN, 1) for g in gusts_ms]
        peak_kn   = max(speeds_kn)

        if peak_kn < _WIND_THRESHOLD_KN:
            print(f"  [weather] {name:<35} peak {peak_kn:.1f} kn — below threshold, skipped.")
            return name, None

        # Build per-hour records only for hours that meet the threshold
        viable_hours = [
            {
                "time":          times[i],
                "wind_kn":       speeds_kn[i],
                "gust_kn":       gusts_kn[i] if i < len(gusts_kn) else None,
                "direction_deg": directions[i] if i < len(directions) else None,
            }
            for i, spd in enumerate(speeds_kn)
            if spd >= _WIND_THRESHOLD_KN
        ]

        forecast = {
            "spot_name":    name,
            "latitude":     spot["latitude"],
            "longitude":    spot["longitude"],
            "distance_km":  spot.get("distance_km"),
            "wind_info":    spot.get("wind_info", {}),
            "peak_kn":      peak_kn,
            "avg_kn":       round(sum(speeds_kn) / len(speeds_kn), 1),
            "viable_hours": viable_hours,
            "unit":         "knots",
        }

        print(f"  [weather] {name:<35} peak {peak_kn:.1f} kn — VIABLE ({len(viable_hours)} hrs ≥ threshold).")
        return name, forecast

    except Exception as exc:
        print(f"  [weather] {name:<35} ERROR: {exc}")
        return name, None


def check_wind_forecasts(
    candidate_spots: list[dict],
    wind_threshold_kn: float = _WIND_THRESHOLD_KN,
    max_workers: int = 10,
) -> dict[str, dict[str, Any]]:
    """
    Fetch wind forecasts for all candidate spots in parallel and return only
    those where the peak wind over the next `_FORECAST_DAYS` days is at or
    above `wind_threshold_kn`.

    Args:
        candidate_spots:   List of spot dicts (output of filter_spots).
        wind_threshold_kn: Minimum peak wind to consider a spot viable (default 15 kn).
        max_workers:       Thread-pool size for parallel HTTP calls (default 10).

    Returns:
        Dict keyed by spot name → forecast dict for each viable spot.
    """
    global _WIND_THRESHOLD_KN
    _WIND_THRESHOLD_KN = wind_threshold_kn

    print(
        f"\n[check_wind_forecasts] Querying {len(candidate_spots)} spots "
        f"(threshold: {wind_threshold_kn} kn, workers: {max_workers}) ..."
    )

    viable: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_spot_forecast, spot): spot for spot in candidate_spots}
        for future in as_completed(futures):
            name, forecast = future.result()
            if forecast is not None:
                viable[name] = forecast

    print(
        f"[check_wind_forecasts] {len(viable)}/{len(candidate_spots)} spots viable.\n"
    )
    return viable

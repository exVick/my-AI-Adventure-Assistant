import json
import math
import os
from typing import Optional

#TODO: improve distance measure with travelling times/ maybe diesel prices

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _months_overlap(best_months_str: Optional[str], target_months: list[str]) -> bool:
    """Return True if any target month appears in the spot's best_months string.
    Spots with no best_months data are included (benefit of the doubt)."""
    if not best_months_str:
        return True
    normalised = best_months_str.lower()
    return any(m.lower() in normalised for m in target_months)


def filter_spots(
    spots_json_path: str,
    target_months: list[str],
    max_distance_km: float,
    origin_lat: float,
    origin_lon: float,
    max_results: int = 40,
) -> list[dict]:
    """
    Read the kitespot JSON database, filter by season and proximity, and return
    up to `max_results` candidates sorted by ascending distance from the origin.

    Args:
        spots_json_path:  Absolute path to closeby_kitespots.json (set via SPOTS_JSON_PATH env var).
        target_months:    Month names to match against each spot's best_months field
                          e.g. ["April", "May"].  Null best_months → included automatically.
        max_distance_km:  Hard cutoff — spots beyond this radius are dropped.
        origin_lat/lon:   Trip departure point (default: Sofia, Bulgaria ≈ 42.70, 23.32).
        max_results:      Cap on returned spots (default 40).

    Returns:
        List of spot dicts, each enriched with a ``distance_km`` key, sorted nearest-first.
    """
    path = spots_json_path or os.environ.get("SPOTS_JSON_PATH", "")
    if not path:
        raise ValueError(
            "spots_json_path is empty and SPOTS_JSON_PATH env var is not set."
        )

    with open(path, "r", encoding="utf-8") as fh:
        countries: list[dict] = json.load(fh)

    candidates: list[dict] = []

    for country_entry in countries:
        country_name: str = country_entry["country"]
        for region_name, spots in country_entry["regions"].items():
            for spot in spots:
                lat = spot.get("latitude")
                lon = spot.get("longitude")
                if lat is None or lon is None:
                    continue

                dist_km = _haversine_km(origin_lat, origin_lon, lat, lon)
                if dist_km > max_distance_km:
                    continue

                best_months = spot.get("wind", {}).get("best_months")
                if not _months_overlap(best_months, target_months):
                    continue

                candidates.append({
                    "name":          spot["name"],
                    "slug":          spot.get("slug", ""),
                    "spot_id":       spot.get("spot_id"),
                    "url":           spot.get("url", ""),
                    "latitude":      lat,
                    "longitude":     lon,
                    "country":       country_name,
                    "region":        region_name,
                    "distance_km":   round(dist_km, 1),
                    "wind_info":     spot.get("wind", {}),
                })

    candidates.sort(key=lambda s: s["distance_km"])
    selected = candidates[:max_results]

    print(
        f"[filter_spots] {len(candidates)} spots matched filters "
        f"(months={target_months}, max_dist={max_distance_km} km) → "
        f"returning top {len(selected)}."
    )
    return selected

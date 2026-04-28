import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from state import GraphState

# Define model
model_used = "openai/gpt-oss-120b"

# ---------------------------------------------------------------------------
# Direction-matching helpers  (pure Python — no LLM arithmetic)
# ---------------------------------------------------------------------------

_COMPASS_TO_DEG: dict[str, float] = {
    "N": 0.0,   "NNE": 22.5,  "NE": 45.0,  "ENE": 67.5,
    "E": 90.0,  "ESE": 112.5, "SE": 135.0, "SSE": 157.5,
    "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}


def _angular_diff(a: float, b: float) -> float:
    """Smallest unsigned angle between two bearings (result in [0, 180])."""
    return min(abs(a - b) % 360, 360 - abs(a - b) % 360)


def _direction_match_pct(viable_hours: list[dict], preferred_dirs: list[str]) -> int:
    """
    Percentage of viable wind hours whose direction falls within ±45° of at
    least one preferred compass direction.  Returns 0-100.
    """
    if not viable_hours or not preferred_dirs:
        return 50  # neutral when data is absent

    preferred_degs = [
        _COMPASS_TO_DEG[d.upper().strip()]
        for d in preferred_dirs
        if d.upper().strip() in _COMPASS_TO_DEG
    ]
    if not preferred_degs:
        return 50

    hours_with_data = [h for h in viable_hours if h.get("direction_deg") is not None]
    if not hours_with_data:
        return 50

    matches = sum(
        1 for h in hours_with_data
        if any(_angular_diff(h["direction_deg"], pd) <= 45.0 for pd in preferred_degs)
    )
    return round(matches / len(hours_with_data) * 100)


def _compute_score(
    forecast: dict,
    preferred_dirs: list[str],
    preferred_wind_types: list[str],
    max_distance_km: float,
) -> float:
    """
    Composite score (0-100) for one spot:

      Component          Weight   Logic
      ─────────────────  ──────   ──────────────────────────────────────────
      Wind intensity      35 pts  15 kn → 0, 30 kn+ → 35 (linear)
      Direction match     35 pts  direction_match_pct / 100 × 35
      Session window      20 pts  viable hours capped at 12 h → 20
      Distance proximity  10 pts  1 − (dist / max_dist) × 10
    """
    viable_hours: list[dict] = forecast.get("viable_hours", [])
    peak_kn:     float       = forecast.get("peak_kn", 15.0)
    dist_km:     float       = forecast.get("distance_km") or max_distance_km

    intensity  = min(35.0, max(0.0, (peak_kn - 15.0) / 15.0 * 35.0))
    dir_score  = _direction_match_pct(viable_hours, preferred_dirs) / 100.0 * 35.0
    window     = min(20.0, len(viable_hours) / 12.0 * 20.0)
    proximity  = max(0.0, (1.0 - dist_km / max_distance_km) * 10.0)

    return round(intensity + dir_score + window + proximity, 2)


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class RankedSpot(BaseModel):
    spot_name:           str   = Field(description="Exact spot name from the forecast data.")
    rank:                int   = Field(description="1 = best, 5 = fifth-best.")
    composite_score:     float = Field(description="Pre-computed numeric score (0-100).")
    peak_kn:             float = Field(description="Peak wind speed in knots over the forecast window.")
    avg_kn:              float = Field(description="Average wind speed in knots over the forecast window.")
    viable_hours_count:  int   = Field(description="Number of hours at or above the wind threshold.")
    best_session_window: str   = Field(
        description="ISO datetime range for the best consecutive block of viable hours. "
                    "Format: 'YYYY-MM-DDTHH:MM / YYYY-MM-DDTHH:MM'. Use 'unknown' if indeterminate."
    )
    direction_match_pct: int   = Field(description="Percentage of viable hours in a preferred wind direction (0-100).")
    distance_km:         float = Field(description="Aerial distance from the origin in km.")
    country:             str   = Field(description="Country where the spot is located.")
    url:                 str   = Field(description="Spot detail URL.")
    rationale:           str   = Field(
        description="2-3 sentences explaining the ranking: "
                    "dominant wind regime, quality of session windows, "
                    "and how the thermal/direction profile matches the forecast."
    )


class WindExpertOutput(BaseModel):
    ranked_spots:   list[RankedSpot] = Field(description="Top 5 spots ordered best-first (rank 1-5).")
    expert_summary: str              = Field(
        description="One paragraph overview of the overall forecast quality for this trip window."
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are a world-class kitesurf meteorologist and spot selector.

You will receive a pre-scored table of candidate spots. Each row contains:
  • A numeric composite_score (0-100) computed from: wind intensity, direction
    match percentage, session window length, and distance from the origin.
  • Raw forecast statistics: peak_kn, avg_kn, viable_hours_count,
    direction_match_pct.
  • The spot's known wind character from the database (wind_info).

Your job:
  1. Select the TOP 5 spots from the list.
  2. Use the composite_score as your primary ranking signal, but apply expert
     judgment to break ties or downrank a spot whose wind_info description
     indicates structural problems (e.g. "gusty and unreliable", "onshore only").
  3. Identify the best_session_window: find the longest uninterrupted block of
     viable hours in the provided time series and report it as an ISO range.
  4. Write a concise rationale for each spot.
  5. Provide an expert_summary paragraph covering the overall forecast quality.

Return your answer using the provided response schema — no extra prose.
""".strip()


# ---------------------------------------------------------------------------
# Scoring helper: find best consecutive window
# ---------------------------------------------------------------------------

def _best_window(viable_hours: list[dict]) -> str:
    """Return the ISO range of the longest consecutive block of viable hours."""
    if not viable_hours:
        return "unknown"

    times = [h["time"] for h in viable_hours if h.get("time")]
    if not times:
        return "unknown"

    # viable_hours are already filtered to ≥ threshold; find the longest
    # run of entries whose timestamps are consecutive (hourly cadence).
    best_start = best_end = times[0]
    run_start  = times[0]
    prev       = times[0]

    for t in times[1:]:
        # timestamps look like "2025-05-01T14:00" — check if exactly 1 hour apart
        from datetime import datetime
        try:
            dt_prev = datetime.fromisoformat(prev)
            dt_curr = datetime.fromisoformat(t)
            gap_h   = (dt_curr - dt_prev).total_seconds() / 3600
        except ValueError:
            gap_h = 99

        if gap_h <= 1.5:  # treat ≤1.5h gap as consecutive (accounts for rounding)
            if (datetime.fromisoformat(t) - datetime.fromisoformat(run_start)).total_seconds() > \
               (datetime.fromisoformat(best_end)  - datetime.fromisoformat(best_start)).total_seconds():
                best_start, best_end = run_start, t
        else:
            run_start = t
        prev = t

    return f"{best_start} / {best_end}"


# ---------------------------------------------------------------------------
# Agent function
# ---------------------------------------------------------------------------

def run_wind_expert(state: GraphState) -> dict[str, Any]:
    """
    Agent 2 — Wind Expert.

    Merges candidate_spots metadata with weather_forecasts, computes a numeric
    composite score per spot in pure Python (direction math, window length,
    intensity, proximity), then passes the pre-scored table to Llama 3 70B
    for holistic expert ranking and qualitative rationale.

    Reads from state : candidate_spots, weather_forecasts, trip_parameters
    Writes to state  : ranked_spots, messages
    """
    candidate_spots:   list[dict]       = state.get("candidate_spots", [])
    weather_forecasts: dict[str, dict]  = state.get("weather_forecasts", {})
    trip_params:       dict             = state.get("trip_parameters", {})

    preferred_dirs       = trip_params.get("preferred_directions", [])
    preferred_wind_types = trip_params.get("preferred_wind_types", [])
    max_distance_km      = float(trip_params.get("max_distance_km", 800))

    # Build a name → spot lookup for metadata not present in forecasts
    spot_meta: dict[str, dict] = {s["name"]: s for s in candidate_spots}

    # Score every forecasted (viable) spot
    scored: list[dict] = []
    for name, forecast in weather_forecasts.items():
        meta     = spot_meta.get(name, {})
        score    = _compute_score(forecast, preferred_dirs, preferred_wind_types, max_distance_km)
        dir_pct  = _direction_match_pct(forecast.get("viable_hours", []), preferred_dirs)
        window   = _best_window(forecast.get("viable_hours", []))

        scored.append({
            "spot_name":           name,
            "composite_score":     score,
            "peak_kn":             forecast.get("peak_kn", 0),
            "avg_kn":              forecast.get("avg_kn", 0),
            "viable_hours_count":  len(forecast.get("viable_hours", [])),
            "best_session_window": window,
            "direction_match_pct": dir_pct,
            "distance_km":         forecast.get("distance_km") or meta.get("distance_km", 0),
            "country":             meta.get("country", "unknown"),
            "url":                 meta.get("url", ""),
            "wind_info":           forecast.get("wind_info") or meta.get("wind_info", {}),
        })

    scored.sort(key=lambda x: x["composite_score"], reverse=True)

    if not scored:
        print("[Wind Expert] No viable forecasts to rank.")
        return {"ranked_spots": [], "messages": []}

    # Trim to top-15 before sending to LLM to keep the prompt lean
    top_candidates = scored[:15]

    table_json = json.dumps(top_candidates, indent=2)
    human_text = (
        f"Trip parameters:\n"
        f"  Preferred directions : {preferred_dirs}\n"
        f"  Preferred wind types : {preferred_wind_types}\n"
        f"  Max distance         : {max_distance_km} km\n\n"
        f"Pre-scored candidate spots (top {len(top_candidates)} by composite_score):\n"
        f"{table_json}\n\n"
        "Select and rank the top 5 spots. Return them using the required schema."
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_text),
    ]

    llm = ChatGroq(model=model_used, temperature=0.2)
    structured_llm = llm.with_structured_output(WindExpertOutput)

    print(f"[Wind Expert] Scoring {len(weather_forecasts)} viable spots, ranking top 5 ...")
    output: WindExpertOutput = structured_llm.invoke(messages)

    ranked_spots = [s.model_dump() for s in output.ranked_spots]

    print(f"[Wind Expert] Top 5 ranked:")
    for s in ranked_spots:
        print(f"  #{s['rank']} {s['spot_name']:<35} score={s['composite_score']}")

    return {
        "ranked_spots": ranked_spots,
        "messages": [*messages, output.expert_summary],
    }

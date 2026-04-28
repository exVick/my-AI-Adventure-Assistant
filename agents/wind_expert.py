import json
from collections import defaultdict
from datetime import datetime
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
    Percentage of viable wind hours whose blended direction falls within ±45°
    of at least one preferred compass direction.  Returns 0-100.
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
    Composite score (0-100) over the 16-day forecast window.

      Component            Weight   Logic
      ─────────────────    ──────   ──────────────────────────────────────────────────
      Wind intensity         30 pts  7.5 m/s → 0, ≥15 m/s → 30 (linear; based on peak)
      Direction match        25 pts  direction_match_pct / 100 × 25
      Session window         20 pts  viable hours capped at 48 h total → full score
      Distance proximity     10 pts  1 − (dist / max_dist) × 10
      Temperature bonus       8 pts  ≤15°C → 0, ≥30°C → 8 (linear; over viable hours)
      Weather quality         7 pts  equal blend of low-precipitation and low-cloud scores
    """
    viable_hours: list[dict] = forecast.get("viable_hours", [])
    peak_ms:  float = forecast.get("peak_ms",               7.5)
    dist_km:  float = forecast.get("distance_km")       or  max_distance_km
    avg_temp: float = forecast.get("avg_temp_c")         or  20.0
    avg_prec: float = forecast.get("avg_precip_per_day_mm") or 0.0
    avg_cld:  float = forecast.get("avg_cloud_pct")      or  50.0

    intensity  = min(30.0, max(0.0, (peak_ms - 7.5) / 7.5 * 30.0))
    dir_score  = _direction_match_pct(viable_hours, preferred_dirs) / 100.0 * 25.0
    # 48 viable hours across 16 days earns the full session-window score
    window     = min(20.0, len(viable_hours) / 48.0 * 20.0)
    proximity  = max(0.0, (1.0 - dist_km / max_distance_km) * 10.0)
    temp_score = min(8.0,  max(0.0, (avg_temp - 15.0) / 15.0 * 8.0))
    # precip: 0 mm/day → 1.0, ≥5 mm/day → 0.  cloud: 0% → 1.0, 100% → 0.
    precip_fac = max(0.0, 1.0 - avg_prec / 5.0)
    cloud_fac  = max(0.0, 1.0 - avg_cld / 100.0)
    wx_score   = 7.0 * (0.5 * precip_fac + 0.5 * cloud_fac)

    return round(intensity + dir_score + window + proximity + temp_score + wx_score, 2)


# ---------------------------------------------------------------------------
# Session-window helpers
# ---------------------------------------------------------------------------

def _best_window(viable_hours: list[dict]) -> str:
    """Return the ISO range of the longest consecutive block of viable hours."""
    if not viable_hours:
        return "unknown"

    times = [h["time"] for h in viable_hours if h.get("time")]
    if not times:
        return "unknown"

    best_start = best_end = times[0]
    run_start  = times[0]
    prev       = times[0]

    for t in times[1:]:
        try:
            gap_h = (datetime.fromisoformat(t) - datetime.fromisoformat(prev)).total_seconds() / 3600
        except ValueError:
            gap_h = 99

        if gap_h <= 1.5:
            run_duration  = (datetime.fromisoformat(t)     - datetime.fromisoformat(run_start)).total_seconds()
            best_duration = (datetime.fromisoformat(best_end) - datetime.fromisoformat(best_start)).total_seconds()
            if run_duration > best_duration:
                best_start, best_end = run_start, t
        else:
            run_start = t
        prev = t

    return f"{best_start} / {best_end}"


def _best_day(viable_hours: list[dict]) -> str:
    """Return the ISO date (YYYY-MM-DD) with the highest peak wind in viable hours."""
    if not viable_hours:
        return "unknown"

    day_peak: dict[str, float] = defaultdict(float)
    for h in viable_hours:
        day = h.get("time", "")[:10]
        wind = h.get("wind_ms") or 0.0
        if wind > day_peak[day]:
            day_peak[day] = wind

    return max(day_peak, key=lambda d: day_peak[d]) if day_peak else "unknown"


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class RankedSpot(BaseModel):
    spot_name:               str   = Field(description="Exact spot name from the forecast data.")
    rank:                    int   = Field(description="1 = best, 5 = fifth-best.")
    composite_score:         float = Field(description="Pre-computed numeric score (0-100).")
    peak_ms:                 float = Field(description="Peak blended wind speed in m/s over the 16-day window.")
    avg_ms:                  float = Field(description="Average blended wind speed in m/s during viable hours.")
    viable_hours_count:      int   = Field(description="Total hours at or above the wind threshold across the window.")
    best_session_window:     str   = Field(
        description="ISO range of the longest uninterrupted block of viable hours. "
                    "Format: 'YYYY-MM-DDTHH:MM / YYYY-MM-DDTHH:MM'."
    )
    best_day:                str   = Field(
        description="ISO date (YYYY-MM-DD) of the single best kitesurf day in the 16-day window "
                    "(highest peak wind). Use for a focused day-trip recommendation."
    )
    direction_match_pct:     int   = Field(description="% of viable hours within ±45° of preferred directions (0-100).")
    avg_temp_c:              float = Field(description="Average temperature (°C) during viable kite hours.")
    avg_precip_per_day_mm:   float = Field(description="Average daily precipitation (mm) over the full forecast window.")
    avg_cloud_pct:           float = Field(description="Average cloud cover (%) over the full forecast window.")
    distance_km:             float = Field(description="Aerial distance from the origin in km.")
    country:                 str   = Field(description="Country where the spot is located.")
    url:                     str   = Field(description="Spot detail URL.")
    rationale:               str   = Field(
        description=(
            "2 sentences max: (1) wind strength + direction alignment with wind_info; "
            "(2) weather/temperature verdict and best_day recommendation."
        )
    )


class WindExpertOutput(BaseModel):
    ranked_spots:   list[RankedSpot] = Field(description="Top 5 spots ordered best-first (rank 1-5).")
    expert_summary: str              = Field(
        description="2-3 sentences: dominant wind regime, overall forecast quality, standout pattern."
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are a world-class kitesurf meteorologist and trip planner for the Mediterranean,
Aegean, Adriatic, and Black Sea region.

You will receive a pre-scored table of candidate spots covering the next 16-day window.
The forecast is a WG-Mix ensemble blend of 7 NWP models (ECMWF IFS, GFS, ICON, GEM,
AROME, ARPEGE), weighted by resolution and reliability. All values are in m/s.

Each row contains:
  • composite_score (0-100): Python pre-computed from —
      Wind intensity 30pts | Direction match 25pts | Session window 20pts |
      Distance 10pts | Temperature bonus 8pts | Weather quality 7pts
  • Wind: peak_ms (peak over 16 days), avg_ms (avg during viable hours),
    viable_hours_count (total hours ≥ threshold), direction_match_pct
  • Weather context: avg_temp_c (during kite sessions), avg_precip_per_day_mm
    (daily avg over full window), avg_cloud_pct (overall cloud cover %)
  • Spot database: wind_info with best_direction, wind_type, main_direction, description
  • Pre-computed windows: best_session_window (longest consecutive block), best_day

─────────────────────────────────────────────
RANKING PRIORITIES
─────────────────────────────────────────────

1. WIND STRENGTH (primary):
   Prefer spots with strong sustained ensemble wind (avg_ms ≥ 9 m/s is excellent,
   7.5-9 m/s is marginal). A spot with one exceptional 5-hour window at 12 m/s
   beats a spot with 40 hours of 8 m/s if the athlete wants a committed session.

2. DIRECTION FIT:
   Cross-reference direction_match_pct WITH wind_info.best_direction and
   wind_info.description. A spot whose database profile (e.g. "NE thermal, April-Oct")
   structurally aligns with the forecasted direction is reliable — not just
   coincidentally windy. Spots with "onshore only", "gusty", or "unreliable" in their
   description should be penalised even if the score is high.

3. WEATHER QUALITY:
   • Temperature: avg_temp_c ≥ 22°C is ideal (warm beach day). Below 15°C is
     unpleasant and reduces session duration.
   • Precipitation: avg_precip_per_day_mm ≤ 1 mm is excellent, ≥ 4 mm ruins
     beach days. Penalise heavily above 4 mm even if wind is strong.
   • Cloud cover: avg_cloud_pct ≤ 40% is great, ≥ 80% is oppressive. Weight
     this less than precipitation, but factor it into overall enjoyment.

4. SESSION OPPORTUNITY:
   viable_hours_count spread across the 16-day window matters — more hours = more
   chances to pick a day. Flag best_day as the single best day-trip target.

─────────────────────────────────────────────
RATIONALE (2 sentences per spot — be concise)
─────────────────────────────────────────────
Sentence 1: wind strength + direction alignment with wind_info.
Sentence 2: weather verdict (temp/rain/cloud) + best_day call-out.

Return your answer using the provided response schema — no extra prose.
Output must be valid JSON, ASCII only, no special symbols or non-ASCII spaces.
""".strip()


def _extract_json_blob(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start:end + 1]


def _rule_based_output(scored_rows: list[dict]) -> WindExpertOutput:
    top = scored_rows[:5]
    ranked = []
    for idx, row in enumerate(top, start=1):
        avg_temp = row.get("avg_temp_c")
        avg_prec = row.get("avg_precip_per_day_mm")
        avg_cld = row.get("avg_cloud_pct")
        rationale = (
            f"Peak {row.get('peak_ms', 0)} m/s with avg {row.get('avg_ms', 0)} m/s; "
            f"direction match {row.get('direction_match_pct', 0)}%. "
            f"Temp {avg_temp} C, precip {avg_prec} mm/day, cloud {avg_cld}%; "
            f"best day {row.get('best_day', 'unknown')}."
        )
        ranked.append({
            "spot_name": row.get("spot_name", "unknown"),
            "rank": idx,
            "composite_score": row.get("composite_score", 0.0),
            "peak_ms": row.get("peak_ms", 0.0),
            "avg_ms": row.get("avg_ms", 0.0),
            "viable_hours_count": row.get("viable_hours_count", 0),
            "best_session_window": row.get("best_session_window", "unknown"),
            "best_day": row.get("best_day", "unknown"),
            "direction_match_pct": row.get("direction_match_pct", 0),
            "avg_temp_c": avg_temp or 0.0,
            "avg_precip_per_day_mm": avg_prec or 0.0,
            "avg_cloud_pct": avg_cld or 0.0,
            "distance_km": row.get("distance_km", 0.0),
            "country": row.get("country", "unknown"),
            "url": row.get("url", ""),
            "rationale": rationale,
        })

    summary = "Auto-ranked by composite score due to LLM tool failure."
    return WindExpertOutput(ranked_spots=ranked, expert_summary=summary)


# ---------------------------------------------------------------------------
# Agent function
# ---------------------------------------------------------------------------

def run_wind_expert(state: GraphState) -> dict[str, Any]:
    """
    Agent 2 — Wind Expert.

    Merges candidate_spots metadata with the WG-Mix ensemble weather_forecasts,
    computes a composite score per spot in pure Python (direction math, window
    length, wind intensity, distance, temperature, weather quality), then passes
    the pre-scored table to the LLM for holistic expert ranking, weather
    interpretation, and personalised rationale generation.

    Reads from state : candidate_spots, weather_forecasts, trip_parameters
    Writes to state  : ranked_spots, messages
    """
    candidate_spots:   list[dict]      = state.get("candidate_spots", [])
    weather_forecasts: dict[str, dict] = state.get("weather_forecasts", {})
    trip_params:       dict            = state.get("trip_parameters", {})

    preferred_dirs       = trip_params.get("preferred_directions", [])
    preferred_wind_types = trip_params.get("preferred_wind_types", [])
    max_distance_km      = float(trip_params.get("max_distance_km", 800))

    spot_meta: dict[str, dict] = {s["name"]: s for s in candidate_spots}

    scored: list[dict] = []
    for name, forecast in weather_forecasts.items():
        meta    = spot_meta.get(name, {})
        score   = _compute_score(forecast, preferred_dirs, preferred_wind_types, max_distance_km)
        dir_pct = _direction_match_pct(forecast.get("viable_hours", []), preferred_dirs)
        window  = _best_window(forecast.get("viable_hours", []))
        day     = _best_day(forecast.get("viable_hours", []))

        scored.append({
            "spot_name":             name,
            "composite_score":       score,
            "peak_ms":               forecast.get("peak_ms", 0),
            "avg_ms":                forecast.get("avg_ms", 0),
            "viable_hours_count":    len(forecast.get("viable_hours", [])),
            "best_session_window":   window,
            "best_day":              day,
            "direction_match_pct":   dir_pct,
            "avg_temp_c":            forecast.get("avg_temp_c"),
            "avg_precip_per_day_mm": forecast.get("avg_precip_per_day_mm"),
            "avg_cloud_pct":         forecast.get("avg_cloud_pct"),
            "distance_km":           forecast.get("distance_km") or meta.get("distance_km", 0),
            "country":               meta.get("country", "unknown"),
            "url":                   meta.get("url", ""),
            "wind_info":             forecast.get("wind_info") or meta.get("wind_info", {}),
        })

    scored.sort(key=lambda x: x["composite_score"], reverse=True)

    if not scored:
        print("[Wind Expert] No viable forecasts to rank.")
        return {"ranked_spots": [], "messages": []}

    # Trim wind_info before serialisation: drop best_months (used upstream),
    # cap description at 120 chars to keep the JSON payload small and avoid
    # hitting the model's output-token limit on the structured response.
    def _trim_wind_info(wi: dict) -> dict:
        desc = (wi.get("description") or "")[:120]
        return {
            "wind_type":      wi.get("wind_type"),
            "best_direction": wi.get("best_direction"),
            "main_direction": wi.get("main_direction"),
            "description":    desc,
        }

    top_candidates = [
        {**s, "wind_info": _trim_wind_info(s.get("wind_info") or {})}
        for s in scored[:10]   # top-10 keeps prompt + output well within token limits
    ]

    table_json = json.dumps(top_candidates, ensure_ascii=True, separators=(",", ":"))
    human_text = (
        f"Trip parameters:\n"
        f"  Preferred directions : {preferred_dirs}\n"
        f"  Preferred wind types : {preferred_wind_types}\n"
        f"  Max distance         : {max_distance_km} km\n"
        f"  Forecast window      : 16 days\n\n"
        f"Pre-scored candidate spots (top {len(top_candidates)} by composite_score):\n"
        f"{table_json}\n\n"
        "Select and rank the top 5 spots. Return them using the required schema."
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_text),
    ]

    llm = ChatGroq(model=model_used, temperature=0.2, max_tokens=1200)
    structured_llm = llm.with_structured_output(WindExpertOutput)

    print(f"[Wind Expert] Scoring {len(weather_forecasts)} viable spots, ranking top 5 ...")
    def _invoke_json_fallback(msgs: list) -> WindExpertOutput:
        fallback_msgs = [
            SystemMessage(
                content=(
                    _SYSTEM_PROMPT
                    + "\nReturn ONLY the JSON object for the schema."
                    + " Do not wrap in markdown or add commentary."
                )
            ),
            *[m for m in msgs if isinstance(m, HumanMessage)],
        ]
        raw = llm.invoke(fallback_msgs).content or ""
        if not raw.strip():
            raise ValueError("Empty LLM response in JSON fallback.")
        blob = _extract_json_blob(raw)
        data = json.loads(blob)
        return WindExpertOutput.model_validate(data)

    try:
        output: WindExpertOutput = structured_llm.invoke(messages)
    except Exception as exc:
        msg = str(exc)
        print(f"[Wind Expert] Structured output failed, retrying with smaller payload: {exc}")
        fallback_limit = 5 if "tool choice is required" in msg.lower() or "tool_use_failed" in msg.lower() else 7
        top_candidates = [
            {**s, "wind_info": _trim_wind_info(s.get("wind_info") or {})}
            for s in scored[:fallback_limit]
        ]
        table_json = json.dumps(top_candidates, ensure_ascii=True, separators=(",", ":"))
        human_text = (
            f"Trip parameters:\n"
            f"  Preferred directions : {preferred_dirs}\n"
            f"  Preferred wind types : {preferred_wind_types}\n"
            f"  Max distance         : {max_distance_km} km\n"
            f"  Forecast window      : 16 days\n\n"
            f"Pre-scored candidate spots (top {len(top_candidates)} by composite_score):\n"
            f"{table_json}\n\n"
            "Select and rank the top 5 spots. Return them using the required schema."
        )
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=human_text),
        ]

        if "tool choice is required" in msg.lower() or "tool_use_failed" in msg.lower():
            try:
                output = _invoke_json_fallback(messages)
            except Exception as fallback_exc:
                print(f"[Wind Expert] JSON fallback failed, using rule-based ranking: {fallback_exc}")
                output = _rule_based_output(scored)
        else:
            output = structured_llm.invoke(messages)

    ranked_spots = [s.model_dump() for s in output.ranked_spots]

    print("[Wind Expert] Top 5 ranked:")
    for s in ranked_spots:
        print(
            f"  #{s['rank']} {s['spot_name']:<35} score={s['composite_score']}  "
            f"peak={s['peak_ms']} m/s  temp={s['avg_temp_c']}°C  "
            f"best_day={s['best_day']}"
        )

    return {
        "ranked_spots": ranked_spots,
        "messages": [*messages, output.expert_summary],
    }

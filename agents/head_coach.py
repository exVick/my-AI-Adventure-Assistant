import math
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from state import GraphState

# Define model
model_used = "openai/gpt-oss-120b"

# ---------------------------------------------------------------------------
# Drive-time helper
# ---------------------------------------------------------------------------

_ROAD_FACTOR   = 1.30   # road distance ≈ 1.3× aerial for the Balkans
_AVG_SPEED_KPH = 80.0   # conservative motorway average including stops


def _drive_time_str(distance_km: float | None) -> str:
    """Convert aerial km → estimated driving time string, e.g. '5 h 30 min'."""
    if not distance_km:
        return "unknown"
    road_km  = distance_km * _ROAD_FACTOR
    total_h  = road_km / _AVG_SPEED_KPH
    hours    = int(total_h)
    minutes  = round((total_h - hours) * 60 / 15) * 15   # round to 15-min slots
    if minutes == 60:
        hours  += 1
        minutes = 0
    return f"{hours} h {minutes} min" if minutes else f"{hours} h"


# ---------------------------------------------------------------------------
# Physiological load classifier
# ---------------------------------------------------------------------------

def _classify_load(health: dict) -> dict[str, str]:
    """
    Derive simple qualitative labels from raw Garmin metrics so the LLM
    receives pre-interpreted signals rather than raw numbers.
    """
    bb    = health.get("body_battery", 50)
    sleep = health.get("sleep_score",  50)
    hrv   = health.get("hrv_status",   "balanced").lower()
    mins  = health.get("minutes_of_load_last7days", 0)
    rhr   = health.get("resting_hr_bpm", 55)

    battery_level = "high"    if bb    >= 70 else ("moderate" if bb    >= 40 else "low")
    sleep_quality = "good"    if sleep >= 70 else ("fair"     if sleep >= 50 else "poor")
    hrv_state     = "balanced" if any(k in hrv for k in ("balanced", "optimal", "good")) \
                    else ("poor" if any(k in hrv for k in ("poor", "low", "unbalanced")) else "moderate")
    weekly_load   = "high"    if mins  >= 400 else ("moderate" if mins  >= 200 else "low")
    cardiac_state = "elevated" if rhr  >= 65   else "normal"

    # Overall readiness label
    score = (
        (2 if battery_level == "high"     else 1 if battery_level == "moderate" else 0) +
        (2 if sleep_quality  == "good"    else 1 if sleep_quality  == "fair"    else 0) +
        (2 if hrv_state      == "balanced" else 1 if hrv_state     == "moderate" else 0) +
        (1 if weekly_load    == "low"     else 0 if weekly_load    == "moderate" else -1) +
        (1 if cardiac_state  == "normal"  else 0)
    )
    readiness = "peak" if score >= 7 else ("solid" if score >= 4 else ("tired" if score >= 2 else "fatigued"))

    return {
        "battery_level": battery_level,
        "sleep_quality": sleep_quality,
        "hrv_state":     hrv_state,
        "weekly_load":   weekly_load,
        "cardiac_state": cardiac_state,
        "readiness":     readiness,
    }


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are an elite Endurance & Adventure Coach — part sports physiologist, part
kitesurf meteorologist, and equal parts hype engine.

A good wind window has already been confirmed. The athlete IS going to kite.
Your job is NOT to gatekeep the session — it is to maximise performance and
minimise physiological risk given the athlete's current biometrics and the
specific conditions at each ranked spot.

────────────────────────────────────────────────────────────
OUTPUT FORMAT  (strict Markdown, ≤ 220 words)
────────────────────────────────────────────────────────────

## 🪁 Kitesurf Watchdog — GO ALERT

**Top Pick: <spot_name>, <country>**
<one punchy sentence on WHY this spot wins today>

| # | Spot | Peak wind | Drive | Best window |
|---|------|-----------|-------|-------------|
| (fill top-5 table) |

---
### 🔋 Athlete Readiness: <readiness_label>
<Biometric snapshot: 2-3 bullet points interpreting the Garmin data scientifically>

### 🎯 Session Strategy
<2-3 bullet points: concrete session tactics calibrated to readiness level>
  - If readiness is peak/solid → push progression, go unhooked, extend duration.
  - If readiness is tired/fatigued → modulate intensity: smooth technique focus,
    shorter session, specific pre-session protocol (hydration, carb-loading, nap).

### 🚗 Logistics
<Drive time to top pick. Any second-choice contingency worth mentioning.>

────────────────────────────────────────────────────────────
Tone: stoked, candid, scientifically grounded. Speak to the athlete directly.
Never recommend skipping. Always recommend HOW to approach the session.
""".strip()


# ---------------------------------------------------------------------------
# Agent function
# ---------------------------------------------------------------------------

def run_head_coach(state: GraphState) -> dict[str, Any]:
    """
    Agent 3 — Head Coach.

    Synthesises the ranked spot list with the athlete's Garmin health summary
    to produce a personalised Markdown GO alert that balances trip logistics
    with physiological load management.

    Reads from state : ranked_spots, health_summary, trip_parameters
    Writes to state  : final_alert, messages
    """
    ranked_spots:   list[dict] = state.get("ranked_spots", [])
    health_summary: dict       = state.get("health_summary", {})
    trip_params:    dict       = state.get("trip_parameters", {})

    if not ranked_spots:
        no_go = "## Kitesurf Watchdog — No viable spots found for this window.\nStay home and recharge. 🔋"
        return {"final_alert": no_go, "messages": []}

    load_labels = _classify_load(health_summary)

    # ── Build spot table for the prompt ──────────────────────────────────────
    spot_lines: list[str] = []
    for s in ranked_spots:
        drive = _drive_time_str(s.get("distance_km"))
        spot_lines.append(
            f"  Rank #{s['rank']} | {s['spot_name']} ({s['country']})\n"
            f"    Peak: {s['peak_ms']} m/s  Avg: {s['avg_ms']} m/s  "
            f"Dir match: {s['direction_match_pct']}%  "
            f"Viable hrs: {s['viable_hours_count']}\n"
            f"    Best window : {s['best_session_window']}\n"
            f"    Drive time  : {drive}  ({s['distance_km']} km aerial)\n"
            f"    Score       : {s['composite_score']}/100\n"
            f"    Rationale   : {s['rationale']}\n"
            f"    URL         : {s.get('url', 'N/A')}"
        )

    spots_block = "\n\n".join(spot_lines)

    # ── Build health block for the prompt ────────────────────────────────────
    health_block = (
        f"  Body Battery        : {health_summary.get('body_battery', '?')}/100"
        f" → {load_labels['battery_level']}\n"
        f"  Sleep Score         : {health_summary.get('sleep_score', '?')}/100"
        f" → {load_labels['sleep_quality']}\n"
        f"  HRV Status          : {health_summary.get('hrv_status', '?')}"
        f" → {load_labels['hrv_state']}\n"
        f"  Resting HR          : {health_summary.get('resting_hr_bpm', '?')} bpm"
        f" → {load_labels['cardiac_state']}\n"
        f"  7-day training load : {health_summary.get('minutes_of_load_last7days', '?')} min"
        f" → {load_labels['weekly_load']}\n"
        f"  ── Overall readiness: {load_labels['readiness'].upper()} ──"
    )

    human_text = (
        f"Trip date context : {trip_params.get('current_date', 'today')}\n"
        f"Origin            : {trip_params.get('origin_city', 'unknown')}\n\n"
        f"RANKED KITESURF SPOTS:\n{spots_block}\n\n"
        f"ATHLETE BIOMETRICS:\n{health_block}\n\n"
        "Generate the personalised GO alert now. Follow the output format exactly."
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_text),
    ]

    llm = ChatGroq(model=model_used, temperature=0.6)

    print("[Head Coach] Generating personalised alert ...")
    response: AIMessage = llm.invoke(messages)

    alert_text: str = response.content
    print(f"[Head Coach] Alert generated ({len(alert_text)} chars).")

    return {
        "final_alert": alert_text,
        "messages": [*messages, response],
    }

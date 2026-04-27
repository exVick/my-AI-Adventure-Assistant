# ============================================================
# graph.py — Autonomous Black Sea Kitesurf Watchdog
# Stack : Python 3.12 | uv | LangGraph | Groq (Llama 3 70B)
# ============================================================
#
# Expected config.json (place in same directory as this file):
# {
#   "wind_threshold_knots": 15,
#   "kitesurf_spots": [
#     {"name": "Burgas Bay",    "lat": 42.53, "lon": 27.49},
#     {"name": "Pomorie Lake",  "lat": 42.61, "lon": 27.63},
#     {"name": "Nessebar Cape", "lat": 42.65, "lon": 27.72},
#     {"name": "Sozopol",       "lat": 42.42, "lon": 27.70}
#   ]
# }
#
# Required .env variables:
#   GROQ_API_KEY=gsk_...
#   GARMIN_EMAIL=your@email.com
#
# Optional .env variable (headless/CI only — less secure, see _get_garmin_client):
#   GARMIN_PASSWORD=...
#
# Garmin auth strategy:
#   1. Load cached OAuth tokens from ~/.garminconnect/  ← no credentials needed
#   2. If tokens missing/expired: prompt password once via getpass() (never written to disk)
#      Tokens are then saved and re-used on every future run (~30 day lifetime).
#
# Pipeline control flow:
#   START
#     └─► Weather_Scanner_Node
#               ├─► (no viable spots) ─► END
#               └─► (viable spots)   ─► Health_Node
#                                              └─► Reasoning_Node
#                                                        └─► Alert_Node ─► END
# ============================================================

import os
import json
import random
import getpass
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Dict, Any

from dotenv import load_dotenv
from garminconnect import Garmin, GarminConnectAuthenticationError
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# Reads GROQ_API_KEY (and any other vars) from the .env file in this directory
load_dotenv()

# ============================================================
# PATHS — always relative to this script, not the working dir
# ============================================================

_HERE = Path(__file__).parent
CONFIG_PATH = _HERE / "config.json"

# OAuth tokens are cached here after the first login; the password is never stored.
_GARMIN_TOKENSTORE = str(Path("~/.garminconnect").expanduser())


# ============================================================
# GARMIN AUTH HELPER
# ============================================================

def _get_garmin_client() -> Garmin:
    """
    Returns an authenticated Garmin client using a two-stage strategy:

    Stage 1 — Token cache (no credentials needed):
        Loads OAuth tokens from ~/.garminconnect/. This path is used on every
        run after the first successful login. Tokens are valid ~30 days and are
        silently refreshed by the library before each API call.

    Stage 2 — Fresh login (first run or tokens expired):
        Reads GARMIN_EMAIL from .env. For the password, it first checks for a
        GARMIN_PASSWORD env var (headless/CI use only — less secure because the
        password lives in a file). If that var is absent, it prompts with
        getpass(), which masks typing and never writes the password to disk.
        After a successful login, tokens are saved to ~/.garminconnect/ so
        Stage 1 succeeds on all future runs.
    """
    os.makedirs(_GARMIN_TOKENSTORE, exist_ok=True)

    # Stage 1: attempt to restore from cached tokens
    try:
        client = Garmin()
        client.login(_GARMIN_TOKENSTORE)
        print("[Garmin Auth] Logged in via cached OAuth tokens.")
        return client
    except (GarminConnectAuthenticationError, Exception):
        pass  # Tokens missing or expired — fall through to fresh login

    # Stage 2: fresh login
    email = os.environ.get("GARMIN_EMAIL")
    if not email:
        raise RuntimeError(
            "GARMIN_EMAIL is not set and no cached Garmin tokens were found.\n"
            "Add  GARMIN_EMAIL=your@email.com  to my-AI-Adventure-Assistant/.env"
        )

    # Prefer interactive prompt so the password is never written to any file.
    # GARMIN_PASSWORD in .env is a fallback for fully headless/automated use only.
    password = os.environ.get("GARMIN_PASSWORD") or getpass.getpass(
        f"Garmin password for {email} (input hidden): "
    )

    client = Garmin(
        email=email,
        password=password,
        prompt_mfa=lambda: input("Garmin MFA code: "),
    )
    client.login(_GARMIN_TOKENSTORE)
    print(f"[Garmin Auth] Login successful. Tokens cached → {_GARMIN_TOKENSTORE}")
    return client


# ============================================================
# 1. GRAPH STATE
# ============================================================

class GraphState(TypedDict):
    """
    Single source of truth threaded through every node.

    viable_spots        — Spots where wind >= threshold (populated by Weather_Scanner_Node).
                          An empty list is the signal that today is a no-go.
    health_metrics      — Garmin data dict (populated by Health_Node).
    final_alert_message — LLM-generated alert string (populated by Reasoning_Node).
    """
    viable_spots: List[Dict[str, Any]]
    health_metrics: Dict[str, Any]
    final_alert_message: str


# ============================================================
# 2. NODES
# ============================================================

def weather_scanner_node(state: GraphState) -> Dict[str, Any]:
    """
    NODE 1 — Weather Scanner

    Reads config.json, iterates every configured kitesurf spot, and mocks
    an Open-Meteo API call to retrieve current wind speed. Only spots where
    wind >= wind_threshold_knots are written into `viable_spots`.

    Production swap-in:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": spot["lat"], "longitude": spot["lon"],
            "current": "wind_speed_10m", "wind_speed_unit": "kn",
            "timezone": "auto",
        }
        wind_kn = requests.get(url, params=params, timeout=10)
                            .json()["current"]["wind_speed_10m"]
    """
    with open(CONFIG_PATH, "r") as fh:
        config = json.load(fh)

    spots: List[Dict] = config["kitesurf_spots"]
    threshold: float = float(config["wind_threshold_knots"])

    print(f"\n[Weather Scanner] Scanning {len(spots)} spot(s)  |  threshold: {threshold} kn")

    viable: List[Dict[str, Any]] = []

    for spot in spots:
        # ── MOCK: replace with real API call in production ──────────────────
        wind_kn = round(random.uniform(8.0, 28.0), 1)
        # ────────────────────────────────────────────────────────────────────

        status = "✓ VIABLE" if wind_kn >= threshold else "✗ below threshold"
        print(f"  → {spot['name']:<20} {wind_kn:>5} kn   {status}")

        if wind_kn >= threshold:
            viable.append({
                "name":       spot["name"],
                "lat":        spot["lat"],
                "lon":        spot["lon"],
                "wind_knots": wind_kn,
            })

    print(f"[Weather Scanner] {len(viable)} viable spot(s) identified.\n")

    # LangGraph merges this partial dict with the existing state automatically
    return {"viable_spots": viable}


def health_node(state: GraphState) -> Dict[str, Any]:
    """
    NODE 3 — Garmin Health Node

    Fetches live health metrics from Garmin Connect for the last 1-7 days:
      - Body battery   : most recent reading from today's/yesterday's time-series
      - Sleep score    : overall score from last night (yesterday)
      - HRV status     : yesterday's HRV classification from Garmin
      - Resting HR     : yesterday's resting heart rate
      - Recent load    : classified from total training minutes in the last 7 days

    Falls back to clearly-labelled mock data if the Garmin API is unreachable,
    so a network blip never silently kills the pipeline.
    """
    today     = date.today()
    yesterday = (today - timedelta(days=1)).isoformat()
    today_str = today.isoformat()

    try:
        client = _get_garmin_client()

        # TODO: think of better informing metrics
        # ── Body Battery ──────────────────────────────────────────────────────
        # Request a 2-day window to ensure we catch today's readings even if the
        # device hasn't synced today's data yet (falls back to yesterday's last value).
        bb_raw = client.get_body_battery(yesterday, today_str)
        body_battery = 0
        if bb_raw:
            # Each element in the list covers one day; take the most recent day last.
            series = bb_raw[-1].get("bodyBatteryValuesArray", [])
            if series:
                # Each entry is [unix_timestamp_ms, battery_level_int]
                body_battery = int(series[-1][1])

        # ── Sleep Score (last night) ──────────────────────────────────────────
        sleep_raw  = client.get_sleep_data(yesterday) or {}
        sleep_score = (
            sleep_raw.get("dailySleepDTO", {})
                     .get("sleepScores", {})
                     .get("overall", {})
                     .get("value", 0)
        ) or 0

        # ── HRV Status (yesterday) ────────────────────────────────────────────
        hrv_raw    = client.get_hrv_data(yesterday) or {}
        hrv_status = (
            hrv_raw.get("hrvSummary", {})
                   .get("status", "unknown")
                   .lower()
        )

        # ── Resting Heart Rate (yesterday) ────────────────────────────────────
        hr_raw    = client.get_heart_rates(yesterday) or {}
        resting_hr = hr_raw.get("restingHeartRate", 0) or 0

        # ── Recent Training Load (last 7 days) ────────────────────────────────
        # Fetch the 10 most recent activities and sum those within the 7-day window.
        # TODO: improve logic - maybe sample not only minutes
        activities   = client.get_activities(0, 10) or []
        cutoff_str   = (today - timedelta(days=7)).isoformat()
        recent_minutes = sum(
            a.get("duration", 0) / 60
            for a in activities
            if a.get("startTimeLocal", "")[:10] >= cutoff_str
        )

        metrics: Dict[str, Any] = {
            "body_battery":   body_battery,
            "sleep_score":    sleep_score,
            "minutes_of_load_last7days":    recent_minutes,
            "resting_hr_bpm": resting_hr,
            "hrv_status":     hrv_status,
        }

        print(f"[Health Node] Live Garmin metrics: {metrics}\n")
        return {"health_metrics": metrics}

    except Exception as exc:
        # Surface the error clearly but let the pipeline finish with mock data.
        print(f"[Health Node] WARNING — Garmin fetch failed: {exc}")
        print("[Health Node] Falling back to mock metrics.\n")
        return {"health_metrics": {
            "body_battery":   30,
            "sleep_score":    45,
            "minutes_of_load_last7days":   420,
            "resting_hr_bpm": 58,
            "hrv_status":     "poor",
            "_source":        "mock_fallback",
        }}


def reasoning_node(state: GraphState) -> Dict[str, Any]:
    """
    NODE 4 — Groq Reasoning Node (Llama 3 70B)

    The go/no-go decision is NOT re-evaluated here — the wind scanner
    already confirmed viable conditions. The LLM's sole responsibility is to
    adjust advisory tone and generate health-aware safety/recovery guidance.
    """
    viable_spots = state["viable_spots"]
    metrics      = state["health_metrics"]

    # Build a readable spot list for the prompt
    spot_lines = "\n".join(
        f"  • {s['name']:<20} — {s['wind_knots']} kn"
        for s in viable_spots
    )

    # ── System prompt: defines the LLM's locked role ────────────────────────
    system_prompt = ("""
        You are an elite Endurance & Adventure Coach specializing in human physiology and extreme sports performance. 

        CRITICAL CONTEXT: 
        If you are being invoked, it means the system has already confirmed there is a highly favorable kitesurfing wind forecast in the coming days. The user IS going to kite. A good wind day is non-negotiable. 

        YOUR PRIME DIRECTIVE:
        Never advise the user to skip or cancel a kitesurfing session due to poor health metrics. Instead, use their provided Garmin biosignals (Body Battery, Sleep Score, Minutes of Load, Resting HR, HRV Status) to dynamically coach them on HOW to approach the session and manage their physiological load.

        COACHING GUIDELINES:
        1. High Energy / High Recovery: If their metrics are solid, hype them up. Tell them to push progression, aim for big air, or extend the session duration.
        2. Low Energy / CNS Fatigue: If their Body Battery is low or sleep is poor, do not tell them to stay home. Instead, advise them to modulate the intensity. Suggest a shorter, punchy session, focusing on smooth technique rather than heavy unhooked tricks. Provide specific biological recovery advice (e.g., pre-hydration protocols, targeted carb-refeeding, or prioritizing a nap before hitting the beach).
        3. Weather Context: Briefly integrate the specific wind conditions (speed, spot) into your hype message.

        TONE: 
        Highly stoked, candid, and scientifically grounded. Speak to them as someone who deeply understands biological recovery systems. 

        FORMAT:
        Output your response as a punchy, highly readable alert suitable for a push notification. Use markdown formatting, bullet points for the recovery/session tactics, and keep it under 150 words.
    """)

    # ── Human turn: the live runtime data ────────────────────────────────────
    human_prompt = (
        f"VIABLE KITE SPOTS CONFIRMED TODAY:\n{spot_lines}\n\n"
        f"ATHLETE HEALTH METRICS RIGHT NOW:\n"
        f"  • Body Battery                  : {metrics['body_battery']}/100\n"
        f"  • Sleep Score                   : {metrics['sleep_score']}/100\n"
        f"  • Minutes of Load (last 7 days) : {metrics['minutes_of_load_last7days']}\n"
        f"  • Resting HR                    : {metrics['resting_hr_bpm']} bpm\n"
        f"  • HRV Status                    : {metrics['hrv_status']}\n\n"
        "Generate the tailored kite alert now."
    )

    print("[Reasoning Node] Invoking Groq / Llama 3 70B ...")

    # ChatGroq picks up GROQ_API_KEY from the environment automatically
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.6)

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])

    alert_text: str = response.content
    print(f"[Reasoning Node] Response received ({len(alert_text)} chars).\n")

    return {"final_alert_message": alert_text}


def alert_node(state: GraphState) -> Dict[str, Any]:
    """
    NODE 5 — Alert Dispatcher

    Prints the final alert in a loud, unmissable console block.

    Production swap-in options (pick one):
        • Pushover / ntfy / Telegram bot  → HTTP POST with the message body
        • Home Assistant webhook          → requests.post(HA_WEBHOOK_URL, json={...})
        • SMS via Twilio                  → twilio_client.messages.create(...)
        • Desktop notification (Linux)    → subprocess.run(["notify-send", message])
    """
    message = state["final_alert_message"]
    border  = "=" * 62

    print(border)
    print("  🪁  KITESURF WATCHDOG — ✅  GO ALERT  ✅")
    print(border)
    print(message)
    print(border)

    # Nothing new to write to state; LangGraph expects a dict return
    return {}


# ============================================================
# 3. CONDITIONAL ROUTER
# ============================================================

def conditional_router(state: GraphState) -> str:
    """
    Inspects `viable_spots` after the weather scan and returns the name of
    the next node (or the END sentinel) for LangGraph to route to.

    Returns:
        "Health_Node" — at least one spot exceeded the wind threshold.
        END           — no viable spots; pipeline halts without sending an alert.
    """
    if state.get("viable_spots"):
        print("[Router] Viable spots detected → routing to Health_Node.\n")
        return "Health_Node"

    print("[Router] No viable spots today → routing to END. Stay home and recharge.\n")
    return END


# ============================================================
# 4. GRAPH ASSEMBLY
# ============================================================

workflow = StateGraph(GraphState)

# ── Register nodes ────────────────────────────────────────────────────────────
workflow.add_node("Weather_Scanner_Node", weather_scanner_node)
workflow.add_node("Health_Node",          health_node)
workflow.add_node("Reasoning_Node",       reasoning_node)
workflow.add_node("Alert_Node",           alert_node)

# ── Entry edge ────────────────────────────────────────────────────────────────
workflow.add_edge(START, "Weather_Scanner_Node")

# ── Conditional branch off the weather scanner ────────────────────────────────
# The dict maps each possible return value of `conditional_router` to a node
# name (or END). LangGraph uses this mapping to resolve the actual edge target.
workflow.add_conditional_edges(
    source="Weather_Scanner_Node",
    path=conditional_router,
    path_map={
        "Health_Node": "Health_Node",
        END:           END,
    },
)

# ── Linear happy-path tail ────────────────────────────────────────────────────
workflow.add_edge("Health_Node",    "Reasoning_Node")
workflow.add_edge("Reasoning_Node", "Alert_Node")
workflow.add_edge("Alert_Node",     END)

# Compile into a runnable LangGraph Application object
app = workflow.compile()


# ============================================================
# 5. ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "ERROR: GROQ_API_KEY is not set.\n"
            "Add it to your .env file:   GROQ_API_KEY=gsk_...\n"
            "Or export it:               export GROQ_API_KEY='gsk_...'"
        )

    print("=" * 62)
    print("  AUTONOMOUS KITESURF WATCHDOG — PIPELINE START")
    print("=" * 62)

    # All fields default to their empty/falsy equivalents on first run
    initial_state: GraphState = {
        "viable_spots":        [],
        "health_metrics":      {},
        "final_alert_message": "",
    }

    final_state = app.invoke(initial_state)

    print("\n[Pipeline] Execution complete.")
    if not final_state.get("viable_spots"):
        print("[Pipeline] No viable spots were found — no alert was generated.")

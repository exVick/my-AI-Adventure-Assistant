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
# Required .env variable:
#   GROQ_API_KEY=gsk_...
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
from pathlib import Path
from typing import TypedDict, List, Dict, Any

from dotenv import load_dotenv
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

    Mocks a garminconnect pull. Deliberately set to low/poor values so the
    Reasoning Node's tone customisation is clearly exercised.

    Production swap-in (inside a try/except for network resilience):
        from garminconnect import Garmin
        from datetime import date
        client = Garmin(os.environ["GARMIN_EMAIL"], os.environ["GARMIN_PASSWORD"])
        client.login()
        bb    = client.get_body_battery()[0].get("charged", 0)
        sleep = client.get_sleep_data(date.today().isoformat()) \
                      ["dailySleepDTO"]["sleepScores"]["overall"]["value"]
        ...
    """
    # ── MOCK: replace with real Garmin client calls in production ────────────
    metrics: Dict[str, Any] = {
        "body_battery":   30,      # Low — intentionally stressful for tone testing
        "sleep_score":    45,      # Poor night
        "recent_load":    "high",  # Heavy training block
        "resting_hr_bpm": 58,
        "hrv_status":     "poor",
    }
    # ─────────────────────────────────────────────────────────────────────────

    print(f"[Health Node] Garmin metrics retrieved: {metrics}\n")
    return {"health_metrics": metrics}


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
    system_prompt = (
        "You are an elite Endurance & Adventure Coach embedded inside an autonomous "
        "kitesurfing watchdog system.\n\n"
        "CRITICAL INSTRUCTION: The decision to kite TODAY IS ALREADY CONFIRMED — "
        "the wind scanner has verified that sufficient wind conditions exist at one "
        "or more spots. You must NOT re-evaluate or second-guess this go/no-go call.\n\n"
        "Your ONLY responsibilities are:\n"
        "1. Acknowledge the confirmed session with energy and enthusiasm.\n"
        "2. Use the athlete's health metrics EXCLUSIVELY to calibrate the advisory "
        "tone and provide tailored safety and recovery advice:\n"
        "   • If body battery < 40 → recommend a shorter session (≤ 90 min) and "
        "mandatory post-session rest.\n"
        "   • If sleep score < 50  → flag dehydration risk; push fluids before launch.\n"
        "   • If recent_load is 'high' → advise lighter kite size and active cool-down.\n"
        "   • If hrv_status is 'poor' → emphasise warm-up and avoiding max-effort runs.\n"
        "3. Output a concise, motivating, safety-first alert — no more than 150 words. "
        "Use clear bullet points for actionable advice items."
    )

    # ── Human turn: the live runtime data ────────────────────────────────────
    human_prompt = (
        f"VIABLE KITE SPOTS CONFIRMED TODAY:\n{spot_lines}\n\n"
        f"ATHLETE HEALTH METRICS RIGHT NOW:\n"
        f"  • Body Battery  : {metrics['body_battery']}/100\n"
        f"  • Sleep Score   : {metrics['sleep_score']}/100\n"
        f"  • Recent Load   : {metrics['recent_load']}\n"
        f"  • Resting HR    : {metrics['resting_hr_bpm']} bpm\n"
        f"  • HRV Status    : {metrics['hrv_status']}\n\n"
        "Generate the tailored kite alert now."
    )

    print("[Reasoning Node] Invoking Groq / Llama 3 70B ...")

    # ChatGroq picks up GROQ_API_KEY from the environment automatically
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.6)

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

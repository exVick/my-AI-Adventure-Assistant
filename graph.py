# ============================================================
# graph.py — Autonomous Kitesurf Watchdog  (V2)
# Stack : Python 3.12 | uv | LangGraph | Groq (Llama 3 70B)
# ============================================================
#
# Required .env variables:
#   GROQ_API_KEY=gsk_...
#   GARMIN_EMAIL=your@email.com
#   SPOTS_JSON_PATH=/absolute/path/to/closeby_kitespots.json
#
# Optional .env variables:
#   GARMIN_PASSWORD=...          # headless/CI only; prefer interactive prompt
#   ORIGIN_CITY=Sofia, Bulgaria  # departure city label
#   ORIGIN_LAT=42.70             # departure latitude
#   ORIGIN_LON=23.32             # departure longitude
#
# ──────────────────────────────────────────────────────────
# V2 PIPELINE — CONTROL FLOW
# ──────────────────────────────────────────────────────────
#
#   START
#     │
#     ▼
#   [Trip_Planner_Node]          ← Agent 1 (LLM)
#     │  Reasons about date/season; issues a Set_Trip_Parameters tool call.
#     │  Output → state.trip_parameters, state.messages
#     ▼
#   [Database_Filter_Node]       ← Pure Python
#     │  Intercepts Agent 1's tool call result; runs filter_spots() with the
#     │  extracted parameters (months, distance, origin coords).
#     │  Output → state.candidate_spots
#     ▼
#   [Weather_Node]               ← Pure Python (parallel HTTP)
#     │  Calls check_wind_forecasts() on every candidate spot concurrently.
#     │  Output → state.weather_forecasts
#     │
#     ├─► (weather_forecasts is EMPTY) ──────────────────────► END
#     │     No viable wind anywhere — pipeline halts, no alert sent.
#     │
#     └─► (at least one viable forecast)
#           │
#           ▼
#   [Wind_Expert_Node]           ← Agent 2 (LLM + Python scoring)
#     │  Scores every viable spot (direction match, intensity, window, proximity).
#     │  LLM selects top-5 with expert rationale.
#     │  Output → state.ranked_spots
#     ▼
#   [Health_And_Coach_Node]      ← garmin_tools + Agent 3 (LLM)
#     │  Fetches live Garmin biometrics → state.health_summary
#     │  Head Coach LLM synthesises ranked spots + health data → state.final_alert
#     ▼
#   END
#
# ============================================================

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from state import GraphState
from tools.database_tools import filter_spots
from tools.weather_tools import check_wind_forecasts
from tools.garmin_tools import fetch_health_summary
from agents.trip_planner import run_trip_planner
from agents.wind_expert import run_wind_expert
from agents.head_coach import run_head_coach

# Reads GROQ_API_KEY and all other vars from the project's .env file
load_dotenv(Path(__file__).parent / ".env")

_HERE = Path(__file__).parent


# ============================================================
# NODE 1 — Trip Planner
# ============================================================

def trip_planner_node(state: GraphState) -> dict[str, Any]:
    """
    Delegates to Agent 1 (run_trip_planner).

        The LLM issues a SetTripParameters tool call whose arguments encode:
      • target_months          — which months to search in the spot database
      • preferred_directions   — compass points the Wind Expert will score against
      • preferred_wind_types   — thermal / bora / mistral priority
      • trip_rationale         — LLM's reasoning for these parameters (stored for inspection)

        Deterministic policy/config values are merged in by the Trip Planner code:
            • max_distance_km        — radius from the origin city
            • min_wind_kn            — viability threshold forwarded to weather_tools

    These parameters are stored in state.trip_parameters so every downstream
    node can read them without re-querying the LLM.
    """
    print("\n" + "=" * 62)
    print("  NODE 1 — TRIP PLANNER")
    print("=" * 62)
    return run_trip_planner(state)


# ============================================================
# NODE 2 — Database Filter
# ============================================================

def database_filter_node(state: GraphState) -> dict[str, Any]:
    """
    Pure Python node — no LLM involved.

    Intercepts the trip_parameters written by Node 1 and feeds them directly
    into filter_spots().  This node is the bridge between the LLM's structured
    tool call and the actual kitespot database.

    Design note: we intentionally keep the DB query in its own node (rather
    than inside the Trip Planner agent) so the graph makes the tool-call
    execution explicit and traceable in LangGraph Studio.
    """
    print("\n" + "=" * 62)
    print("  NODE 2 — DATABASE FILTER")
    print("=" * 62)

    params = state["trip_parameters"]

    spots_json_path = os.environ.get("SPOTS_JSON_PATH", "")
    candidate_spots = filter_spots(
        spots_json_path  = spots_json_path,
        target_months    = params["target_months"],
        max_distance_km  = float(params["max_distance_km"]),
        origin_lat       = params["origin_lat"],
        origin_lon       = params["origin_lon"],
    )

    return {"candidate_spots": candidate_spots}


# ============================================================
# NODE 3 — Weather Scanner
# ============================================================

def weather_node(state: GraphState) -> dict[str, Any]:
    """
    Pure Python node — parallel HTTP via concurrent.futures.

    Calls check_wind_forecasts() which fans out one Open-Meteo request per
    candidate spot using a thread pool, then filters out any spot whose peak
    wind over the forecast window is below the threshold.

    The resulting weather_forecasts dict is keyed by spot name.  An empty
    dict is the signal that today is a no-go — the conditional edge below
    will route to END in that case.
    """
    print("\n" + "=" * 62)
    print("  NODE 3 — WEATHER SCANNER")
    print("=" * 62)

    params           = state["trip_parameters"]
    candidate_spots  = state["candidate_spots"]
    min_wind_kn      = float(params.get("min_wind_kn", 15))

    weather_forecasts = check_wind_forecasts(
        candidate_spots   = candidate_spots,
        wind_threshold_kn = min_wind_kn,
    )

    return {"weather_forecasts": weather_forecasts}


# ============================================================
# CONDITIONAL ROUTER — after Weather Node
# ============================================================

def route_after_weather(state: GraphState) -> str:
    """
    Inspects state.weather_forecasts after the parallel API scan.

    Returns the name of the NEXT NODE for LangGraph to route to:
      • "Wind_Expert_Node"  — at least one spot has viable wind conditions.
      • END                 — no viable spots found; pipeline halts cleanly
                              without generating a false alert.

    This is the only conditional branch in the V2 graph.  All other edges
    are deterministic.
    """
    if state.get("weather_forecasts"):
        n = len(state["weather_forecasts"])
        print(f"\n[Router] {n} viable forecast(s) → routing to Wind Expert.\n")
        return "Wind_Expert_Node"

    print("\n[Router] No viable wind anywhere → routing to END. Rest day. 🛋️\n")
    return END


# ============================================================
# NODE 4 — Wind Expert
# ============================================================

def wind_expert_node(state: GraphState) -> dict[str, Any]:
    """
    Delegates to Agent 2 (run_wind_expert).

    Agent 2 scores every viable spot using a pure-Python composite scorer
    (direction match + wind intensity + session window + proximity distance),
    then passes the pre-scored table to Llama 3 70B for qualitative expert
    ranking and rationale generation.  Returns the top-5 spots as a list of
    dicts stored in state.ranked_spots.
    """
    print("\n" + "=" * 62)
    print("  NODE 4 — WIND EXPERT")
    print("=" * 62)
    return run_wind_expert(state)


# ============================================================
# NODE 5 — Health & Coach  (combined node)
# ============================================================

def health_and_coach_node(state: GraphState) -> dict[str, Any]:
    """
    Two responsibilities combined into one node to avoid a redundant state
    round-trip:

    Step A — Garmin fetch (pure Python, garmin_tools.fetch_health_summary):
        Authenticates with Garmin Connect via cached OAuth tokens and pulls
        body battery, sleep score, HRV status, resting HR, and 7-day training
        load.  Falls back to clearly-labelled mock data on network failure.
        Writes to state.health_summary.

    Step B — Head Coach alert (Agent 3, run_head_coach):
        Reads state.ranked_spots + state.health_summary, classifies the
        athlete's physiological readiness, estimates drive times, and invokes
        Llama 3 70B to generate a personalised Markdown GO alert.
        Writes to state.final_alert.

    Why combined? The health fetch is cheap and synchronous; splitting it into
    its own node would add a graph hop without any routing benefit.  The Coach
    LLM always follows the health fetch unconditionally.
    """
    print("\n" + "=" * 62)
    print("  NODE 5 — HEALTH & COACH")
    print("=" * 62)

    # ── Step A: pull live Garmin biometrics ──────────────────────────────────
    print("\n[Health] Fetching Garmin biometrics ...")
    health_summary = fetch_health_summary()

    # Merge health data into state before calling the coach so run_head_coach
    # can read it directly from the state dict.
    state_with_health: GraphState = {**state, "health_summary": health_summary}  # type: ignore[misc]

    # ── Step B: generate the personalised alert ──────────────────────────────
    coach_updates = run_head_coach(state_with_health)

    # Return both the health summary and the coach's output as one state patch.
    return {
        "health_summary": health_summary,
        **coach_updates,
    }


# ============================================================
# GRAPH ASSEMBLY
# ============================================================

def build_graph() -> StateGraph:
    """
    Assembles and compiles the V2 LangGraph StateGraph.

    Edge map:
      START
        → Trip_Planner_Node     (always)
        → Database_Filter_Node  (always)
        → Weather_Node          (always)
        → [conditional branch]
              ├─ Wind_Expert_Node     → Health_And_Coach_Node → END
              └─ END  (no viable forecasts)
    """
    workflow = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────────────────────
    workflow.add_node("Trip_Planner_Node",     trip_planner_node)
    workflow.add_node("Database_Filter_Node",  database_filter_node)
    workflow.add_node("Weather_Node",          weather_node)
    workflow.add_node("Wind_Expert_Node",      wind_expert_node)
    workflow.add_node("Health_And_Coach_Node", health_and_coach_node)

    # ── Deterministic edges ───────────────────────────────────────────────────
    #
    # The first three nodes always run sequentially: the LLM plans the trip,
    # the DB filter narrows the candidate list, and the weather scanner checks
    # live conditions.  No branching until we know if there is viable wind.
    workflow.add_edge(START,                   "Trip_Planner_Node")
    workflow.add_edge("Trip_Planner_Node",     "Database_Filter_Node")
    workflow.add_edge("Database_Filter_Node",  "Weather_Node")

    # ── Conditional branch — the only decision point in the graph ─────────────
    #
    # route_after_weather() returns either:
    #   "Wind_Expert_Node"  → at least one viable forecast exists
    #   END                 → no viable wind; pipeline halts here
    #
    # path_map makes the routing explicit and visible in LangGraph Studio.
    workflow.add_conditional_edges(
        source   = "Weather_Node",
        path     = route_after_weather,
        path_map = {
            "Wind_Expert_Node": "Wind_Expert_Node",
            END:                END,
        },
    )

    # ── Happy-path tail ───────────────────────────────────────────────────────
    #
    # Once viable forecasts exist the remaining two nodes always run in order:
    # the Wind Expert ranks spots, then the Head Coach generates the alert.
    workflow.add_edge("Wind_Expert_Node",      "Health_And_Coach_Node")
    workflow.add_edge("Health_And_Coach_Node", END)

    return workflow.compile()


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    # ── Pre-flight checks ─────────────────────────────────────────────────────
    missing = [v for v in ("GROQ_API_KEY", "GARMIN_EMAIL", "SPOTS_JSON_PATH") if not os.environ.get(v)]
    if missing:
        raise SystemExit(
            f"ERROR: missing required env var(s): {', '.join(missing)}\n"
            "Add them to my-AI-Adventure-Assistant/.env"
        )

    # ── Compile ───────────────────────────────────────────────────────────────
    app = build_graph()

    # ── Banner ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  AUTONOMOUS KITESURF WATCHDOG V2 — PIPELINE START")
    print("=" * 62)

    # ── Initial state — all fields empty; each node fills its own slice ───────
    #
    # LangGraph merges partial dicts returned by each node into the running
    # state automatically.  No field needs a value at startup — nodes that read
    # a field always run after the node that writes it (enforced by edge order).
    initial_state: GraphState = {
        "messages":         [],
        "trip_parameters":  {},
        "candidate_spots":  [],
        "weather_forecasts": {},
        "ranked_spots":     [],
        "health_summary":   {},
        "final_alert":      "",
    }

    final_state = app.invoke(initial_state)

    # ── Terminal output ───────────────────────────────────────────────────────
    border = "=" * 62
    if final_state.get("final_alert"):
        print(f"\n{border}")
        print(final_state["final_alert"])
        print(border)
    else:
        print(f"\n{border}")
        print("  No viable kitesurf conditions found for this window.")
        print("  Recover well. The wind will come back. 🔋")
        print(border)

    print("\n[Pipeline] Execution complete.")

import os
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from state import GraphState

# Define model
model_used = "openai/gpt-oss-120b"

# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

class SetTripParameters(BaseModel):
    """Structured output that parameterises the kitespot database filter and
    downstream wind/health scoring for this pipeline run."""

    target_months: list[str] = Field(
        description=(
            "Full English month names to match against spot seasonality data. "
            "Include the current month plus the next 1-2 months so imminent and "
            "near-future windows are both captured. E.g. ['April', 'May', 'June']."
        )
    )
    max_distance_km: int = Field(
        description=(
            "Maximum one-way driving distance from the origin in km. "
            "Weekend trip from Sofia → Black Sea / Aegean / Adriatic: 400-800 km. "
            "Never exceed 1200 km."
        )
    )
    preferred_wind_types: list[str] = Field(
        description=(
            "Wind type(s) to prioritise when scoring spots. "
            "Common values: 'Thermal', 'Seabreeze', 'Mistral', 'Bora', 'Cross-shore'. "
            "Prioritise 'Thermal' for April-September in the Mediterranean / Black Sea."
        )
    )
    preferred_directions: list[str] = Field(
        description=(
            "Preferred wind compass directions for kitesurfing (side-shore or "
            "side-onshore are safest). Use standard 16-point abbreviations: "
            "N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW."
        )
    )
    min_wind_kn: int = Field(
        description=(
            "Minimum sustained wind speed in knots to consider a spot viable. "
            "Standard kitesurf threshold: 15 kn. Lower only for large-kite setups."
        )
    )
    trip_rationale: str = Field(
        description=(
            "One or two sentences explaining why these parameters were chosen — "
            "mention the season, dominant wind regime, and any distance trade-offs."
        )
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are an expert kitesurf trip planner with deep knowledge of Mediterranean,
Aegean, Adriatic, and Black Sea wind patterns.

Your job: analyse the current date and the user's departure city, then call the
SetTripParameters tool with optimal filter parameters for finding kitesurf spots
for an upcoming weekend or short trip.

## Reasoning rules:
• target_months  : always include the current month + next 1-2 months.
• max_distance_km: for an inland origin (Sofia), the weekend sweet spot is
  400-800 km. Stretch to 1000 km only for exceptional destinations.
• preferred_wind_types:
    – April-September  → "Thermal" dominates Black Sea / Aegean coasts.
    – Oct-March        → "Bora" (Adriatic), "Mistral" (W. Mediterranean) are
                          the primary offshore drivers.
• preferred_directions: side-shore (≈ 45-90° off the beach) is the safest
  kite angle. For the Black Sea / Aegean in spring-summer: NE, ENE, E thermals.
  For the Adriatic: N, NNE bora.
• min_wind_kn: default 15. Use 12 only if explicitly noted.

You MUST call SetTripParameters. Plain text responses are not acceptable.
""".strip()


# ---------------------------------------------------------------------------
# Agent function
# ---------------------------------------------------------------------------

def run_trip_planner(state: GraphState) -> dict[str, Any]:
    """
    Agent 1 — Trip Planner.

    Reads the current date and the user's origin location (env vars), then
    invokes Llama 3 70B with the SetTripParameters tool bound.  The tool-call
    arguments are extracted and merged with the runtime origin coordinates to
    form `trip_parameters`.

    Reads from state : nothing (uses live date + env config)
    Writes to state  : trip_parameters, messages
    """
    today       = date.today()
    origin_city = os.environ.get("ORIGIN_CITY", "Sofia, Bulgaria")
    origin_lat  = float(os.environ.get("ORIGIN_LAT", "42.70"))
    origin_lon  = float(os.environ.get("ORIGIN_LON", "23.32"))

    human_text = (
        f"Today's date    : {today.strftime('%A, %d %B %Y')}\n"
        f"Departure point : {origin_city} "
        f"(lat={origin_lat:.4f}, lon={origin_lon:.4f})\n\n"
        "Determine the optimal trip parameters and call SetTripParameters now."
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=human_text),
    ]

    llm = ChatGroq(model=model_used, temperature=0)
    llm_with_tool = llm.bind_tools([SetTripParameters])

    print("[Trip Planner] Invoking LLM ...")
    response: AIMessage = llm_with_tool.invoke(messages)

    if not response.tool_calls:
        raise RuntimeError(
            "[Trip Planner] LLM did not call SetTripParameters.\n"
            f"Raw response: {response.content}"
        )

    args: dict = response.tool_calls[0]["args"]
    trip_parameters: dict[str, Any] = {
        **args,
        "origin_city":  origin_city,
        "origin_lat":   origin_lat,
        "origin_lon":   origin_lon,
        "current_date": today.isoformat(),
    }

    print(f"[Trip Planner] Parameters locked in:\n  {trip_parameters}")
    return {
        "trip_parameters": trip_parameters,
        "messages": [*messages, response],
    }

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
    trip_rationale: str = Field(
        description=(
            "One or two sentences explaining why these wind/season parameters were chosen — "
            "mention seasonality and dominant regional wind regime."
        )
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are an expert kitesurf trip planner with deep knowledge of Mediterranean,
Aegean, Adriatic, and Black Sea wind patterns.

Your job: analyse the current date and the user's departure city, then call the
SetTripParameters tool with the target months and a short rationale for the
search parameters.

## Reasoning rules:
• target_months  : always include the current month + next 2-3 months.

Operational constraints are handled in code, not by this tool call:
• max_distance_km is set by policy/config.

You MUST call SetTripParameters. Plain text responses are not acceptable.
""".strip()


def _default_max_distance_km(today: date) -> int:
    """Deterministic policy so LLM does not invent distance thresholds."""
    env_override = os.environ.get("MAX_DISTANCE_KM")
    if env_override:
        return int(env_override)

    # Slightly larger radius in winter when thermal options are fewer.
    if today.month in {11, 12, 1, 2}:
        return 900
    return 800


# ---------------------------------------------------------------------------
# Agent function
# ---------------------------------------------------------------------------

def run_trip_planner(state: GraphState) -> dict[str, Any]:
    """
    Agent 1 — Trip Planner.

    Reads the current date and the user's origin location (env vars), then
    invokes specified LLM with the SetTripParameters tool bound.  The tool-call
    arguments are extracted and merged with deterministic policy values and
    runtime origin coordinates to form `trip_parameters`.

    Reads from state : nothing (uses live date + env config)
    Writes to state  : trip_parameters, messages
    """
    today       = date.today()
    origin_city = os.environ.get("ORIGIN_CITY", "Sofia, Bulgaria")
    origin_lat  = float(os.environ.get("ORIGIN_LAT", "42.70"))
    origin_lon  = float(os.environ.get("ORIGIN_LON", "23.32"))
    max_distance_km = _default_max_distance_km(today)

    human_text = (
        f"Today's date    : {today.strftime('%A, %d %B %Y')}\n"
        f"Departure point : {origin_city} "
        f"(lat={origin_lat:.4f}, lon={origin_lon:.4f})\n\n"
        f"Fixed constraints (policy): max_distance_km={max_distance_km}.\n"
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
        "max_distance_km": max_distance_km,
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

from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    trip_parameters: dict
    candidate_spots: list
    weather_forecasts: dict
    ranked_spots: list
    health_summary: dict
    final_alert: str

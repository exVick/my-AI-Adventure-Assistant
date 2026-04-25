import os
import requests
from typing import TypedDict, Annotated, List, Dict, Any
import operator

# Modern LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Langchain imports
from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# This automatically reads your .env file and loads the keys into your environment
load_dotenv()

# ==========================================
# 1. ARCHITECTURE & STATE
# ==========================================

# We define a custom TypedDict for the Graph State.
# - messages: We use Annotated with add_messages to append messages natively in LangGraph.
# - weather_data: A standard dictionary holding wind conditions.
# - body_battery: An integer representing the Garmin health metric.
# - final_recommendation: A string holding the LLM's verdict.
class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    weather_data: Dict[str, Any]
    body_battery: int
    final_recommendation: str

# ==========================================
# 2. THE TOOLS (Mocks/Wrappers)
# ==========================================

@tool
def check_kitesurf_weather() -> Dict[str, Any]:
    """
    Checks the local weather conditions for kitesurfing in Burgas, Bulgaria.
    Uses the Open-Meteo API to fetch wind speed (in knots) and wind direction.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 42.53,
        "longitude": 27.49,
        "current": "wind_speed_10m,wind_direction_10m",
        "wind_speed_unit": "kn", # Request wind speed directly in knots
        "timezone": "auto"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})
        return {
            "wind_speed_knots": current.get("wind_speed_10m"),
            "wind_direction_degrees": current.get("wind_direction_10m")
        }
    except Exception as e:
        return {"error": f"Failed to fetch weather data: {str(e)}"}

@tool
def check_garmin_health() -> int:
    """
    Checks the user's Garmin health data, specifically the Body Battery.
    Returns a mocked value for now.
    """
    # TODO: Drop garminconnect Python logic here later.
    # Example for future integration:
    # client = Garmin("email", "password")
    # client.login()
    # return client.get_body_battery()
    return 45

# ==========================================
# 3. THE NODES
# ==========================================

def data_gatherer_node(state: GraphState) -> GraphState:
    """
    Calls both tools to populate the weather_data and body_battery fields.
    """
    # Invoke tools. Since they take no arguments, we can pass an empty dict.
    weather = check_kitesurf_weather.invoke({})
    battery = check_garmin_health.invoke({})
    
    # Return a partial state to update the overall graph state
    return {
        "weather_data": weather,
        "body_battery": battery
    }

def reasoning_node(state: GraphState) -> GraphState:
    """
    Initializes an LLM, reads the state data, and generates a final verdict.
    """
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=0.7)
    
    weather = state.get("weather_data", {})
    battery = state.get("body_battery", 0)
    
    # Construct the message payload for Gemini
    system_instruction = "You are an Endurance & Adventure Coach. You provide concise recommendations on whether the user should kitesurf or rest based on wind data and body battery."
    human_instruction = (
        f"Wind Conditions: {weather}\n"
        f"Current Body Battery: {battery}/100\n"
        "Given this data, should I hit the water or rest? Give me a concise verdict."
    )
    
    messages = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=human_instruction)
    ]
    
    # Invoke the model
    response = llm.invoke(messages)
    
    # Return the recommendation and optionally append the conversation to messages
    return {
        "final_recommendation": response.content,
        "messages": messages + [response]
    }

# ==========================================
# 4. THE CONTROL FLOW
# ==========================================

# Initialize StateGraph with our custom TypedDict
workflow = StateGraph(GraphState)

# Add our custom nodes
workflow.add_node("Data_Gatherer_Node", data_gatherer_node)
workflow.add_node("Reasoning_Node", reasoning_node)

# Define the control flow edges
workflow.add_edge(START, "Data_Gatherer_Node")
workflow.add_edge("Data_Gatherer_Node", "Reasoning_Node")
workflow.add_edge("Reasoning_Node", END)

# Compile the graph
app = workflow.compile()

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # Ensure GOOGLE_API_KEY is present
    if "GOOGLE_API_KEY" not in os.environ:
        print("WARNING: GOOGLE_API_KEY environment variable is missing!")
        print("Please export it before running: export GOOGLE_API_KEY='your_api_key'")
    
    print("--- Starting AI Adventure Assistant Pipeline ---")
    
    # Initialize the starting state
    initial_state = {
        "messages": [],
        "weather_data": {},
        "body_battery": 0,
        "final_recommendation": ""
    }
    
    # Invoke the graph
    print("Gathering data and consulting the Endurance & Adventure Coach...")
    final_state = app.invoke(initial_state)
    
    print("\n=== Final Recommendation ===")
    print(final_state.get("final_recommendation"))
    print("============================")

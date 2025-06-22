from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_community.llms import Ollama, ollama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b:int):
    """This is an addition function that adds two functions together"""
    return a + b

tools = [add]
model = ChatGoogleGenerativeAI(
    model= "gemini-2.5-flash",
    google_api_key=api_key,
).bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + list(state["messages"]))
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    # Check if the last message is a tool call or requests tool usage
    if hasattr(last_message, "tool_calls") and getattr(last_message, "tool_calls", None):
        return "continue"
    else:
        return "end"
    

graph = StateGraph(AgentState)

graph.add_node("our_agent", model_call)
graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END 
    },
)


graph.add_node("tools", ToolNode(tools))
graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {
    "messages": [
        (
            "user",
            "Add 40 + 30."
        )
    ]
}
print_stream(app.stream(inputs, stream_mode="values"))
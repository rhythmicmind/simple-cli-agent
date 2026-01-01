import operator
import re
from typing import List, Literal
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from tools import calc, utc_now

TOOLS_BY_NAME = {t.name: t for t in [calc, utc_now]}


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    steps: int


def _mk_ai_tool_call(content: str, name: str, args: dict, call_id: str = "1") -> AIMessage:
    # Store tool call in additional_kwargs so .tool_calls works across versions
    return AIMessage(
        content=content,
        additional_kwargs={"tool_calls": [{"id": call_id, "name": name, "args": args}]},
    )


def assistant(state: AgentState):
    last = state["messages"][-1]

    # If we just got a tool result, produce final answer
    if isinstance(last, ToolMessage):
        return {
            "messages": [AIMessage(content=f"Result: {last.content}")],
            "steps": state.get("steps", 0) + 1,
        }

    # Otherwise, decide tool based on user text
    if isinstance(last, HumanMessage):
        text = (last.content or "").strip()
        lower = text.lower()

        if "utc" in lower or "time" in lower:
            return {
                "messages": [_mk_ai_tool_call("Calling utc_now()", "utc_now", {})],
                "steps": state.get("steps", 0) + 1,
            }

        # Extract a math-like expression
        m = re.findall(r"[0-9\+\-\*\/\(\)\.\s]+", text)
        expr = (m[0].strip() if m else "")
        if any(ch.isdigit() for ch in expr) and any(op in expr for op in "+-*/"):
            return {
                "messages": [_mk_ai_tool_call(f"Calling calc({expr})", "calc", {"expression": expr})],
                "steps": state.get("steps", 0) + 1,
            }

        return {
            "messages": [AIMessage(content="Try a math expression (e.g., 12*(7+3)/2) or ask for UTC time.")],
            "steps": state.get("steps", 0) + 1,
        }

    return {"messages": [AIMessage(content="Send a question.")], "steps": state.get("steps", 0) + 1}


def run_tools(state: AgentState):
    ai = state["messages"][-1]
    tool_calls = ai.additional_kwargs.get("tool_calls", []) if hasattr(ai, "additional_kwargs") else []

    msgs: List[ToolMessage] = []
    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("args", {}) or {}
        call_id = str(tc.get("id") or name)

        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            obs = f"Error: unknown tool '{name}'"
        else:
            try:
                obs = tool.invoke(args)
            except Exception as e:
                obs = f"Tool error: {e}"

        msgs.append(ToolMessage(content=str(obs), tool_call_id=call_id))

    return {"messages": msgs}


def route(state: AgentState) -> Literal["tools", "end"]:
    if state.get("steps", 0) >= 6:
        return "end"
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.additional_kwargs.get("tool_calls"):
        return "tools"
    return "end"


def build_graph(max_steps: int = 6):
    g = StateGraph(AgentState)
    g.add_node("assistant", assistant)
    g.add_node("tools", run_tools)

    g.add_edge(START, "assistant")
    g.add_conditional_edges("assistant", route, {"tools": "tools", "end": END})
    g.add_edge("tools", "assistant")

    return g.compile()

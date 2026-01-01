# Simple CLI Agent

Python sandbox for agentic patterns: tool use via LangChain and orchestration via LangGraph (simple CLI agent).

This repo currently runs **fully offline** using a **rule-based router** (no LLM/API key required) while still demonstrating:
- **LangGraph** orchestration (nodes/edges + loop)
- **Tool execution** (`calc`, `utc_now`)
- A basic **step limit** guardrail

Once an approved LLM provider/API key is available, the router can be upgraded to **LLM-driven tool calling** with minimal changes.

---

## What it does

You type a prompt in the terminal. The agent:
1) stores your input in `state.messages`
2) decides if a tool is needed (rule-based router for now)
3) routes to a `tools` node to execute the tool
4) returns the tool output and prints a final response

Example:
- Input: `what time is it in utc?`
- Tool called: `utc_now()`
- Output: `Result: 2026-01-01T08:38:39...+00:00`

---

## Files (what each one is for)

- **`main.py`** — CLI entry point  
  Starts the CLI, reads user input, invokes the graph, prints the final assistant message.

- **`graph.py`** — LangGraph workflow (agent control loop)  
  Defines the state (`messages`, `steps`), defines nodes (`assistant`, `tools`), routes between them, and compiles the graph via `build_graph()`.

- **`tools.py`** — Tools callable by the agent  
  Implements:
  - `calc(expression: str)` — basic arithmetic evaluation
  - `utc_now()` — current UTC timestamp (ISO format)

- **`requirements.txt`** — Python dependencies  
  Installs LangChain/LangGraph and supporting libraries.

- **`.env.example`** — Environment variable template (for later LLM mode)  
  Shows where provider keys/model names would go. Do **not** commit real keys.

---

## Setup & Run (Windows CMD)

### 0) Go into the repo folder
```bat
cd C:\Users\sriad\simple-cli-agent

from datetime import datetime, timezone
from langchain.tools import tool

@tool
def calc(expression: str) -> str:
    """Evaluate a basic arithmetic expression: digits, + - * / ( ) . and spaces."""
    allowed = set("0123456789+-*/(). ")
    if any(ch not in allowed for ch in expression):
        return "Error: expression contains disallowed characters."
    try:
        value = eval(expression, {"__builtins__": {}}, {})
        return str(value)
    except Exception as e:
        return f"Error: {e}"

@tool
def utc_now() -> str:
    """Return the current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()

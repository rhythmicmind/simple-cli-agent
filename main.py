from langchain_core.messages import HumanMessage
from graph import build_graph

def main():
    agent = build_graph(max_steps=6)

    history = []
    print("Simple CLI Agent (rule-based router) — type 'exit' to quit\n")

    while True:
        user = input("You: ").strip()
        if not user or user.lower() == "exit":
            break

        history.append(HumanMessage(content=user))
        out = agent.invoke({"messages": history, "steps": 0})
        history = out["messages"]

        for msg in reversed(history):
            if getattr(msg, "type", None) == "ai":
                print(f"Assistant: {msg.content}\n")
                break

if __name__ == "__main__":
    main()

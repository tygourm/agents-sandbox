from typing import cast

from langchain_core.messages import AIMessage, HumanMessage

from agents_sandbox.agent import agent


def test_agent() -> None:
    result = agent.invoke({"messages": [HumanMessage("Hello, Agent!")]})
    assert cast("HumanMessage", result["messages"][0]).content == "Hello, Agent!"
    assert cast("AIMessage", result["messages"][1]).content == "Hello, World!"

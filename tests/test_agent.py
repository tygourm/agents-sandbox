from typing import cast

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage


def test_agent() -> None:
    model = GenericFakeChatModel(messages=iter([AIMessage("Hello, World!")]))
    result = create_agent(model).invoke({"messages": [HumanMessage("Hello, Agent!")]})
    assert cast("HumanMessage", result["messages"][0]).content == "Hello, Agent!"
    assert cast("AIMessage", result["messages"][1]).content == "Hello, World!"

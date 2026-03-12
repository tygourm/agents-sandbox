from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from agents_sandbox.langgraph import math_toolbox, search_toolbox
from agents_sandbox.langgraph.workflows.models import create_model

load_dotenv()


def main() -> None:
    agent = create_agent(create_model(), [*math_toolbox, *search_toolbox])
    content = "What is 123456 + 456789 - 123456789 ? Use your tools."
    for mode, data in agent.stream(
        {"messages": [HumanMessage(content)]},
        stream_mode=["values", "messages"],
    ):
        print(f"{mode} {data}\n")  # noqa: T201


if __name__ == "__main__":
    main()

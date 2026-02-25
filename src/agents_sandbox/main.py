from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agents_sandbox.tools import math_toolbox, search_toolbox


def main() -> None:
    load_dotenv()

    model = ChatOpenAI(
        model="openai/gpt-oss-120b",  # ty: ignore[unknown-argument]
        base_url="https://albert.api.etalab.gouv.fr/v1",  # ty: ignore[unknown-argument]
    )
    agent = create_agent(model, [*math_toolbox, *search_toolbox])

    content = "What is 123456 + 456789 - 123456789 ? Use your tools."
    for mode, chunk in agent.stream(
        {"messages": [HumanMessage(content)]},
        stream_mode=["updates", "messages"],
    ):
        print(f"{mode} {chunk}\n")  # noqa: T201


if __name__ == "__main__":
    main()

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState, StateGraph

from .models import create_model


class State(MessagesState):
    context: list[str]
    prompt: str


RAG_PROMPT = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

--------------------
Question: {question}
--------------------

------------------
Context: {context}
------------------

Answer:
"""


def retrieve(state: State) -> dict[str, list[str]]:  # noqa: ARG001
    context = [
        "Brest est une ville super.",
        "Lorient est une ville pas si mal.",
        "Rennes ressemble beaucoup trop à Paris.",
    ]
    return {"context": context}


def augment(state: State) -> dict[str, str]:
    question = next(m for m in reversed(state["messages"]) if m.type == "human").content
    context = "".join("\n\n" + chunk for chunk in state["context"])
    prompt = RAG_PROMPT.format(question=question, context=context)
    return {"prompt": prompt}


def generate(state: State) -> dict[str, list[AnyMessage]]:
    return {"messages": [create_model().invoke(state["prompt"])]}


workflow = StateGraph(State)  # ty: ignore[invalid-argument-type]

workflow.add_node("retrieve", retrieve)
workflow.add_node("augment", augment)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "augment")
workflow.add_edge("augment", "generate")
workflow.set_finish_point("generate")

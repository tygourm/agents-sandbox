from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph import MessagesState, StateGraph

from .models import create_model


class State(MessagesState):
    joke: str
    poem: str
    story: str
    topic: str


TOPIC_PROMPT = """You are an assistant for topic identification.
Identify the topic of the following content: {content}
Topic:
"""

JOKE_PROMPT = """You are an assistant for joke generation.
Write a short joke about the following topic: {topic}
Joke:
"""

POEM_PROMPT = """You are an assistant for poem generation.
Write a short poem about the following topic: {topic}
Poem:
"""

STORY_PROMPT = """You are an assistant for story generation.
Write a short story about the following topic: {topic}
Story:
"""


def topic(state: State) -> dict[str, str]:
    content = next(m for m in reversed(state["messages"]) if m.type == "human").content
    prompt = TOPIC_PROMPT.format(content=content)
    content = create_model().invoke(prompt).content
    return {"topic": str(content)}


def joke(state: State) -> dict[str, str]:
    prompt = JOKE_PROMPT.format(topic=state["topic"])
    content = create_model().invoke(prompt).content
    return {"joke": str(content)}


def poem(state: State) -> dict[str, str]:
    prompt = POEM_PROMPT.format(topic=state["topic"])
    content = create_model().invoke(prompt).content
    return {"poem": str(content)}


def story(state: State) -> dict[str, str]:
    prompt = STORY_PROMPT.format(topic=state["topic"])
    content = create_model().invoke(prompt).content
    return {"story": str(content)}


def combine(state: State) -> dict[str, list[AnyMessage]]:
    output = f"""{state["topic"]}

# Joke
{state["joke"]}

# Poem
{state["poem"]}

# Story
{state["story"]}
"""
    return {"messages": [AIMessage(content=output)]}


workflow = StateGraph(State)  # ty: ignore[invalid-argument-type]

workflow.add_node("topic", topic)
workflow.add_node("joke", joke)
workflow.add_node("poem", poem)
workflow.add_node("story", story)
workflow.add_node("combine", combine)

workflow.set_entry_point("topic")
workflow.add_edge("topic", "joke")
workflow.add_edge("topic", "poem")
workflow.add_edge("topic", "story")
workflow.add_edge("joke", "combine")
workflow.add_edge("poem", "combine")
workflow.add_edge("story", "combine")
workflow.set_finish_point("combine")

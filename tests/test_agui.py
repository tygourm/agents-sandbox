from itertools import pairwise
from uuid import uuid4

from ag_ui.core import (
    Event,
    MessagesSnapshotEvent,
    RunAgentInput,
    RunFinishedEvent,
    RunStartedEvent,
    StepFinishedEvent,
    StepStartedEvent,
    UserMessage,
)
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agents_sandbox.agents import AguiAgent


async def test_generic() -> None:
    input_data = RunAgentInput(
        thread_id=str(uuid4()),
        run_id=str(uuid4()),
        parent_run_id=str(uuid4()),
        state={},
        messages=[UserMessage(id=str(uuid4()), content="Hello, Agent!")],
        tools=[],
        context=[],
        forwarded_props={},
    )
    model = GenericFakeChatModel(messages=iter([AIMessage("Hello, World!")]))
    events = [e async for e in AguiAgent(create_agent(model)).run(input_data)]

    assert len(events) == 5
    assert_timestamps(events)
    assert_raw_events(events)

    assert isinstance(events[0], RunStartedEvent)
    assert events[0].thread_id == input_data.thread_id
    assert events[0].run_id == input_data.run_id
    assert events[0].parent_run_id == input_data.parent_run_id
    assert events[0].input == input_data

    assert isinstance(events[1], StepStartedEvent)
    assert events[1].step_name == "model"

    assert isinstance(events[2], StepFinishedEvent)
    assert events[2].step_name == "model"

    assert isinstance(events[3], MessagesSnapshotEvent)
    assert len(events[3].messages) == 2
    assert (
        events[3].messages[0].role == "user"
        and events[3].messages[0].content == "Hello, Agent!"
    )
    assert (
        events[3].messages[1].role == "assistant"
        and events[3].messages[1].content == "Hello, World!"
    )

    assert isinstance(events[4], RunFinishedEvent)
    assert events[4].thread_id == input_data.thread_id
    assert events[4].run_id == input_data.run_id


def assert_timestamps(events: list[Event]) -> None:
    timestamps = [e.timestamp for e in events]
    assert all(t for t in timestamps) and all(
        t1 < t2 for t1, t2 in pairwise(timestamps)
    )


def assert_raw_events(events: list[Event]) -> None:
    assert all(e.raw_event for e in events)

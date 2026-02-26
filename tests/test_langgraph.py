from collections.abc import AsyncGenerator
from itertools import pairwise
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
from ag_ui.core import Event, EventType, RunAgentInput
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.runnables.schema import StreamEvent
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pytest_mock import MockerFixture

from agents_sandbox.langgraph import LangGraphAgent


@pytest.mark.parametrize(
    ("file", "events"),
    [
        (
            "generic.txt",
            [
                EventType.RUN_STARTED,
                EventType.MESSAGES_SNAPSHOT,
                EventType.STEP_STARTED,
                EventType.TEXT_MESSAGE_START,
                *[EventType.TEXT_MESSAGE_CONTENT] * 3,
                EventType.TEXT_MESSAGE_END,
                EventType.STEP_FINISHED,
                EventType.MESSAGES_SNAPSHOT,
                EventType.RUN_FINISHED,
            ],
        ),
        (
            "hello.txt",
            [
                EventType.RUN_STARTED,
                EventType.MESSAGES_SNAPSHOT,
                EventType.STEP_STARTED,
                EventType.TEXT_MESSAGE_START,
                *[EventType.TEXT_MESSAGE_CONTENT] * 9,
                EventType.TEXT_MESSAGE_END,
                EventType.STEP_FINISHED,
                EventType.MESSAGES_SNAPSHOT,
                EventType.RUN_FINISHED,
            ],
        ),
        (
            "tools.txt",
            [
                EventType.RUN_STARTED,
                EventType.MESSAGES_SNAPSHOT,
                EventType.STEP_STARTED,
                EventType.TOOL_CALL_START,
                *[EventType.TOOL_CALL_ARGS] * 16,
                EventType.TOOL_CALL_END,
                EventType.STEP_FINISHED,
                EventType.STEP_STARTED,
                EventType.TOOL_CALL_RESULT,
                EventType.STEP_FINISHED,
                EventType.STEP_STARTED,
                EventType.TOOL_CALL_START,
                *[EventType.TOOL_CALL_ARGS] * 12,
                EventType.TOOL_CALL_END,
                EventType.STEP_FINISHED,
                EventType.STEP_STARTED,
                EventType.TOOL_CALL_RESULT,
                EventType.STEP_FINISHED,
                EventType.STEP_STARTED,
                EventType.TEXT_MESSAGE_START,
                *[EventType.TEXT_MESSAGE_CONTENT] * 8,
                EventType.TEXT_MESSAGE_END,
                EventType.STEP_FINISHED,
                EventType.MESSAGES_SNAPSHOT,
                EventType.RUN_FINISHED,
            ],
        ),
        (
            "sequential.txt",
            [
                EventType.RUN_STARTED,
                EventType.MESSAGES_SNAPSHOT,
                EventType.STEP_STARTED,
                EventType.STEP_FINISHED,
                EventType.STEP_STARTED,
                EventType.STEP_FINISHED,
                EventType.STEP_STARTED,
                EventType.TEXT_MESSAGE_START,
                *[EventType.TEXT_MESSAGE_CONTENT] * 63,
                EventType.TEXT_MESSAGE_END,
                EventType.STEP_FINISHED,
                EventType.MESSAGES_SNAPSHOT,
                EventType.RUN_FINISHED,
            ],
        ),
    ],
)
async def test_stream(
    file: str,
    events: list[EventType],
    mocker: MockerFixture,
) -> None:
    async def _stream() -> AsyncGenerator[StreamEvent, None]:
        path = Path(__file__).parent / "streams" / file
        context = {
            "Command": Command,
            "AIMessage": AIMessage,
            "ToolMessage": ToolMessage,
            "HumanMessage": HumanMessage,
            "AIMessageChunk": AIMessageChunk,
        }
        for e in [
            cast("StreamEvent", eval(line.strip(), context))  # noqa: S307
            for line in path.read_text().split("\n")
            if line.strip()
        ]:
            yield e

    graph = mocker.MagicMock(spec=CompiledStateGraph)
    graph.name = "LangGraph"
    graph.astream_events.return_value = _stream()

    input_data = RunAgentInput(
        thread_id=str(uuid4()),
        run_id=str(uuid4()),
        parent_run_id=str(uuid4()),
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props={},
    )
    result = [e async for e in LangGraphAgent(graph).run(input_data)]
    assert events == [e.type for e in result]
    _assert_timestamps(result)
    _assert_raw_events(result)


def _assert_timestamps(events: list[Event]) -> None:
    timestamps = [e.timestamp for e in events]
    assert all(t for t in timestamps) and all(
        t1 < t2 for t1, t2 in pairwise(timestamps)
    )


def _assert_raw_events(events: list[Event]) -> None:
    assert all(e.raw_event for e in events)

import time
from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING, cast
from uuid import UUID

from ag_ui.core import (
    Event,
    MessagesSnapshotEvent,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui_langgraph.utils import agui_messages_to_langchain, langchain_messages_to_agui
from langchain_core.runnables.schema import StreamEvent
from langgraph.graph.state import CompiledStateGraph

if TYPE_CHECKING:
    from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
    from langchain_core.runnables.config import RunnableConfig


class LangGraphAgent:
    def __init__(self, graph: CompiledStateGraph) -> None:
        self._graph = graph
        self._running = False
        self._messages: dict[str, AIMessage] = {}

    async def run(self, input_data: RunAgentInput) -> AsyncGenerator[Event]:
        try:
            async for event in self._stream(input_data):
                yield event
        except Exception as e:
            if self._running:
                yield RunErrorEvent(timestamp=time.time_ns(), message=str(e))
            raise

    async def _stream(self, input_data: RunAgentInput) -> AsyncGenerator[Event]:
        state = {"messages": agui_messages_to_langchain(input_data.messages)}
        config: RunnableConfig = {
            "run_id": UUID(input_data.run_id),
            "configurable": {"thread_id": input_data.thread_id},
        }

        async for e in self._graph.astream_events(state, config):
            for event in self._handle_event(e, input_data):
                event.timestamp = time.time_ns()
                event.raw_event = e
                yield event

    def _handle_event(
        self,
        e: StreamEvent,
        input_data: RunAgentInput,
    ) -> Generator[Event]:
        match e["event"]:
            case "on_chain_start":
                yield from self._handle_chain_start(e, input_data)
            case "on_chain_end":
                yield from self._handle_chain_end(e, input_data)
            case "on_chat_model_stream":
                yield from self._handle_chat_model_stream(e)
            case "on_chat_model_end":
                yield from self._handle_chat_model_end(e)
            case "on_tool_end":
                yield from self._handle_tool_end(e)

    def _handle_chain_start(
        self,
        e: StreamEvent,
        input_data: RunAgentInput,
    ) -> Generator[Event]:
        if (name := e["name"]) == self._graph.name:
            self._running = True
            yield RunStartedEvent(
                thread_id=input_data.thread_id,
                run_id=input_data.run_id,
                parent_run_id=input_data.parent_run_id,
                input=input_data,
            )
            messages = langchain_messages_to_agui(e["data"]["input"]["messages"])
            yield MessagesSnapshotEvent(messages=messages)
        else:
            yield StepStartedEvent(step_name=name)

    def _handle_chain_end(
        self,
        e: StreamEvent,
        input_data: RunAgentInput,
    ) -> Generator[Event]:
        if (name := e["name"]) == self._graph.name:
            messages = langchain_messages_to_agui(e["data"]["output"]["messages"])
            yield MessagesSnapshotEvent(messages=messages)
            yield RunFinishedEvent(
                thread_id=input_data.thread_id,
                run_id=input_data.run_id,
                result=e["data"]["output"],
            )
            self._running = False
        else:
            yield StepFinishedEvent(step_name=name)

    def _handle_chat_model_stream(self, e: StreamEvent) -> Generator[Event]:  # noqa: C901
        chunk = cast("AIMessageChunk", e["data"]["chunk"])
        if not (message_id := chunk.id):
            return
        if chunk.content:
            if message_id not in self._messages:
                yield TextMessageStartEvent(message_id=message_id)
                self._messages[message_id] = chunk
            else:
                self._messages[message_id] += chunk
            content = chunk.content
            yield TextMessageContentEvent(
                message_id=message_id,
                delta=content if isinstance(content, str) else "",
            )
        elif chunk.tool_call_chunks:
            if message_id not in self._messages:
                for tc in chunk.tool_call_chunks:
                    if not (tool_call_id := tc["id"]) or not (
                        tool_call_name := tc["name"]
                    ):
                        continue
                    yield ToolCallStartEvent(
                        tool_call_id=tool_call_id,
                        tool_call_name=tool_call_name,
                        parent_message_id=message_id,
                    )
                self._messages[message_id] = chunk
            else:
                for tc in chunk.tool_call_chunks:
                    if not (message := self._messages.get(message_id)):
                        continue
                    if not (
                        tool_call_id := message.tool_calls[tc["index"] or 0]["id"]
                    ) or not (delta := tc["args"]):
                        continue
                    yield ToolCallArgsEvent(tool_call_id=tool_call_id, delta=delta)

    def _handle_chat_model_end(self, e: StreamEvent) -> Generator[Event]:
        if not (message := cast("AIMessage", e["data"]["output"])):
            return
        if not (message_id := message.id):
            return
        if message.content:
            yield TextMessageEndEvent(message_id=message_id)
        elif message.tool_calls:
            for tc in message.tool_calls:
                if tool_call_id := tc["id"]:
                    yield ToolCallEndEvent(tool_call_id=tool_call_id)

    def _handle_tool_end(self, e: StreamEvent) -> Generator[Event]:
        if not (tool_call := cast("ToolMessage", e["data"]["output"])):
            return
        if not (tool_call_id := tool_call.tool_call_id):
            return
        if not (
            message := next(
                (
                    m
                    for m in self._messages.values()
                    if m.tool_calls
                    and any(tc["id"] == tool_call_id for tc in m.tool_calls)
                ),
                None,
            )
        ):
            return
        if not (message_id := message.id):
            return
        content = tool_call.content
        yield ToolCallResultEvent(
            message_id=message_id,
            tool_call_id=tool_call_id,
            content=content if isinstance(content, str) else "",
        )

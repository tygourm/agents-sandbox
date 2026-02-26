import time
from asyncio import Queue
from collections.abc import AsyncGenerator
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
        self._messages: list[AIMessage] = []
        self._event_queue: Queue[Event] = Queue()

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
            if e["event"] == "on_chain_start":
                self._handle_chain_start(e, input_data)
            elif e["event"] == "on_chain_end":
                self._handle_chain_end(e, input_data)
            elif e["event"] == "on_chat_model_stream":
                self._handle_chat_model_stream(e)
            elif e["event"] == "on_chat_model_end":
                self._handle_chat_model_end(e)
            elif e["event"] == "on_tool_end":
                self._handle_tool_end(e)

            while not self._event_queue.empty():
                event = await self._event_queue.get()
                event.timestamp = time.time_ns()
                event.raw_event = e
                yield event

    def _handle_chain_start(self, e: StreamEvent, input_data: RunAgentInput) -> None:
        if (name := e["name"]) == self._graph.name:
            self._running = True
            self._event_queue.put_nowait(
                RunStartedEvent(
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                    parent_run_id=input_data.parent_run_id,
                    input=input_data,
                )
            )
            messages = langchain_messages_to_agui(e["data"]["input"]["messages"])
            self._event_queue.put_nowait(MessagesSnapshotEvent(messages=messages))
        else:
            self._event_queue.put_nowait(StepStartedEvent(step_name=name))

    def _handle_chain_end(self, e: StreamEvent, input_data: RunAgentInput) -> None:
        if (name := e["name"]) == self._graph.name:
            messages = langchain_messages_to_agui(e["data"]["output"]["messages"])
            self._event_queue.put_nowait(MessagesSnapshotEvent(messages=messages))
            self._event_queue.put_nowait(
                RunFinishedEvent(
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                    result=e["data"]["output"],
                )
            )
            self._running = False
        else:
            self._event_queue.put_nowait(StepFinishedEvent(step_name=name))

    def _handle_chat_model_stream(self, e: StreamEvent) -> None:  # noqa: C901, PLR0912
        chunk = cast("AIMessageChunk", e["data"]["chunk"])
        if not (message_id := chunk.id):
            return
        if len(chunk.content) > 0:
            if not any(m.id == message_id for m in self._messages):
                self._event_queue.put_nowait(
                    TextMessageStartEvent(message_id=message_id)
                )
                self._messages.append(chunk)
            else:
                for i, m in enumerate(self._messages):
                    if m.id == message_id:
                        self._messages[i] += chunk
                        break
            self._event_queue.put_nowait(
                TextMessageContentEvent(
                    message_id=message_id,
                    delta=chunk.content if isinstance(chunk.content, str) else "",
                )
            )
        elif len(chunk.tool_call_chunks) > 0:
            if not any(m.id == message_id for m in self._messages):
                for tc in chunk.tool_call_chunks:
                    if not (tool_call_id := tc["id"]) or not (
                        tool_call_name := tc["name"]
                    ):
                        continue
                    self._event_queue.put_nowait(
                        ToolCallStartEvent(
                            tool_call_id=tool_call_id,
                            tool_call_name=tool_call_name,
                            parent_message_id=message_id,
                        )
                    )
                self._messages.append(chunk)
            else:
                for tc in chunk.tool_call_chunks:
                    if not (
                        message := next(
                            (m for m in self._messages if m.id == message_id), None
                        )
                    ):
                        continue
                    if not (
                        tool_call_id := message.tool_calls[tc["index"] or 0]["id"]
                    ) or not (delta := tc["args"]):
                        continue
                    self._event_queue.put_nowait(
                        ToolCallArgsEvent(tool_call_id=tool_call_id, delta=delta)
                    )

    def _handle_chat_model_end(self, e: StreamEvent) -> None:
        if not (message := cast("AIMessage", e["data"]["output"])):
            return
        if not (message_id := message.id):
            return
        if len(message.content) > 0:
            self._event_queue.put_nowait(TextMessageEndEvent(message_id=message_id))
        elif len(message.tool_calls) > 0:
            for tc in message.tool_calls:
                if not (tool_call_id := tc["id"]):
                    continue
                self._event_queue.put_nowait(
                    ToolCallEndEvent(tool_call_id=tool_call_id)
                )

    def _handle_tool_end(self, e: StreamEvent) -> None:
        if not (tool_call := cast("ToolMessage", e["data"]["output"])):
            return
        if not (tool_call_id := tool_call.tool_call_id):
            return
        if not (
            message := next(
                (
                    m
                    for m in self._messages
                    if m.tool_calls
                    and any(tc["id"] == tool_call_id for tc in m.tool_calls)
                ),
                None,
            )
        ):
            return
        if not (message_id := message.id):
            return
        self._event_queue.put_nowait(
            ToolCallResultEvent(
                message_id=message_id,
                tool_call_id=tool_call_id,
                content=tool_call.content if isinstance(tool_call.content, str) else "",
            )
        )

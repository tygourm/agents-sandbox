import time
from collections.abc import AsyncGenerator
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
)
from ag_ui_langgraph.utils import agui_messages_to_langchain, langchain_messages_to_agui
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph


class AguiAgent:
    def __init__(self, graph: CompiledStateGraph) -> None:
        self._graph = graph
        self._run_started = False

    async def run(self, input_data: RunAgentInput) -> AsyncGenerator[Event]:

        try:
            async for e in self._stream(input_data):
                yield e
        except Exception as e:
            if self._run_started:
                yield RunErrorEvent(
                    timestamp=time.time_ns(),
                    message=str(e),
                )
            raise

    async def _stream(self, input_data: RunAgentInput) -> AsyncGenerator[Event]:
        config = RunnableConfig(
            run_id=UUID(input_data.run_id),
            configurable={"thread_id": UUID(input_data.thread_id)},
        )
        async for e in self._graph.astream_events(
            {"messages": agui_messages_to_langchain(input_data.messages)},
            config,
        ):
            if e["event"] == "on_chain_start" and e["name"] == self._graph.name:
                self._run_started = True
                yield RunStartedEvent(
                    timestamp=time.time_ns(),
                    raw_event=e,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                    parent_run_id=input_data.parent_run_id,
                    input=input_data,
                )

            elif e["event"] == "on_chain_start" and e["name"] == "model":
                yield StepStartedEvent(
                    timestamp=time.time_ns(),
                    raw_event=e,
                    step_name=e["name"],
                )

            elif e["event"] == "on_chain_end" and e["name"] == "model":
                yield StepFinishedEvent(
                    timestamp=time.time_ns(),
                    raw_event=e,
                    step_name=e["name"],
                )

            elif e["event"] == "on_chain_end" and e["name"] == self._graph.name:
                messages = langchain_messages_to_agui(e["data"]["output"]["messages"])
                yield MessagesSnapshotEvent(
                    timestamp=time.time_ns(),
                    raw_event=e,
                    messages=messages,
                )

                yield RunFinishedEvent(
                    timestamp=time.time_ns(),
                    raw_event=e,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                )

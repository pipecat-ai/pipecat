#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from pipecat.frames.frames import (
    EndFrame,
    InputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.frames import (
    RTVIFlowNodeEnteredFrame,
    RTVIServerMessageFrame,
)
from pipecat.processors.frameworks.rtvi.models import FlowNodeEnteredMessageData, Message
from pipecat.tests.utils import QueuedFrameProcessor
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.websocket.server import (
    SingleClientWebsocketServerParams,
    SingleClientWebsocketServerTransport,
)
from pipecat.workers.runner import WorkerRunner


@pytest.mark.asyncio
async def test_reconnecting_client_gets_current_protocol_until_its_handshake(monkeypatch):
    params = SingleClientWebsocketServerParams()
    transport = SingleClientWebsocketServerTransport(params)
    # Exercise the server's connection callback without opening a socket.
    input_transport = BaseInputTransport(params)
    transport._input = input_transport
    transport._output = Mock(set_client_connection=AsyncMock())
    output = asyncio.Queue()
    sink = QueuedFrameProcessor(queue=output, queue_direction=FrameDirection.DOWNSTREAM)
    worker = PipelineWorker(Pipeline([input_transport, sink]), cancel_on_idle_timeout=False)
    started = asyncio.Event()
    old_message_started = asyncio.Event()
    release_old_message = asyncio.Event()
    replacement = Mock()

    @worker.event_handler("on_pipeline_started")
    async def on_pipeline_started(worker, frame):
        started.set()

    async def hold_old_message(*args):
        old_message_started.set()
        await release_old_message.wait()

    monkeypatch.setattr(worker.rtvi, "_handle_client_message", hold_old_message)

    async def send_message(message):
        await input_transport.broadcast_frame(
            InputTransportMessageFrame, message=message.model_dump()
        )

    async def send_ready(request_id, version):
        await send_message(Message(type="client-ready", id=request_id, data={"version": version}))

    async def read_through(message_type):
        messages = []
        async with asyncio.timeout(2):
            while True:
                frame = await output.get()
                if isinstance(frame, OutputTransportMessageUrgentFrame):
                    messages.append(frame.message)
                    if frame.message["type"] == message_type:
                        return messages

    async def send_flow_node():
        await worker.queue_frames(
            [
                RTVIFlowNodeEnteredFrame(data=FlowNodeEnteredMessageData(node="welcome")),
                RTVIServerMessageFrame(data={"marker": True}),
            ]
        )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, websocket):
        if websocket is replacement:
            await send_flow_node()

    async def exercise_reconnection():
        await asyncio.wait_for(started.wait(), timeout=2)
        await transport._on_client_connected(Mock())
        await send_ready("old", "2.2.0")
        assert (await read_through("bot-ready"))[-1]["id"] == "old"

        await send_flow_node()
        assert [message["type"] for message in await read_through("server-message")] == [
            "server-message"
        ]

        # The connection arrives while the old client's message is still being handled.
        await send_message(Message(type="client-message", id="pending", data={"t": "app"}))
        await asyncio.wait_for(old_message_started.wait(), timeout=2)
        await transport._on_client_connected(replacement)
        assert [message["type"] for message in await read_through("server-message")] == [
            "flow-node-entered",
            "server-message",
        ]
        assert not worker.rtvi.client_ready
        assert worker.rtvi.client_version == [0, 0, 0]

        await send_ready("new", "2.4.0")
        release_old_message.set()
        assert (await read_through("bot-ready"))[-1]["id"] == "new"
        assert worker.rtvi.client_ready
        assert worker.rtvi.client_version == [2, 4, 0]
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner()
    await runner.add_workers(worker)
    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(runner.run())
        tasks.create_task(exercise_reconnection())

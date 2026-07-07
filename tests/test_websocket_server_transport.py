#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Behavioral tests for the single-client WebSocket server transport.

The contract under test: output written before an EndFrame (e.g. a farewell
spoken by the TTS via Flows' `end_conversation` action) must reach the client
before the server closes the shared connection — and the server must still be
torn down once the session ends.

These drive a real transport over a real WebSocket connection and assert on
what the client observes (bytes received, connection closed, server no longer
listening), not on the transport's internal teardown bookkeeping.
"""

import asyncio
import socket
import unittest

import websockets

from pipecat.frames.frames import EndFrame, Frame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.websocket.server import (
    SingleClientWebsocketServerParams,
    SingleClientWebsocketServerTransport,
)
from pipecat.workers.runner import WorkerRunner

SAMPLE_RATE = 16000

# 0.2s of recognizable audio @ 16kHz mono s16 (multiple of the 20ms chunk size).
GOODBYE_AUDIO = b"\x01\x02\x03\x04" * 1600

END_MARKER = b"__END__"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class _RawAudioSerializer(FrameSerializer):
    """Serialize audio frames as their raw bytes and EndFrame as a marker."""

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        if isinstance(frame, EndFrame):
            return END_MARKER
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        return None


def _params() -> SingleClientWebsocketServerParams:
    return SingleClientWebsocketServerParams(
        serializer=_RawAudioSerializer(),
        audio_out_enabled=True,
    )


class WebsocketServerTransportTest(unittest.IsolatedAsyncioTestCase):
    """Base with helpers to run a transport and connect a client to it."""

    async def _serve(self, transport, processors) -> tuple[PipelineWorker, asyncio.Task]:
        """Run `processors` on a worker, returning it and its run task."""
        worker = PipelineWorker(
            Pipeline(processors),
            cancel_on_idle_timeout=False,
            params=PipelineParams(audio_out_sample_rate=SAMPLE_RATE),
        )
        runner = WorkerRunner()
        await runner.add_workers(worker)
        run_task = asyncio.create_task(runner.run())
        self.addAsyncCleanup(runner.cancel)
        self.addCleanup(run_task.cancel)
        return worker, run_task

    async def _connect(self, port: int):
        """Connect a client, retrying until the server is listening."""
        for _ in range(50):
            try:
                return await websockets.connect(f"ws://localhost:{port}")
            except OSError:
                await asyncio.sleep(0.1)
        self.fail(f"could not connect to ws://localhost:{port}")

    async def _assert_not_listening(self, port: int):
        """Assert nothing accepts connections on `port` (server torn down)."""
        with self.assertRaises(OSError):
            await websockets.connect(f"ws://localhost:{port}")


class TestGoodbyeFlushedBeforeClose(WebsocketServerTransportTest):
    """Output queued before an EndFrame must reach the client before close."""

    async def test_output_queued_before_endframe_reaches_client_before_close(self):
        # The farewell scenario: audio (the goodbye) queued before the EndFrame,
        # exactly what Flows' end_conversation action produces. The input sees the
        # EndFrame first, so a premature teardown there would cut the goodbye off.
        port = _free_port()
        transport = SingleClientWebsocketServerTransport(
            params=_params(), host="localhost", port=port
        )
        worker, run_task = await self._serve(transport, [transport.input(), transport.output()])
        client = await self._connect(port)

        received: list[bytes] = []

        async def read_until_closed():
            try:
                async for message in client:
                    if isinstance(message, bytes):
                        received.append(message)
            except websockets.ConnectionClosed:
                pass

        reader_task = asyncio.create_task(read_until_closed())

        # Let on_client_connected propagate to the output transport.
        await asyncio.sleep(0.2)

        await worker.queue_frames(
            [
                OutputAudioRawFrame(audio=GOODBYE_AUDIO, sample_rate=SAMPLE_RATE, num_channels=1),
                EndFrame(),
            ]
        )

        # The session ends and, once the goodbye has been delivered, the server
        # closes the connection — so both the worker and the reader finish.
        await asyncio.wait_for(run_task, 10)
        await asyncio.wait_for(reader_task, 5)
        await client.close()

        # The output may pad trailing silence after the goodbye, so assert the
        # whole goodbye arrived, intact and first (a premature close would leave
        # a truncated prefix).
        audio_received = b"".join(m for m in received if m != END_MARKER)
        self.assertTrue(
            audio_received.startswith(GOODBYE_AUDIO),
            f"goodbye not delivered in full before close (received {len(audio_received)} bytes)",
        )


class TestServerTornDownAfterSession(WebsocketServerTransportTest):
    """The server is drained once the session ends (it must not leak)."""

    async def test_server_stops_listening_after_pipeline_ends(self):
        port = _free_port()
        transport = SingleClientWebsocketServerTransport(
            params=_params(), host="localhost", port=port
        )
        worker, run_task = await self._serve(transport, [transport.input(), transport.output()])
        client = await self._connect(port)

        await asyncio.sleep(0.2)
        await worker.queue_frames([EndFrame()])
        await asyncio.wait_for(run_task, 10)
        await client.close()

        await self._assert_not_listening(port)

    async def test_server_torn_down_without_output_transport(self):
        # With no output transport, the input is the only holder of the shared
        # server, so its EndFrame alone drives the teardown — no separate backstop
        # needed.
        port = _free_port()
        transport = SingleClientWebsocketServerTransport(
            params=_params(), host="localhost", port=port
        )
        worker, run_task = await self._serve(transport, [transport.input()])
        client = await self._connect(port)

        await asyncio.sleep(0.2)
        await worker.queue_frames([EndFrame()])
        await asyncio.wait_for(run_task, 10)
        await client.close()

        await self._assert_not_listening(port)

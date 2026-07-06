#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the single-client WebSocket server transport.

The main contract under test: output written before an EndFrame (e.g. a
farewell spoken by the TTS via Flows' `end_conversation` action) must reach
the client before the server closes the shared connection. The input
transport sees the EndFrame first (it is the first processor in the
pipeline), so it must not tear the server down; the graceful drain belongs
to the output transport's `stop()`, which runs only after all pending
output has been written.
"""

import asyncio
import socket
import unittest
from types import SimpleNamespace

import websockets

from pipecat.frames.frames import EndFrame, Frame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.transports.websocket.server import (
    SingleClientWebsocketServerOutputTransport,
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


class TestInputStopDefersServerTeardown(unittest.IsolatedAsyncioTestCase):
    """The input transport must not drain the server on EndFrame."""

    async def test_input_stop_leaves_server_running(self):
        transport = SingleClientWebsocketServerTransport(
            params=_params(), host="localhost", port=_free_port()
        )
        input_transport = transport.input()

        # Simulate the running server loop: like the real one, it exits when
        # the stop event is set.
        input_transport._server_task = asyncio.create_task(
            input_transport._stop_server_event.wait()
        )

        await input_transport.stop(EndFrame())

        self.assertFalse(input_transport._stop_server_event.is_set())
        self.assertIsNotNone(input_transport._server_task)

        # stop_server() is where the graceful drain actually happens.
        await input_transport.stop_server()

        self.assertTrue(input_transport._stop_server_event.is_set())
        self.assertIsNone(input_transport._server_task)


class TestOutputStopOrdering(unittest.IsolatedAsyncioTestCase):
    """The output transport writes the final frame before draining the server."""

    async def test_output_stop_writes_before_server_drain(self):
        calls = []

        async def record_stop_server():
            calls.append("stop_server")

        input_stub = SimpleNamespace(stop_server=record_stop_server)
        transport_stub = SimpleNamespace(input=lambda: input_stub)

        output = SingleClientWebsocketServerOutputTransport(transport_stub, _params())

        async def record_write_frame(frame):
            calls.append("write")

        output._write_frame = record_write_frame

        await output.stop(EndFrame())

        self.assertEqual(calls, ["write", "stop_server"])


class TestGoodbyeFlushedBeforeClose(unittest.IsolatedAsyncioTestCase):
    """End-to-end regression test for the farewell cutoff bug.

    Audio queued before an EndFrame must reach the connected client before
    the server closes the connection. Before the fix, the input transport
    tore the server down as soon as the EndFrame passed it, so the client
    connection died while the audio was still being written downstream.
    """

    async def test_audio_queued_before_endframe_reaches_client(self):
        port = _free_port()
        transport = SingleClientWebsocketServerTransport(
            params=_params(), host="localhost", port=port
        )
        pipeline = Pipeline([transport.input(), transport.output()])
        worker = PipelineWorker(
            pipeline,
            cancel_on_idle_timeout=False,
            params=PipelineParams(audio_out_sample_rate=SAMPLE_RATE),
        )
        runner = WorkerRunner()
        await runner.add_workers(worker)
        run_task = asyncio.create_task(runner.run())
        self.addAsyncCleanup(runner.cancel)
        self.addCleanup(run_task.cancel)
        # Connect a client, retrying until the server is listening.
        client = None
        for _ in range(50):
            try:
                client = await websockets.connect(f"ws://localhost:{port}")
                break
            except OSError:
                await asyncio.sleep(0.1)
        self.assertIsNotNone(client, "could not connect to the transport's server")

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

        # The farewell scenario: audio (the goodbye) queued before the
        # EndFrame, exactly what Flows' end_conversation action produces.
        await worker.queue_frames(
            [
                OutputAudioRawFrame(audio=GOODBYE_AUDIO, sample_rate=SAMPLE_RATE, num_channels=1),
                EndFrame(),
            ]
        )

        await asyncio.wait_for(run_task, 10)
        await asyncio.wait_for(reader_task, 5)
        await client.close()

        # The output transport may pad trailing silence after the goodbye, so
        # assert the goodbye itself arrived, intact and first.
        audio_received = b"".join(m for m in received if m != END_MARKER)
        self.assertTrue(
            audio_received.startswith(GOODBYE_AUDIO),
            f"goodbye audio not flushed before close (received {len(audio_received)} bytes)",
        )


if __name__ == "__main__":
    unittest.main()

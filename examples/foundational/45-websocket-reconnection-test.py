#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
WebSocket Reconnection Test Example

This example demonstrates and tests that the on_client_connected and
on_client_disconnected event handlers fire correctly on multiple
reconnections in a long-running container.

This test is useful for verifying the fix for the issue where on_client_disconnected
would stop firing after the first disconnection when the container stays alive.

Usage:
    Run the bot with the WebSocket transport (twilio):

    python examples/foundational/45-websocket-reconnection-test.py --transport twilio

    Then manually connect and disconnect a WebSocket client multiple times.
    You should see the connection/disconnection counts increment correctly.

    Example with websocat:
        # In another terminal:
        websocat ws://localhost:8765
        # Press Ctrl+C to disconnect
        # Repeat multiple times

Expected output on each cycle:
    ✅ CLIENT CONNECTED - Connection #1 | Total Disconnects: 0
    🔌 CLIENT DISCONNECTED - Disconnection #1 | Total Connections: 1
    ✅ CLIENT CONNECTED - Connection #2 | Total Disconnects: 1
    🔌 CLIENT DISCONNECTED - Disconnection #2 | Total Connections: 2
    🎉 SUCCESS: on_client_disconnected fired 2 times! The fix is working!
    ...

If the fix is NOT applied, you'll see:
    ✅ CLIENT CONNECTED - Connection #1 | Total Disconnects: 0
    🔌 CLIENT DISCONNECTED - Disconnection #1 | Total Connections: 1
    ✅ CLIENT CONNECTED - Connection #2 | Total Disconnects: 1
    (no disconnect message on subsequent disconnects)
"""

from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams

# Global counters to track connection events
connection_count = 0
disconnection_count = 0

transport_params = {
    "daily": lambda: DailyParams(audio_out_enabled=False, audio_in_enabled=False),
    "twilio": lambda: FastAPIWebsocketParams(audio_out_enabled=False, audio_in_enabled=False),
    "webrtc": lambda: TransportParams(audio_out_enabled=False, audio_in_enabled=False),
}


class SimpleProcessor(FrameProcessor):
    """A simple pass-through processor."""

    async def process_frame(self, frame, direction):
        await self.push_frame(frame, direction)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    global connection_count, disconnection_count

    logger.info("=" * 80)
    logger.info("🧪 WEBSOCKET RECONNECTION TEST - Long-running container")
    logger.info("=" * 80)
    logger.info("")
    logger.info("This bot will stay alive. Connect and disconnect multiple times")
    logger.info("to verify that on_client_disconnected fires on each disconnect.")
    logger.info("")
    logger.info("Connect with: websocat ws://localhost:8765")
    logger.info("Or use any WebSocket client.")
    logger.info("")
    logger.info("=" * 80)

    task = PipelineTask(
        Pipeline([SimpleProcessor(), transport.output()]),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    # Register event handler for client connection
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        global connection_count
        connection_count += 1
        logger.success("")
        logger.success("=" * 80)
        logger.success(
            f"✅ CLIENT CONNECTED - Connection #{connection_count} | Total Disconnects: {disconnection_count}"
        )
        logger.success("=" * 80)

        # Send a greeting
        greeting = f"Connection number {connection_count}. "
        if connection_count == 1:
            greeting += "Try disconnecting and reconnecting to test the event handlers!"
        else:
            greeting += (
                f"Successfully reconnected! Disconnect count should increment when you disconnect."
            )

        await task.queue_frames([TTSSpeakFrame(greeting), EndFrame()])

    # Register event handler for client disconnection
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        global disconnection_count
        disconnection_count += 1
        logger.success("")
        logger.success("=" * 80)
        logger.success(
            f"🔌 CLIENT DISCONNECTED - Disconnection #{disconnection_count} | Total Connections: {connection_count}"
        )
        logger.success("=" * 80)
        logger.success(
            f"📊 Session Stats - Connections: {connection_count}, Disconnections: {disconnection_count}"
        )

        if disconnection_count > 1:
            logger.success("")
            logger.success("🎉 " * 20)
            logger.success(
                f"🎉 SUCCESS: on_client_disconnected fired {disconnection_count} times! The fix is working!"
            )
            logger.success("🎉 " * 20)
            logger.success("")

        logger.success("=" * 80)
        logger.success("")

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()

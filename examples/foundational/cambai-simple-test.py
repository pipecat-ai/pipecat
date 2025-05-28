#!/usr/bin/env python3
#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""
CambAI TTS Simple Test

A minimal example to test CambAI TTS functionality without requiring
external services like OpenAI or Deepgram. Perfect for testing your
CambAI API key and voice configuration.

Usage:
    python cambai-simple-test.py --api-key YOUR_API_KEY --voice-id 1
    
Or set environment variables:
    export CAMBAI_API_KEY="your_api_key"
    export CAMBAI_VOICE_ID="1"
    python cambai-simple-test.py
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Add pipecat to path if running from examples directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.cambai.tts import CambAITTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

load_dotenv(override=True)


# Transport configuration
transport_params = {
    "daily": lambda: DailyParams(audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_out_enabled=True),
}


async def run_example(transport: BaseTransport, args: argparse.Namespace, handle_sigint: bool):
    logger.info("Starting CambAI TTS simple test")

    # Get API key and voice ID from args or environment
    api_key = args.api_key or os.getenv("CAMBAI_API_KEY")
    voice_id = args.voice_id or int(os.getenv("CAMBAI_VOICE_ID", "1"))

    if not api_key:
        logger.error("CambAI API key is required. Set CAMBAI_API_KEY environment variable or use --api-key")
        return

    logger.info(f"Using CambAI voice ID: {voice_id}")

    # Create CambAI TTS service
    try:
        tts = CambAITTSService(
            api_key=api_key,
            voice_id=voice_id,
            sample_rate=24000,
        )
        logger.info("✓ CambAI TTS service created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create CambAI TTS service: {e}")
        return

    # Create pipeline
    task = PipelineTask(Pipeline([tts, transport.output()]))

    # Test messages to demonstrate different voice characteristics
    test_messages = [
        "Hello! This is a test of the CambAI text-to-speech service.",
        "I can speak with natural intonation and clear pronunciation.",
        "CambAI provides high-quality voice synthesis for your applications.",
        "Thank you for testing the CambAI integration with pipecat!",
    ]

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected - starting TTS test")
        
        # Queue all test messages
        frames = []
        for i, message in enumerate(test_messages, 1):
            logger.info(f"Queuing message {i}: {message[:50]}...")
            frames.append(TTSSpeakFrame(message))
            
        # Add end frame
        frames.append(EndFrame())
        
        await task.queue_frames(frames)
        logger.info(f"Queued {len(test_messages)} test messages")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    try:
        await runner.run(task)
        logger.info("✓ CambAI TTS test completed successfully")
    except Exception as e:
        logger.error(f"✗ Error during TTS test: {e}")


def main():
    parser = argparse.ArgumentParser(description="CambAI TTS Simple Test")
    
    # CambAI-specific arguments
    parser.add_argument(
        "--api-key",
        help="CambAI API key (or set CAMBAI_API_KEY environment variable)"
    )
    parser.add_argument(
        "--voice-id",
        type=int,
        help="CambAI voice ID (or set CAMBAI_VOICE_ID environment variable)"
    )
    
    # Standard pipecat arguments
    parser.add_argument(
        "--host", 
        default="localhost", 
        help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860, 
        help="Port for HTTP server (default: 7860)"
    )
    parser.add_argument(
        "--transport",
        "-t",
        type=str,
        choices=["daily", "webrtc", "twilio"],
        default="webrtc",
        help="The transport this example should use (default: webrtc)",
    )
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="count", 
        default=0,
        help="Increase verbosity"
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove(0)
    log_level = "TRACE" if args.verbose >= 2 else "DEBUG" if args.verbose >= 1 else "INFO"
    logger.add(sys.stderr, level=log_level)

    # Import and run with the specified transport
    if args.transport not in transport_params:
        logger.error(f"Transport '{args.transport}' not supported")
        return

    logger.info(f"Starting CambAI TTS test with {args.transport} transport")
    
    from run import run_main
    run_main(run_example, args, transport_params)


if __name__ == "__main__":
    main()

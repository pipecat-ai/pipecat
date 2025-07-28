#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat Cloud-compatible bot example.

Transports are Daily or SmallWebRTC."""

import os
from dataclasses import dataclass
from typing import Any, Optional

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext

# from pipecat.runner.cloud import SmallWebRTCSessionArguments # Need a release of Pipecat to use this
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService

try:
    from pipecatcloud.agent import DailySessionArguments
except ImportError:
    raise ImportError(
        "pipecatcloud package is required for cloud-compatible bots. "
        "Install with: pip install pipecat-ai[[pipecatcloud]]"
    )

load_dotenv(override=True)


# For now, we'll just define SmallWebRTCSessionArguments here directly since Pipecat
# isn't released with the pipecat.runner.cloud module yet.
# This saves us from having to build a Docker container from my branch or main to
# deploy to PCC.
@dataclass
class SmallWebRTCSessionArguments:
    """Small WebRTC session arguments for local development.

    This will be replaced by pipecatcloud.agent.SmallWebRTCSessionArguments
    when WebRTC support is added to Pipecat Cloud.
    """

    webrtc_connection: Any
    session_id: Optional[str] = None


# Check if we're running locally
IS_LOCAL_RUN = os.environ.get("LOCAL_RUN", "0") == "1"


async def run_bot(transport):
    """Main bot logic that works with any transport."""
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": "You are a friendly AI assistant. Respond naturally and keep your answers conversational.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        messages.append({"role": "system", "content": "Say hello and briefly introduce yourself."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(session_args: DailySessionArguments | SmallWebRTCSessionArguments):
    """Main bot entry point compatible with Pipecat Cloud."""

    if isinstance(session_args, DailySessionArguments):
        from pipecat.transports.services.daily import DailyParams, DailyTransport

        if not IS_LOCAL_RUN:
            from pipecat.audio.filters.krisp_filter import KrispFilter

        transport = DailyTransport(
            session_args.room_url,
            session_args.token,
            "Pipecat Bot",
            params=DailyParams(
                audio_in_enabled=True,
                audio_in_filter=None
                if IS_LOCAL_RUN
                else KrispFilter(),  # Only use Krisp in production
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

    elif isinstance(session_args, SmallWebRTCSessionArguments):
        from pipecat.transports.base_transport import TransportParams
        from pipecat.transports.network.small_webrtc import SmallWebRTCTransport

        transport = SmallWebRTCTransport(
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
            webrtc_connection=session_args.webrtc_connection,
        )

    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.cloud import main

    main()

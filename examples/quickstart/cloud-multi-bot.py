#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.runner.cloud import SmallWebRTCSessionArguments
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService

try:
    from pipecatcloud.agent import DailySessionArguments, WebSocketSessionArguments
except ImportError:
    raise ImportError(
        "pipecatcloud package is required for cloud-compatible bots. "
        "Install with: pip install pipecat-ai[[pipecatcloud]]"
    )

load_dotenv(override=True)


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

    pipeline = Pipeline(
        [
            transport.input(),
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


async def bot(
    session_args: DailySessionArguments | SmallWebRTCSessionArguments | WebSocketSessionArguments,
):
    """Main bot entry point compatible with Pipecat Cloud."""

    if isinstance(session_args, DailySessionArguments):
        from pipecat.transports.services.daily import DailyParams, DailyTransport

        transport = DailyTransport(
            session_args.room_url,
            session_args.token,
            "Pipecat Bot",
            params=DailyParams(
                audio_in_enabled=True,
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

    elif isinstance(session_args, WebSocketSessionArguments):
        from pipecat.transports.network.fastapi_websocket import (
            FastAPIWebsocketParams,
            FastAPIWebsocketTransport,
        )

        # Create base parameters for telephony
        params = FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            add_wav_header=False,
        )

        # Create appropriate serializer based on transport type
        transport_type = getattr(session_args, "transport_type", "unknown")
        call_info = getattr(session_args, "call_info", {})

        if transport_type == "twilio":
            from pipecat.serializers.twilio import TwilioFrameSerializer

            params.serializer = TwilioFrameSerializer(
                stream_sid=call_info["stream_sid"],
                call_sid=call_info["call_sid"],
                account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
                auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
            )
        elif transport_type == "telnyx":
            from pipecat.serializers.telnyx import TelnyxFrameSerializer

            params.serializer = TelnyxFrameSerializer(
                stream_id=call_info["stream_id"],
                call_control_id=call_info["call_control_id"],
                outbound_encoding=call_info["outbound_encoding"],
                inbound_encoding="PCMU",
            )
        elif transport_type == "plivo":
            from pipecat.serializers.plivo import PlivoFrameSerializer

            params.serializer = PlivoFrameSerializer(
                stream_id=call_info["stream_id"],
                call_id=call_info["call_id"],
            )
        else:
            raise ValueError(f"Unsupported WebSocket transport type: {transport_type}")

        transport = FastAPIWebsocketTransport(websocket=session_args.websocket, params=params)

    else:
        raise ValueError(f"Unsupported session arguments type: {type(session_args)}")

    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.cloud import main

    main()

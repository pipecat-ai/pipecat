# Copyright 2025 Vonage
"""Example of using AWS Nova Sonic LLM service with Vonage Video WebRTC transport."""

import asyncio
import json
import os
import sys

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services import aws_nova_sonic
from pipecat.services.aws_nova_sonic.aws import AWSNovaSonicLLMService
from pipecat.transports.vonage.video_webrtc import (
    VonageVideoWebrtcTransport,
    VonageVideoWebrtcTransportParams,
)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main(session_str: str):
    """Main entry point for the nova sonic vonage video webrtc example."""
    system_instruction = (
        "You are a friendly assistant. The user and you will engage in a spoken dialog exchanging "
        "the transcripts of a natural real-time conversation. Keep your responses short, generally "
        "two or three sentences for chatty scenarios. "
        f"{AWSNovaSonicLLMService.AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION}"
    )
    chans = 1
    in_sr = 16000
    out_sr = 24000

    session_obj = json.loads(session_str)
    application_id = session_obj.get("apiKey", "")
    session_id = session_obj.get("sessionId", "")
    token = session_obj.get("token", "")

    transport = VonageVideoWebrtcTransport(
        application_id,
        session_id,
        token,
        VonageVideoWebrtcTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            publisher_name="TTS bot",
            audio_in_sample_rate=in_sr,
            audio_in_channels=chans,
            audio_out_sample_rate=out_sr,
            audio_out_channels=chans,
        ),
    )

    ns_params = aws_nova_sonic.aws.Params()
    ns_params.input_sample_rate = in_sr
    ns_params.output_sample_rate = out_sr
    ns_params.input_channel_count = chans
    ns_params.output_channel_count = chans

    llm = AWSNovaSonicLLMService(
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
        region=os.getenv("AWS_REGION", ""),
        session_token=os.getenv("AWS_SESSION_TOKEN", ""),
        voice_id="tiffany",
        params=ns_params,
    )
    context = OpenAILLMContext(
        messages=[
            {"role": "system", "content": f"{system_instruction}"},
            {
                "role": "user",
                "content": "Tell me a fun fact!",
            },
        ],
    )
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            context_aggregator.user(),
            llm,
            transport.output(),
        ]
    )

    task = PipelineTask(pipeline, observers=[TranscriptionLogObserver()])

    # Handle client connection event
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        await task.queue_frames([LLMRunFrame()])
        # HACK: for now, we need this special way of triggering the first assistant response in AWS
        # Nova Sonic. Note that this trigger requires a special corresponding bit of text in the
        # system instruction. In the future, simply queueing the context frame should be sufficient.
        await llm.trigger_assistant_response()

    runner = PipelineRunner()

    await asyncio.gather(runner.run(task))


def cli_main():
    """Console script entry point for the nova sonic vonage video webrtc example."""
    if len(sys.argv) > 1:
        session_str = sys.argv[1]
        logger.info(f"Session str: {session_str}")
    else:
        logger.error(f"Usage: {sys.argv[0]} <VONAGE_SESSION_STR>")
        logger.error("VONAGE_SESSION_STR should be a JSON string with the following format:")
        logger.error(
            '{"apiKey": "your_api_key", "sessionId": "your_session_id", "token": "your_token"}'
        )
        sys.exit(1)

    asyncio.run(main(session_str))


if __name__ == "__main__":
    cli_main()

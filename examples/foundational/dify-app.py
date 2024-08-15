# dify-app.py
import asyncio
import os
import sys
import aiohttp

from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator
)
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.services.dify import DifyLLMService
from pipecat.processors.frameworks.dify import DifyProcessor

from loguru import logger
from dotenv import load_dotenv
from runner import configure

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "DEBUG"))  # Configurable log level

message_store = {}


async def main():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:  # Added timeout
        room_url, token = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
            ),
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        dify_service = DifyLLMService(api_key=os.getenv("DIFY_API_KEY"))
        dify_processor = DifyProcessor(dify_service)

        tma_in = LLMUserResponseAggregator()
        tma_out = LLMAssistantResponseAggregator()

        pipeline = Pipeline([
            transport.input(),
            tma_in,
            dify_processor,
            tts,
            transport.output(),
            tma_out,
        ])

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

        transport.event_handler("on_first_participant_joined")(lambda t, p: asyncio.create_task(on_first_participant_joined(t, p, task, dify_processor)))

        runner = PipelineRunner()
        await runner.run(task)

async def on_first_participant_joined(transport, participant, task, dify_processor):
    """Handles the event when the first participant joins."""
    transport.capture_participant_transcription(participant["id"])
    dify_processor.set_participant_id(participant["id"])
    messages = [{
        "content": "Please briefly introduce yourself to the user."
    }]
    await task.queue_frames([LLMMessagesFrame(messages)])


if __name__ == "__main__":
    asyncio.run(main())

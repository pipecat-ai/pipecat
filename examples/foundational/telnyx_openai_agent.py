import asyncio
import os
import sys

from loguru import logger
from dotenv import load_dotenv

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

from pipecat.services.telnyx.stt import TelnyxSTTService
from pipecat.services.telnyx.tts import TelnyxTTSService

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main():
    transport = DailyTransport(
        room_url=os.getenv("DAILY_ROOM_URL"),
        token=os.getenv("DAILY_ROOM_TOKEN"),
        bot_name="Telnyx Bot",
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=1.2)),
            vad_audio_passthrough=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
        ),
    )

    stt = TelnyxSTTService(
        transcription_engine="telnyx",
        input_format="wav",
        sample_rate=16000,
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",
    )

    tts = TelnyxTTSService(
        voice="Telnyx.NaturalHD.astra",
        sample_rate=24000,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant testing Telnyx voice services. Keep answers concise.",
        },
    ]

    context = OpenAILLMContext(messages)

    user_aggregator = LLMUserResponseAggregator(messages)
    assistant_aggregator = LLMAssistantResponseAggregator(messages)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    runner = PipelineRunner()

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMMessagesFrame(messages)])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        await task.queue_frame(EndFrame())

    print("--- Starting Pipeline ---")
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())

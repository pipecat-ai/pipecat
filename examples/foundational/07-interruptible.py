import asyncio
import aiohttp
import logging
import os
from pipecat.pipeline.aggregators import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)

from pipecat.pipeline.pipeline import Pipeline
from pipecat.services.ai_services import FrameLogger
from pipecat.transports.daily_transport import DailyTransport
from pipecat.services.open_ai_services import OpenAILLMService
from pipecat.services.elevenlabs_ai_services import ElevenLabsTTSService

from runner import configure

from dotenv import load_dotenv
load_dotenv(override=True)

logging.basicConfig(format=f"%(levelno)s %(asctime)s %(message)s")
logger = logging.getLogger("pipecat")
logger.setLevel(logging.DEBUG)


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "Respond bot",
            duration_minutes=5,
            start_transcription=True,
            mic_enabled=True,
            mic_sample_rate=16000,
            camera_enabled=False,
            vad_enabled=True,
        )

        tts = ElevenLabsTTSService(
            aiohttp_session=session,
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4-turbo-preview")

        pipeline = Pipeline([FrameLogger(), llm, FrameLogger(), tts])

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport, participant):
            await transport.say("Hi, I'm listening!", tts)

        async def run_conversation():
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio. Respond to what the user said in a creative and helpful way.",
                },
            ]

            await transport.run_interruptible_pipeline(
                pipeline,
                post_processor=LLMAssistantResponseAggregator(messages),
                pre_processor=LLMUserResponseAggregator(messages),
            )

        await asyncio.gather(transport.run(), run_conversation())


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))

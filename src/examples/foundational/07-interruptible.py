import asyncio
import aiohttp
import os
from dailyai.conversation_wrappers import InterruptibleConversationWrapper
from dailyai.pipeline.aggregators import LLMAssistantContextAggregator, LLMUserContextAggregator

from dailyai.pipeline.frames import StartStreamQueueFrame, TextQueueFrame
from dailyai.pipeline.pipeline import Pipeline
from dailyai.services.ai_services import FrameLogger
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.services.elevenlabs_ai_service import ElevenLabsTTSService

from examples.foundational.support.runner import configure


async def main(room_url: str, token):
    async with aiohttp.ClientSession() as session:
        transport = DailyTransportService(
            room_url,
            token,
            "Respond bot",
            duration_minutes=5,
            start_transcription=True,
            mic_enabled=True,
            mic_sample_rate=16000,
            camera_enabled=False,
        )

        llm = AzureLLMService(
            api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
            endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"),
            model=os.getenv("AZURE_CHATGPT_MODEL"))
        tts = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"))

        pipeline = Pipeline([FrameLogger(), llm, FrameLogger(), tts])

        @transport.event_handler("on_first_other_participant_joined")
        async def on_first_other_participant_joined(transport):
            await tts.say("Hi, I'm listening!", transport.send_queue)

        async def run_conversation():
            messages = [
                {"role": "system", "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio. Respond to what the user said in a creative and helpful way."},
            ]

            await transport.run_interruptible_pipeline(
                pipeline,
                post_processor=LLMAssistantContextAggregator(
                    messages, transport._my_participant_id
                ),
                pre_processor=LLMUserContextAggregator(
                    messages, transport._my_participant_id, complete_sentences=False
                ),
            )

        transport.transcription_settings["extra"]["punctuate"] = False
        await asyncio.gather(transport.run(), run_conversation())


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))

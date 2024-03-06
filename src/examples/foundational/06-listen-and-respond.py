import asyncio
import os
from dailyai.pipeline.pipeline import Pipeline

from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.services.ai_services import FrameLogger
from dailyai.pipeline.aggregators import LLMAssistantContextAggregator, LLMUserContextAggregator
from examples.foundational.support.runner import configure


async def main(room_url: str, token):
    transport = DailyTransportService(
        room_url,
        token,
        "Respond bot",
        duration_minutes=5,
        start_transcription=True,
        mic_enabled=True,
        mic_sample_rate=16000,
        camera_enabled=False,
        vad_enabled=True
    )

    llm = AzureLLMService(
        api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
        endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"),
        model=os.getenv("AZURE_CHATGPT_MODEL"))
    tts = AzureTTSService(
        api_key=os.getenv("AZURE_SPEECH_API_KEY"),
        region=os.getenv("AZURE_SPEECH_REGION"))
    fl = FrameLogger("Inner")
    fl2 = FrameLogger("Outer")
    @transport.event_handler("on_first_other_participant_joined")
    async def on_first_other_participant_joined(transport):
        await tts.say("Hi, I'm listening!", transport.send_queue)

    async def handle_transcriptions():
        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio. Respond to what the user said in a creative and helpful way.",
            },
        ]

        tma_in = LLMUserContextAggregator(messages, transport._my_participant_id)
        tma_out = LLMAssistantContextAggregator(messages, transport._my_participant_id)
        pipeline = Pipeline(
            processors=[
                fl,
                tma_in,
                llm,
                fl2,
                tma_out,
                tts
            ],
        )
        await transport.run_uninterruptible_pipeline(pipeline)

    transport.transcription_settings["extra"]["endpointing"] = True
    transport.transcription_settings["extra"]["punctuate"] = True
    await asyncio.gather(transport.run(), handle_transcriptions())


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))

import asyncio
import os

from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureLLMService, AzureTTSService
from dailyai.queue_aggregators import LLMAssistantContextAggregator, LLMContextAggregator, LLMUserContextAggregator
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
        camera_enabled=False
    )

    llm = AzureLLMService(
        api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
        endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"),
        model=os.getenv("AZURE_CHATGPT_MODEL"))
    tts = AzureTTSService(
        api_key=os.getenv("AZURE_SPEECH_API_KEY"),
        region=os.getenv("AZURE_SPEECH_REGION"),
        voice="es-ES-ElviraNeural")

    @transport.event_handler("on_first_other_participant_joined")
    async def on_first_other_participant_joined(transport):
        await tts.say("Hi, I'm listening!", transport.send_queue)

    async def handle_transcriptions():
        messages = [
            {
                "role": "system",
                "content": "You will be provided with a sentence in English, and your task is to translate it into Spanish.",
            },
        ]

        tma_in = LLMUserContextAggregator(
            messages, transport._my_participant_id, store=False)
        await tts.run_to_queue(
            transport.send_queue,
            llm.run(
                tma_in.run(
                    transport.get_receive_frames()
                )
            )
        )

    transport.transcription_settings["extra"]["punctuate"] = True
    await asyncio.gather(transport.run(), handle_transcriptions())


if __name__ == "__main__":
    (url, token) = configure()
    asyncio.run(main(url, token))

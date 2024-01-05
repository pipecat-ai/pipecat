import argparse
import time
import asyncio

from dailyai.orchestrator import OrchestratorConfig, Orchestrator
from dailyai.output_queue import OutputQueueFrame, FrameType
from dailyai.message_handler.message_handler import MessageHandler
from dailyai.services.ai_services import AIServiceConfig
from dailyai.services.azure_ai_services import AzureTTSService, AzureLLMService
from dailyai.services.open_ai_service import OpenAILLMService
from dailyai.services.deepgram_ai_services import DeepgramTTSService
from dailyai.services.daily_transport_service import DailyTransportService

# For now, use Azure service for the TTS. Todo: make tts service
# and tts args (like which voice to use) configurable via command
# line arguments.
# Need the following environment variables:
# - AZURE_SPEECH_SERVICE_KEY
# - AZURE_SPEECH_SERVICE_REGION


async def main(room_url, text) -> None:
    class Sample01Transport(DailyTransportService):
        def on_participant_joined(self, participant):
            super().on_participant_joined(participant)

    meeting_duration_minutes = 4
    transport = Sample01Transport(
        room_url,
        None,
        "Simple Bot",
        meeting_duration_minutes,
    )
    transport.mic_enabled = True
    transport.camera_enabled = True
    transport.mic_sample_rate = 16000
    transport.camera_width = 1024
    transport.camera_height = 1024

    llm = OpenAILLMService()
    tts = DeepgramTTSService()

    async def get_all_audio(text):
        all_audio = bytearray()
        async for audio in tts.run_tts(text):
            all_audio.extend(audio)

        return all_audio

    async def say_text(text):

        audio = await get_all_audio(text)

        print(f"Got audio for {text}")
        transport.output_queue.put(OutputQueueFrame(FrameType.AUDIO_FRAME, audio))

    try:
        transport.run()
        print("gathering")
        await say_text(text)
        print("waiting")
        sleeper = asyncio.sleep(2)
        await sleeper
        print("done")
    except Exception as e:
        print("Exception", e)
    finally:
        print("finally")
        transport.stop()
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say one phrase and exit")
    parser.add_argument("-u", "--url", type=str,
                        required=True, help="URL of the Daily room")

    parser.add_argument(
        "-t", "--text", type=str, required=True, help="text to send into the session as speech"
    )

    args: argparse.Namespace = parser.parse_args()

    asyncio.run(main(args.url, args.text))

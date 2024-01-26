import asyncio
import time
from typing import AsyncGenerator

from dailyai.queue_frame import QueueFrame, FrameType
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureTTSService
from dailyai.services.deepgram_ai_services import DeepgramTTSService


async def main(room_url):
    # create a transport service object using environment variables for
    # the transport service's API key, room url, and any other configuration.
    # services can all define and document the environment variables they use.
    # services all also take an optional config object that is used instead of
    # environment variables.
    #
    # the abstract transport service APIs presumably can map pretty closely
    # to the daily-python basic API
    meeting_duration_minutes = 1
    transport = DailyTransportService(
        room_url,
        None,
        "Greeter",
        meeting_duration_minutes,
    )
    transport.mic_enabled = True

    # similarly, create a tts service
    tts = DeepgramTTSService()

    # Get the generator for the audio. This will start running in the background,
    # and when we ask the generator for its items, we'll get what it's generated.

    # Register an event handler so we can play the audio when the participant joins.
    print("settting up handler")

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        print(f"participant joined: {participant['info']['userName']}")
        if participant["info"]["isLocal"]:
            return
        audio_generator: AsyncGenerator[bytes, None] = tts.run_tts(
            f"Hello there, {participant['info']['userName']}!")

        async for audio in audio_generator:
            transport.output_queue.put(QueueFrame(FrameType.AUDIO, audio))

    print("setting up call state handler")

    @transport.event_handler("on_call_state_updated")
    async def on_call_joined(transport, state):
        print(f"call state callback: {state}")

    await transport.run()


if __name__ == "__main__":
    asyncio.run(main("https://chad-hq.daily.co/howdy"))

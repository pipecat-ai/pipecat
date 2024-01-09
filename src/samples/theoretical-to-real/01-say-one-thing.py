import argparse
import asyncio
from typing import AsyncGenerator

from dailyai.output_queue import OutputQueueFrame, FrameType
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureTTSService

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
        "Say One Thing",
        meeting_duration_minutes,
    )
    transport.mic_enabled = True

    # similarly, create a tts service
    tts = AzureTTSService()

    # Get the generator for the audio. This will start running in the background,
    # and when we ask the generator for its items, we'll get what it's generated.
    audio_generator: AsyncGenerator[bytes, None] = tts.run_tts("hello world")

    # Register an event handler so we can play the audio when the participant joins.
    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        if participant["info"]["isLocal"]:
            return
        async for audio in audio_generator:
            transport.output_queue.put(OutputQueueFrame(FrameType.AUDIO_FRAME, audio))

        # wait for the output queue to be empty, then leave the meeting
        transport.output_queue.join()
        transport.stop()

    await transport.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Daily Bot Sample")
    parser.add_argument(
        "-u", "--url", type=str, required=True, help="URL of the Daily room to join"
    )

    args: argparse.Namespace = parser.parse_args()

    asyncio.run(main(args.url))

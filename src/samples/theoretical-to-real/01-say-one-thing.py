import asyncio
from typing import AsyncGenerator

from dailyai.output_queue import OutputQueueFrame, FrameType
from dailyai.services.daily_transport_service import DailyTransportService
from dailyai.services.azure_ai_services import AzureTTSService

class Sample01Transport(DailyTransportService):
    def __init__(
        self,
        room_url: str,
        token: str | None,
        bot_name: str,
        duration: float = 10,
    ):
        super().__init__(room_url, token, bot_name, duration)

    def set_audio_generator(self, audio_generator) -> None:
        self.audio_generator = audio_generator

    async def play_audio(self):
        print("playing audio", self.audio_generator)
        async for audio in self.audio_generator:
            print("putting frame on queue")
            self.output_queue.put(OutputQueueFrame(FrameType.AUDIO_FRAME, audio))

    def on_participant_joined(self, participant):
        super().on_participant_joined(participant)
        asyncio.run(self.play_audio())

async def main(room_url):
    # create a transport service object using environment variables for
    # the transport service's API key, room url, and any other configuration.
    # services can all define and document the environment variables they use.
    # services all also take an optional config object that is used instead of
    # environment variables.
    #
    # the abstract transport service APIs presumably can map pretty closely
    # to the daily-python basic API
    meeting_duration_minutes = 4
    transport = Sample01Transport(
        room_url,
        None,
        "Say One Thing",
        meeting_duration_minutes,
    )
    transport.mic_enabled = True

    # similarly, create a tts service
    tts = AzureTTSService()

    # ask the transport to create a local audio "device"/queue for
    # chunks of audio to play sequentially. the "mic" object is a handle
    # we can use to inspect and control the queue if we need to. in this
    # case we will pipe into this queue from the tts service
    audio_generator: AsyncGenerator[bytes, None] = tts.run_tts("hello world")

    # Should this just happen when we create the object?
    transport.set_audio_generator(audio_generator)
    try:
        transport.run()
    except KeyboardInterrupt:
        print("Keyboard interrupt")
    finally:
        transport.stop()


if __name__ == "__main__":
    asyncio.run(main("https://moishe.daily.co/Lettvins"))

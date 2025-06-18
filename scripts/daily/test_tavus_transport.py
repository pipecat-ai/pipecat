import asyncio
import os
import signal

from daily import *
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)


def completion_callback(future):
    def _callback(*args):
        def set_result(future, *args):
            try:
                if len(args) > 1:
                    future.set_result(args)
                else:
                    future.set_result(*args)
            except asyncio.InvalidStateError:
                pass

        future.get_loop().call_soon_threadsafe(set_result, future, *args)

    return _callback


class DailyProxyApp(EventHandler):
    # This is necessary to override EventHandler's __new__ method.
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __init__(self, sample_rate: int):
        super().__init__()
        self._sample_rate = sample_rate
        self._loop = None
        self._audio_queue: asyncio.Queue | None = None
        self._audio_task: asyncio.Task | None = None

        self._client: CallClient = CallClient(event_handler=self)
        self._client.update_subscription_profiles(
            {"base": {"camera": "unsubscribed", "microphone": "subscribed"}}
        )

        self._audio_source = CustomAudioSource(self._sample_rate, 1)
        self._audio_track = CustomAudioTrack(self._audio_source)

    def on_joined(self, data, error):
        logger.debug("Local participant Joined!")
        if error:
            print(f"Unable to join meeting: {error}")
            self._loop.call_soon_threadsafe(self._loop.stop)

    def run(self, meeting_url: str):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._create_audio_task()

        def handle_exit():
            logger.info("Ctrl+C pressed. Leaving the meeting...")
            self._loop.call_soon_threadsafe(self._loop.stop)

        for sig in (signal.SIGINT, signal.SIGTERM):
            self._loop.add_signal_handler(sig, handle_exit)

        self._client.set_user_name("TestTavusTransport")
        self._client.join(
            meeting_url,
            completion=self.on_joined,
            client_settings={
                "inputs": {
                    "microphone": {
                        "isEnabled": True,
                        "settings": {"customTrack": {"id": self._audio_track.id}},
                    },
                }
            },
        )

        try:
            self._loop.run_forever()
        finally:
            self.leave()

    def leave(self):
        if self._audio_task:
            self._loop.run_until_complete(self._cancel_audio_task())

        self._client.leave()
        self._client.release()

    async def update_subscriptions(self, participant_settings=None, profile_settings=None):
        logger.info(f"Updating subscriptions participant_settings: {participant_settings}")
        future = asyncio.get_running_loop().create_future()
        self._client.update_subscriptions(
            participant_settings=participant_settings,
            profile_settings=profile_settings,
            completion=completion_callback(future),
        )
        await future

    def _create_audio_task(self):
        if not self._audio_task:
            self._audio_queue = asyncio.Queue()
            self._audio_task = self._loop.create_task(self._audio_task_handler())

    async def _cancel_audio_task(self):
        if self._audio_task:
            self._audio_task.cancel()
            try:
                # Waits for it to finish
                await self._audio_task
            except asyncio.CancelledError:
                pass
            self._audio_task = None
            self._audio_queue = None

    async def capture_participant_audio(self, participant_id: str):
        logger.info(f"Capturing participant audio: {participant_id}")
        # Receiving from this custom track
        # audio_source: str = "microphone"
        audio_source: str = "stream"
        media = {"media": {"customAudio": {audio_source: "subscribed"}}}
        await self.update_subscriptions(participant_settings={participant_id: media})

        self._client.set_audio_renderer(
            participant_id,
            self._audio_data_received,
            audio_source=audio_source,
            sample_rate=self._sample_rate,
            callback_interval_ms=20,
        )

    async def send_audio(self, audio: AudioData):
        future = asyncio.get_running_loop().create_future()
        self._audio_source.write_frames(audio.audio_frames, completion=completion_callback(future))
        await future

    async def queue_audio(self, audio: AudioData):
        await self._audio_queue.put(audio)

    def _audio_data_received(self, participant_id: str, audio_data: AudioData, audio_source: str):
        # logger.info(f"Received audio data for {participant_id}, audio_source: {audio_source}")
        asyncio.run_coroutine_threadsafe(self.queue_audio(audio_data), self._loop)

    async def _audio_task_handler(self):
        while True:
            audio = await self._audio_queue.get()
            await self.send_audio(audio)

    #
    # Daily (EventHandler)
    #

    def on_participant_joined(self, participant):
        participant_name = participant["info"]["userName"]
        logger.info(f"Participant {participant_name} joined")
        if participant_name != "Pipecat":
            # We are only subscribing for audios from Pipecat.
            return
        asyncio.run_coroutine_threadsafe(
            self.capture_participant_audio(participant_id=participant["id"]), self._loop
        )

    def on_participant_left(self, participant, reason):
        logger.info(f"Participant {participant['id']} left {reason}")


def main():
    Daily.init()
    room_url = os.getenv("TAVUS_SAMPLE_ROOM_URL")
    app = DailyProxyApp(sample_rate=24000)
    app.run(room_url)


if __name__ == "__main__":
    main()

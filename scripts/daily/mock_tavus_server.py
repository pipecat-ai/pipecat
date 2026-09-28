import asyncio
import base64
import os
import signal

from daily import (
    CallClient,
    CustomAudioSource,
    CustomAudioTrack,
    Daily,
    EventHandler,
)
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

    def __init__(self):
        super().__init__()
        self._loop = asyncio.new_event_loop()
        # Raw PCM buffer — filled by app-message audio, drained at received sample rate.
        self._buffer = bytearray()
        self._audio_task: asyncio.Task | None = None
        self._msg_count = 0
        self._msg_bytes = 0

        # Initialized lazily on the first audio message so the sample rate matches
        # what the server actually sends.
        self._sample_rate: int | None = None
        self._audio_source: CustomAudioSource | None = None
        self._audio_track: CustomAudioTrack | None = None

        self._client: CallClient = CallClient(event_handler=self)
        self._client.update_subscription_profiles(
            {"base": {"camera": "unsubscribed", "microphone": "unsubscribed"}}
        )

    def on_joined(self, data, error):
        logger.debug("Local participant Joined!")
        if error:
            print(f"Unable to join meeting: {error}")
            self._loop.call_soon_threadsafe(self._loop.stop)

    def run(self, meeting_url: str):
        asyncio.set_event_loop(self._loop)
        self._create_audio_task()

        def handle_exit():
            logger.info("Ctrl+C pressed. Leaving the meeting...")
            self._loop.call_soon_threadsafe(self._loop.stop)

        for sig in (signal.SIGINT, signal.SIGTERM):
            self._loop.add_signal_handler(sig, handle_exit)

        self._client.set_user_name("TestTavusTransport")
        self._client.join(meeting_url, completion=self.on_joined)

        try:
            self._loop.run_forever()
        finally:
            self.leave()

    def leave(self):
        if self._audio_task:
            self._loop.run_until_complete(self._cancel_audio_task())

        self._client.leave()
        self._client.release()

    def _create_audio_task(self):
        if not self._audio_task:
            self._audio_task = self._loop.create_task(self._audio_task_handler())

    async def _cancel_audio_task(self):
        if self._audio_task:
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass
            self._audio_task = None

    async def _maybe_init_audio_source(self, sample_rate: int):
        """Create CustomAudioSource at the server's sample rate and publish it."""
        if self._audio_source is not None:
            return
        self._sample_rate = sample_rate
        self._audio_source = CustomAudioSource(sample_rate, 1, False)
        self._audio_track = CustomAudioTrack(self._audio_source)
        future = asyncio.get_running_loop().create_future()
        self._client.update_inputs(
            {
                "microphone": {
                    "isEnabled": True,
                    "settings": {"customTrack": {"id": self._audio_track.id}},
                }
            },
            completion=completion_callback(future),
        )
        await future
        logger.info(f"Audio source initialized at {sample_rate}Hz")

    async def _buffer_audio(self, data: bytes, sample_rate: int):
        """Initialize the audio source if needed, then append bytes to the buffer."""
        await self._maybe_init_audio_source(sample_rate)
        self._buffer.extend(data)

    async def _handle_interrupt(self):
        """Clear the audio buffer, mimicking the avatar stopping mid-speech."""
        dropped = len(self._buffer)
        self._buffer.clear()
        if self._sample_rate:
            logger.info(
                f"Interrupt received — dropped {dropped}B "
                f"({dropped / (self._sample_rate * 2):.3f}s) from buffer"
            )

    async def _audio_task_handler(self):
        """Drain the buffer at the received sample rate (real-time playback).

        Waits for the audio source to be initialized, then waits until 100ms of audio
        is accumulated before starting playback. Drains in 20ms steps. If the buffer
        runs dry it re-enters the waiting state.
        """
        buffering = True
        last_log_time = self._loop.time()

        while True:
            if self._audio_source is None or self._sample_rate is None:
                await asyncio.sleep(0.01)
                last_log_time = self._loop.time()
                continue

            chunk_bytes = int(self._sample_rate * 20 / 1000) * 2  # 20ms, 16-bit mono
            min_audio_buffer = chunk_bytes * 5  # 100ms pre-buffer

            if buffering:
                if len(self._buffer) >= min_audio_buffer:
                    buffering = False
                    logger.debug(f"Pre-buffer reached ({min_audio_buffer}B) — starting playback")
                else:
                    await asyncio.sleep(0.001)
                    continue

            if len(self._buffer) >= chunk_bytes:
                chunk = bytes(self._buffer[:chunk_bytes])
                del self._buffer[:chunk_bytes]

                future = asyncio.get_running_loop().create_future()
                self._audio_source.write_frames(chunk, completion=completion_callback(future))
                await future
            else:
                buffering = True
                await asyncio.sleep(0.001)

            now = self._loop.time()
            if now - last_log_time >= 1.0:
                buffer_seconds = len(self._buffer) / (self._sample_rate * 2)
                logger.info(
                    f"msgs/s: {self._msg_count} | "
                    f"KB/s: {self._msg_bytes / 1024:.1f} | "
                    f"buffered: {buffer_seconds:.3f}s"
                )
                self._msg_count = 0
                self._msg_bytes = 0
                last_log_time = now

    #
    # Daily (EventHandler)
    #

    def on_app_message(self, message, sender):
        if not isinstance(message, dict):
            return
        event = message.get("event_type")
        if event == "conversation.echo":
            props = message.get("properties", {})
            if props.get("modality") != "audio":
                return
            try:
                audio_bytes = base64.b64decode(props["audio"])
                sample_rate = props["sample_rate"]
                self._msg_count += 1
                self._msg_bytes += len(audio_bytes)
                done = props.get("done", False)
                asyncio.run_coroutine_threadsafe(
                    self._buffer_audio(audio_bytes, sample_rate), self._loop
                )
                if done:
                    logger.debug(f"inference {props.get('inference_id')} done")
            except Exception as e:
                logger.error(f"Error decoding audio message: {e}")
        elif event == "conversation.interrupt":
            asyncio.run_coroutine_threadsafe(self._handle_interrupt(), self._loop)

    def on_participant_joined(self, participant):
        participant_name = participant["info"]["userName"]
        logger.info(f"Participant {participant_name} joined")

    def on_participant_left(self, participant, reason):
        logger.info(f"Participant {participant['id']} left {reason}")


def main():
    Daily.init()
    room_url = os.environ["TAVUS_SAMPLE_ROOM_URL"]
    app = DailyProxyApp()
    app.run(room_url)


if __name__ == "__main__":
    main()

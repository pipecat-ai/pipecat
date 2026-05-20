import array
import asyncio
import os
import signal

from daily import (
    AudioData,
    CallClient,
    CustomAudioSource,
    CustomAudioTrack,
    Daily,
    EventHandler,
)
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

# Pipecat sends audio at this true content rate but declares it as
# DECLARED_SAMPLE_RATE to write_frames(), which makes delivery faster than
# real-time. We receive at the declared rate (no resampling) and play back at
# the true rate so the avatar consumes audio at normal speed.
TRUE_SAMPLE_RATE = 24000
DECLARED_SAMPLE_RATE = 48000
SPEEDUP = DECLARED_SAMPLE_RATE // TRUE_SAMPLE_RATE


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
        # Raw PCM buffer — filled at DECLARED_SAMPLE_RATE speed, drained at TRUE_SAMPLE_RATE speed.
        self._buffer = bytearray()
        self._audio_task: asyncio.Task | None = None
        # Tracks whether the previous frame was silence, to detect speech→silence transitions.
        # Initialised True so leading silence before first speech is dropped.
        self._last_was_silence: bool = True

        self._client: CallClient = CallClient(event_handler=self)
        self._client.update_subscription_profiles(
            {"base": {"camera": "unsubscribed", "microphone": "subscribed"}}
        )

        # Playback source declared at TRUE_SAMPLE_RATE — consumes audio at real-time speed.
        self._audio_source = CustomAudioSource(TRUE_SAMPLE_RATE, 1)
        self._audio_track = CustomAudioTrack(self._audio_source)

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
            self._audio_task = self._loop.create_task(self._audio_task_handler())

    async def _cancel_audio_task(self):
        if self._audio_task:
            self._audio_task.cancel()
            try:
                await self._audio_task
            except asyncio.CancelledError:
                pass
            self._audio_task = None

    async def capture_participant_audio(self, participant_id: str):
        logger.info(f"Capturing participant audio: {participant_id}")
        audio_source: str = "stream"
        media = {"media": {"customAudio": {audio_source: "subscribed"}}}
        await self.update_subscriptions(participant_settings={participant_id: media})

        # Must match the declared rate Pipecat used so WebRTC skips resampling —
        # every original byte arrives intact.
        self._client.set_audio_renderer(
            participant_id,
            self._audio_data_received,
            audio_source=audio_source,
            sample_rate=DECLARED_SAMPLE_RATE,
            callback_interval_ms=20,
        )
        logger.info(
            f"Receiving at declared_rate={DECLARED_SAMPLE_RATE} Hz "
            f"(true content: {TRUE_SAMPLE_RATE} Hz, ~{SPEEDUP}x faster than real-time)"
        )

    @staticmethod
    def _is_silence(data: bytes, threshold: int = 5) -> bool:
        # Interpret as 16-bit signed PCM samples and check peak amplitude.
        # WebRTC-injected silence is all zeros; real TTS audio has non-trivial
        # amplitude. This lets us skip buffering frames that Pipecat never wrote,
        # so the buffer only grows when actual speech arrives (via our trick).
        samples = array.array("h", data)
        return max(abs(s) for s in samples) < threshold

    async def _buffer_audio(self, audio_data: AudioData):
        """Append received bytes to the buffer, skipping WebRTC-injected silence.

        Speech frames arrive at DECLARED_SAMPLE_RATE speed (~2x real-time) and
        are added in full — the buffer grows ahead of playback, which is the
        fast-audio effect we want to observe. The buffer then drains at
        TRUE_SAMPLE_RATE speed once speech stops.

        On the first silence after speech (speech→silence transition) we insert
        one real-time-equivalent chunk of silence to represent the natural pause.
        All subsequent silence frames are dropped so the buffer drains back down.
        This also limits the impact of any TTS-emitted silence to a single chunk
        per transition rather than letting it accumulate.
        """
        new_bytes = audio_data.audio_frames
        if self._is_silence(new_bytes):
            if not self._last_was_silence:
                # First silence after speech: mark the pause with one chunk.
                self._buffer.extend(bytes(len(new_bytes) // SPEEDUP))
                self._last_was_silence = True
            return

        self._last_was_silence = False
        self._buffer.extend(new_bytes)

    def _audio_data_received(self, participant_id: str, audio_data: AudioData, audio_source: str):
        asyncio.run_coroutine_threadsafe(self._buffer_audio(audio_data), self._loop)

    async def _handle_interrupt(self):
        """Clear the audio buffer, mimicking the avatar stopping mid-speech."""
        dropped = len(self._buffer)
        self._buffer.clear()
        self._last_was_silence = True
        logger.info(
            f"Interrupt received — dropped {dropped}B ({dropped / (TRUE_SAMPLE_RATE * 2):.3f}s) from buffer"
        )

    #
    # Daily (EventHandler)
    #

    def on_app_message(self, message, sender):
        if not isinstance(message, dict):
            return
        if message.get("event_type") == "conversation.interrupt":
            asyncio.run_coroutine_threadsafe(self._handle_interrupt(), self._loop)

    async def _audio_task_handler(self):
        """Drain the buffer at TRUE_SAMPLE_RATE speed (real-time playback)."""
        chunk_frames = int(TRUE_SAMPLE_RATE * 20 / 1000)  # 20 ms chunks
        chunk_bytes = chunk_frames * 2  # 16-bit mono
        last_log_time = self._loop.time()

        while True:
            if len(self._buffer) >= chunk_bytes:
                chunk = bytes(self._buffer[:chunk_bytes])
                del self._buffer[:chunk_bytes]

                future = asyncio.get_running_loop().create_future()
                self._audio_source.write_frames(chunk, completion=completion_callback(future))
                await future
            else:
                await asyncio.sleep(0.001)

            now = self._loop.time()
            if now - last_log_time >= 1.0:
                buffer_seconds = len(self._buffer) / (TRUE_SAMPLE_RATE * 2)
                if buffer_seconds > 0:
                    logger.info(
                        f"Buffer status: {len(self._buffer)}B ({buffer_seconds:.3f}s buffered)"
                    )
                    last_log_time = now

    def on_participant_joined(self, participant):
        participant_name = participant["info"]["userName"]
        logger.info(f"Participant {participant_name} joined")
        if participant_name != "Pipecat":
            # We are only subscribing for audio from Pipecat.
            return
        asyncio.run_coroutine_threadsafe(
            self.capture_participant_audio(participant_id=participant["id"]), self._loop
        )

    def on_participant_left(self, participant, reason):
        logger.info(f"Participant {participant['id']} left {reason}")


def main():
    Daily.init()
    room_url = os.environ["TAVUS_SAMPLE_ROOM_URL"]
    app = DailyProxyApp()
    app.run(room_url)


if __name__ == "__main__":
    main()

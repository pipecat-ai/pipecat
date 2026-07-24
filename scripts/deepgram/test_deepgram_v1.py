"""
Example: Live Transcription with WebSocket V1

Uses Deepgram Listen V1 for real-time speech recognition,
streaming audio directly from the microphone.
"""

import asyncio
import os
from datetime import datetime

import sounddevice as sd
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from deepgram.listen.v1.types import ListenV1Results
from dotenv import load_dotenv

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
CHANNELS = 1
KEEPALIVE_INTERVAL = 5  # Deepgram closes idle connections after 10 s (NET-0001)

# ── Shared state ───────────────────────────────────────────────────────────────

client = AsyncDeepgramClient(api_key=os.environ["DEEPGRAM_API_KEY"])
audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

# ── Helpers ────────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {msg}")


def on_message(message) -> None:
    if not isinstance(message, ListenV1Results):
        log(f"Received {getattr(message, 'type', type(message).__name__)} event")
        log(message)
        return
    if not message.channel or not message.channel.alternatives:
        return
    transcript = message.channel.alternatives[0].transcript
    if not transcript:
        return
    prefix = "Final" if message.is_final else "Interim"
    log(f"{prefix}: {transcript}")


async def send_audio(connection) -> None:
    """Capture mic audio and forward each chunk to the WebSocket."""
    loop = asyncio.get_running_loop()

    def microphone_callback(indata, frames, time_info, status) -> None:
        if status:
            log(f"Microphone status: {status}")
        # sounddevice fires this on its own thread; bridge into the event loop.
        loop.call_soon_threadsafe(audio_queue.put_nowait, bytes(indata))

    log("Microphone streaming started. Press Ctrl+C to stop.")
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=microphone_callback,
        blocksize=320,  # 320 samples @ 16 kHz = 20 ms per chunk
    ):
        while True:
            chunk = await audio_queue.get()
            await connection.send_media(chunk)


async def send_keepalive(connection) -> None:
    """Send periodic KeepAlive messages to prevent server-side timeout."""
    while True:
        await asyncio.sleep(KEEPALIVE_INTERVAL)
        try:
            log("Sending keepalive ...")
            await connection.send_keep_alive()
            log("Keepalive sent")
        except Exception:
            log(f"Keepalive failed: {type(e).__name__}: {e}")
            break


# ── Main ───────────────────────────────────────────────────────────────────────


async def main() -> None:
    async with client.listen.v1.connect(
        model="nova-3-general",
        encoding="linear16",
        sample_rate=SAMPLE_RATE,
        channels=CHANNELS,
        interim_results=True,
        punctuate=True,
        smart_format=True,
    ) as connection:
        connection.on(EventType.OPEN, lambda _: log("Connection opened"))
        connection.on(EventType.MESSAGE, on_message)
        connection.on(EventType.CLOSE, lambda _: log("Connection closed"))
        connection.on(EventType.ERROR, lambda err: log(f"Error: {type(err).__name__}: {err}"))

        sender = asyncio.create_task(send_audio(connection))
        keepalive = asyncio.create_task(send_keepalive(connection))

        try:
            await connection.start_listening()
        finally:
            sender.cancel()
            keepalive.cancel()
            await asyncio.gather(sender, keepalive, return_exceptions=True)
            try:
                await connection.send_close_stream()
            except Exception:
                log(f"Close stream failed: {type(e).__name__}: {e}")
                pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Stopping...")
    except Exception as e:
        log(f"Fatal: {type(e).__name__}: {e}")

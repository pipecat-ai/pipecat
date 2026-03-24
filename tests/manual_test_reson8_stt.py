"""Manual integration test for Reson8 STT.

Usage:
    RESON8_API_KEY=<key> uv run python tests/manual_test_reson8_stt.py [audio.pcm]

Connects to Reson8, sends audio (or silence), and prints transcriptions.
Audio file should be raw PCM s16le, 16kHz, mono.

Generate test audio:
    say -o /tmp/test.aiff "Hello, this is a test"
    ffmpeg -y -i /tmp/test.aiff -ar 16000 -ac 1 -f s16le /tmp/test.pcm
"""

import asyncio
import json
import os
import sys
import time

from loguru import logger

from pipecat.services.reson8.stt import Reson8STTService

API_KEY = os.environ.get("RESON8_API_KEY", "")
API_URL = os.environ.get("RESON8_API_URL", "https://api.reson8.dev")
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 100


async def main():
    if not API_KEY:
        logger.error("Set RESON8_API_KEY env var")
        return

    audio_file = sys.argv[1] if len(sys.argv) > 1 else None

    if audio_file:
        with open(audio_file, "rb") as f:
            audio_data = f.read()
        duration = len(audio_data) / (SAMPLE_RATE * 2)
        logger.info(f"Loaded {audio_file} ({duration:.1f}s)")
    else:
        duration = 3.0
        audio_data = b"\x00" * int(SAMPLE_RATE * 2 * duration)
        logger.info(f"Using {duration:.1f}s of silence (pass a .pcm file for real audio)")

    stt = Reson8STTService(
        api_key=API_KEY,
        api_url=API_URL,
        sample_rate=SAMPLE_RATE,
    )

    logger.info(f"Connecting to {API_URL}...")

    stt._sample_rate = SAMPLE_RATE
    await stt._connect_websocket()

    logger.info("Connected! Sending audio...")

    # Stats
    utterances = []
    interim_count = 0
    final_count = 0
    first_transcript_at = None
    t_start = time.monotonic()

    async def receive_messages():
        nonlocal interim_count, final_count, first_transcript_at
        try:
            async for message in stt._websocket:
                if isinstance(message, bytes):
                    continue
                msg = json.loads(message)
                msg_type = msg.get("type")
                if msg_type == "transcript":
                    text = msg.get("text", "")
                    is_final = msg.get("is_final", False)

                    if not text or text.strip() == "<silence>":
                        continue

                    if first_transcript_at is None:
                        first_transcript_at = time.monotonic()

                    clean = text.replace("<eou>", "").replace("<silence>", "").strip()
                    elapsed = time.monotonic() - t_start
                    if is_final:
                        final_count += 1
                        if clean:
                            utterances.append(clean)
                            print(f"  [{elapsed:6.1f}s] FINAL:   {clean}")
                    else:
                        interim_count += 1
                        if clean:
                            print(f"  [{elapsed:6.1f}s] INTERIM: {clean}")
                elif msg_type == "flush_confirmation":
                    logger.debug(f"Flush confirmed (id={msg.get('id')})")
                elif msg_type == "error":
                    logger.error(f"Server error: {msg.get('message')}")
        except Exception as e:
            logger.error(f"Receive error: {e}")

    recv_task = asyncio.create_task(receive_messages())

    # Send audio in chunks, pacing at ~4x realtime to avoid overwhelming the server
    chunk_bytes = int(SAMPLE_RATE * 2 * CHUNK_DURATION_MS / 1000)
    chunk_delay = CHUNK_DURATION_MS / 1000 / 4  # 4x realtime
    for offset in range(0, len(audio_data), chunk_bytes):
        chunk = audio_data[offset : offset + chunk_bytes]
        await stt._websocket.send(chunk)
        await asyncio.sleep(chunk_delay)

    send_elapsed = time.monotonic() - t_start
    logger.info(f"Audio sent in {send_elapsed:.1f}s ({duration / send_elapsed:.1f}x real-time)")

    # Send flush request to get final transcript
    await stt._websocket.send('{"type": "flush_request"}')
    logger.info("Waiting for final transcription...")

    # Wait for final results
    wait_time = max(5, int(duration * 0.05))
    logger.info(f"Waiting {wait_time}s for results...")
    await asyncio.sleep(wait_time)

    recv_task.cancel()
    try:
        await recv_task
    except asyncio.CancelledError:
        pass

    await stt._disconnect_websocket()

    total_elapsed = time.monotonic() - t_start
    ttft = (first_transcript_at - t_start) if first_transcript_at else None

    print(f"\n--- Results ---")
    print(f"Audio duration:    {duration:.1f}s")
    print(f"Total time:        {total_elapsed:.1f}s")
    print(f"Time to first txt: {ttft:.2f}s" if ttft else "Time to first txt: N/A")
    print(f"Interim messages:  {interim_count}")
    print(f"Final messages:    {final_count}")
    print(f"Utterances (eou):  {len(utterances)}")

    if not utterances:
        logger.warning("No transcription received")


if __name__ == "__main__":
    asyncio.run(main())

"""Smoke-test Fonada STT and TTS without requiring a transport or LLM."""

import asyncio
import os
import sys
import wave
from pathlib import Path

from pipecat.frames.frames import ErrorFrame, TranscriptionFrame, TTSAudioRawFrame
from pipecat.services.fonadalabs import FonadaSTTService, FonadaTTSService


async def main(audio_path: Path) -> None:
    api_key = os.environ["FONADALABS_API_KEY"]
    share_id = os.environ["FONADA_VOICE_SHARE_ID"]

    stt = FonadaSTTService(api_key=api_key, language=os.getenv("FONADA_LANGUAGE", "en"))
    tts = FonadaTTSService(
        api_key=api_key,
        share_id=share_id,
        language=os.getenv("FONADA_LANGUAGE", "en"),
    )
    try:
        transcript = ""
        async for frame in stt.run_stt(audio_path.read_bytes()):
            if isinstance(frame, TranscriptionFrame):
                transcript = frame.text
            elif isinstance(frame, ErrorFrame):
                raise RuntimeError(frame.error)
        print(f"ASR transcript: {transcript}")

        pcm = bytearray()
        async for frame in tts.run_tts(
            transcript or "Hello from FonadaLabs Pipecat.", "smoke-test"
        ):
            if isinstance(frame, TTSAudioRawFrame):
                pcm.extend(frame.audio)
            elif isinstance(frame, ErrorFrame):
                raise RuntimeError(frame.error)

        output_path = Path("fonadalabs-smoke-output.wav")
        with wave.open(str(output_path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(24000)
            output.writeframes(pcm)
        print(f"TTS audio: {output_path} ({len(pcm)} PCM bytes)")
    finally:
        await stt.cleanup()
        await tts.cleanup()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python examples/fonadalabs/smoke_test.py path/to/audio.wav")
    asyncio.run(main(Path(sys.argv[1])))

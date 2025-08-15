import asyncio
import aiohttp
import os

from pipecat.services.hume.tts import HumeTTSService
from pipecat.frames.frames import TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame

async def test_hume():
    # Clean up any existing output file
    output_file = "output.mp3"
    if os.path.exists(output_file):
        os.remove(output_file)

    async with aiohttp.ClientSession() as session:
        utterance = HumeTTSService.UtteranceParams(
            voice_id="ee96fb5f-ec1a-4f41-a9ba-6d119e64c8fd",
            voice_provider="HUME_AI",
        )
        tts = HumeTTSService(
            api_key="3GCACmC1ASidxvOArcc0KbGdjlHIaVPUT2S1ew00XwBfk8Kq",
            aiohttp_session=session,
            utterance_params=utterance,
        )

        print("[INFO] Running Hume TTS...")

        async for frame in tts.run_tts("Hello, this is Hume speaking."):
            if isinstance(frame, TTSStartedFrame):
                print("[TTS Started]")
            elif isinstance(frame, TTSAudioRawFrame):
                print(f"[Audio Frame] size={len(frame.audio)} bytes")
                with open(output_file, "ab") as f:
                    f.write(frame.audio)
            elif isinstance(frame, TTSStoppedFrame):
                print("[TTS Stopped]")
                print(f"[INFO] Audio saved to {output_file}") # not seeing the output file :(

asyncio.run(test_hume())
import asyncio
import unittest

import openai
import pyaudio
from dotenv import load_dotenv

from pipecat.frames.frames import AudioRawFrame, ErrorFrame
from pipecat.services.openai import OpenAITTSService

load_dotenv()


class TestWhisperOpenAIService(unittest.IsolatedAsyncioTestCase):
    async def test_whisper_tts(self):
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16,
                         channels=1,
                         rate=24_000,
                         output=True)

        tts = OpenAITTSService(voice="nova")

        async for frame in tts.run_tts("Hello, there. Nice to meet you, seems to work well"):
            self.assertIsInstance(frame, AudioRawFrame)
            stream.write(frame.audio)

        await asyncio.sleep(.5)
        stream.stop_stream()
        pa.terminate()

        tts = OpenAITTSService(voice="invalid_voice")
        with self.assertRaises(openai.BadRequestError):
            async for frame in tts.run_tts("wont work"):
                self.assertIsInstance(frame, ErrorFrame)


if __name__ == "__main__":
    unittest.main()

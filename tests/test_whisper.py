import unittest

from pipecat.services.openai import WhisperTTSService


class TestWhisperOpenAIService(unittest.IsolatedAsyncioTestCase):
    async def test_whisper_tts(self):
        tts = WhisperTTSService()
        # tts_response = await tts.run_tts("Hello, world")
        await tts.say("Hi! If you want to talk to me, just say 'Hey Robot'.")


if __name__ == "__main__":
    unittest.main()

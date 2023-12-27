import os
import requests
import time

from typing import Generator

from ..services.ai_services import TTSService


class ElevenLabsTTSService(TTSService):
    def __init__(self):
        super().__init__()

        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID")

    def run_tts(self, sentence) -> Generator[bytes, None, None]:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
        payload = {"text": sentence, "model_id": "eleven_turbo_v2"}
        querystring = {"output_format": "pcm_16000", "optimize_streaming_latency": 2}
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        r = requests.request(
            "POST", url, json=payload, headers=headers, params=querystring, stream=True
        )

        if r.status_code != 200:
            self.logger.error(
                f"audio fetch status code: {r.status_code}, error: {r.text}"
            )
            return

        for chunk in r.iter_content(chunk_size=3200):
            if chunk:
                yield chunk

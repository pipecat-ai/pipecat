import os
import requests

from services.ai_service import AIService
from PIL import Image


class DeepgramAIService(AIService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.api_key = os.getenv("DEEPGRAM_API_KEY")

    def get_mic_sample_rate(self):
        return 24000

    def run_tts(self, sentence):
        self.logger.info(f"Running deepgram tts for {sentence}")
        base_url = "https://api.beta.deepgram.com/v1/speak"
        voice = os.getenv("DEEPGRAM_VOICE") or "alpha-apollo-en-v1"  # move this to an environment variable
        request_url = f"{base_url}?model={voice}&encoding=linear16&container=none"
        headers = {"authorization": f"token {self.api_key}"}

        r = requests.post(request_url, headers=headers, data=sentence)
        self.logger.info(
            f"audio fetch status code: {r.status_code}, content length: {len(r.content)}"
        )
        yield r.content

from services.ai_service import AIService
import openai
import os

# To use Google Cloud's AI products, you'll need to install Google Cloud
# CLI and enable the TTS and in your project:
# https://cloud.google.com/sdk/docs/install
from google.cloud import texttospeech


class GoogleAIService(AIService):
    def __init__(self):
        super().__init__()

        self.client = texttospeech.TextToSpeechClient()
        self.voice = texttospeech.VoiceSelectionParams(
            language_code="en-GB", name="en-GB-Neural2-F"
        )

        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000
        )

    def run_tts(self, sentence):
        synthesis_input = texttospeech.SynthesisInput(text=sentence.strip())
        result = self.client.synthesize_speech(
            input=synthesis_input,
            voice=self.voice,
            audio_config=self.audio_config)
        return result

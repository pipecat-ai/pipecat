import json
import io
import openai
import os
import requests

from typing import Generator

from daily_ai.services.ai_services import LLMService, TTSService, ImageGenService
from PIL import Image

# See .env.example for Azure configuration needed
from azure.cognitiveservices.speech import SpeechSynthesizer, SpeechConfig, ResultReason, CancellationReason

class AzureTTSService(TTSService):
    def __init__(self):
        super().__init__()

        self.speech_key = os.getenv("AZURE_SPEECH_SERVICE_KEY")
        self.speech_region = os.getenv("AZURE_SPEECH_SERVICE_REGION")

        self.speech_config = SpeechConfig(subscription=self.speech_key, region=self.speech_region)
        self.speech_synthesizer = SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)

    def run_tts(self, sentence) -> Generator[bytes, None, None]:
        self.logger.info("⌨️ running azure tts async")
        ssml = "<speak version='1.0' xml:lang='en-US' xmlns='http://www.w3.org/2001/10/synthesis' " \
           "xmlns:mstts='http://www.w3.org/2001/mstts'>" \
           "<voice name='en-US-SaraNeural'>" \
           "<mstts:silence type='Sentenceboundary' value='20ms' />" \
           "<mstts:express-as style='lyrical' styledegree='2' role='SeniorFemale'>" \
           "<prosody rate='1.05'>" \
           f"{sentence}" \
           "</prosody></mstts:express-as></voice></speak> "
        result = self.speech_synthesizer.speak_ssml(ssml)
        self.logger.info("⌨️ got azure tts result")
        if result.reason == ResultReason.SynthesizingAudioCompleted:
            self.logger.info("⌨️ returning result")
            # azure always sends a 44-byte header. Strip it off.
            yield result.audio_data[44:]
        elif result.reason == ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            self.logger.info("Speech synthesis canceled: {}".format(cancellation_details.reason))
            if cancellation_details.reason == CancellationReason.Error:
                self.logger.info("Error details: {}".format(cancellation_details.error_details))

class AzureLLMService(LLMService):
    def get_response(self, messages, stream):
        return openai.ChatCompletion.create(
            api_type="azure",
            api_version="2023-06-01-preview",
            api_key=os.getenv("AZURE_CHATGPT_KEY"),
            api_base=os.getenv("AZURE_CHATGPT_ENDPOINT"),
            deployment_id=os.getenv("AZURE_CHATGPT_DEPLOYMENT_ID"),
            stream=stream,
            messages=messages,
        )


    def run_llm_async(self, messages) -> Generator[str, None, None]:
        local_messages = messages.copy()
        messages_for_log = json.dumps(local_messages)
        self.logger.info(f"==== generating chat via azure: {messages_for_log}")

        response = self.get_response(local_messages, stream=True)

        for chunk in response:
            if len(chunk["choices"]) == 0:
                continue

            if "content" in chunk["choices"][0]["delta"]:
                if (
                    chunk["choices"][0]["delta"]["content"] != {}
                ):  # streaming a content chunk
                    yield chunk["choices"][0]["delta"]["content"]


    def run_llm(self, messages) -> str or None:
        local_messages = messages.copy()
        messages_for_log = json.dumps(local_messages)
        self.logger.info(f"==== generating chat via azure: {messages_for_log}")

        response = self.get_response(local_messages, stream=False)
        if (
            response
            and len(response["choices"]) > 0
            and "message" in response["choices"][0]
            and "content" in response["choices"][0]["message"]
        ):
            return response["choices"][0]["message"]["content"]
        else:
            return None


class AzureImageGenService(ImageGenService):
    def run_image_gen(self, sentence) -> Image.Image:
        self.logger.info("generating azure image", sentence)

        image = openai.Image.create(
            api_type = 'azure',
            api_version = '2023-06-01-preview',
            api_key = os.getenv('AZURE_DALLE_KEY'),
            api_base = os.getenv('AZURE_DALLE_ENDPOINT'),
            deployment_id = os.getenv("AZURE_DALLE_DEPLOYMENT_ID"),
            prompt=f'{sentence} in the style of {self.image_style}',
            n=1,
            size=f"1024x1024",
        )

        url = image["data"][0]["url"]
        response = requests.get(url)

        dalle_stream = io.BytesIO(response.content)
        dalle_im = Image.open(dalle_stream)

        return (url, dalle_im)

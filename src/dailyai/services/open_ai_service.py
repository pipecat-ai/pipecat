from dailyai.services.ai_services import AIService, TTSService, LLMService, ImageGenService
from typing import Generator

import requests
from PIL import Image
import io
from openai import OpenAI

import os
import json

class OpenAILLMService(LLMService):
    def __init__(self, api_key=None, model=None):
        super().__init__()
        api_key = api_key or os.getenv("OPEN_AI_KEY")
        self.model = model or os.getenv("OPEN_AI_MODEL")
        self.client = OpenAI(api_key=api_key)

    def get_response(self, messages, stream):
        return self.client.chat.completions.create(
            stream=stream,
            messages=messages,
            model=self.model
        )

    def run_llm_async(self, messages) -> Generator[str, None, None]:
        messages_for_log = json.dumps(messages)
        self.logger.debug(f"Generating chat via openai: {messages_for_log}")

        response = self.get_response(messages, stream=True)

        for chunk in response:
            if len(chunk.choices) == 0:
                continue

            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def run_llm(self, messages) -> str | None:
        messages_for_log = json.dumps(messages)
        self.logger.debug(f"Generating chat via azure: {messages_for_log}")

        response = self.get_response(messages, stream=False)
        if response and len(response.choices) > 0:
            return response.choices[0].message.content
        else:
            return None

class OpenAIImageGenService(ImageGenService):
    def __init__(self, api_key=None, model=None):
        super().__init__()
        api_key = api_key or os.getenv("OPEN_AI_KEY")
        self.model = model or os.getenv("OPEN_AI_MODEL")
        self.client = OpenAI(api_key=api_key)

    def run_image_gen(self, sentence) -> tuple[str, Image.Image]:
        image = self.client.images.generate(
            prompt=sentence,
            n=1,
            size=f"1024x1024"
        )
        image_url = image.data[0].url
        response = requests.get(image_url)
        dalle_stream = io.BytesIO(response.content)
        dalle_im = Image.open(dalle_stream)

        return (image_url, dalle_im)

from services.ai_service import AIService
import requests
from PIL import Image
import io
import openai
import os
import time
import json

class OpenAIService(AIService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_llm(self, messages, latest_user_message=None, stream = True):
        local_messages = messages.copy()
        if latest_user_message:
            local_messages.append({"role": "user", "content": latest_user_message})
        messages_for_log = json.dumps(local_messages, indent=2)
        self.logger.info(f"==== generating chat via openai: {messages_for_log}")

        model = os.getenv("OPEN_AI_MODEL")
        if not model:
            model = "gpt-4"
        response = openai.ChatCompletion.create(
            api_type = 'openai',
            api_version = '2020-11-07',
            api_base = "https://api.openai.com/v1",
            api_key = os.getenv("OPEN_AI_KEY"),
            model=model,
            stream=stream,
            messages=local_messages
        )

        return response

    def run_image_gen(self, sentence):
        self.logger.info("🖌️ generating openai image async for ", sentence)
        start = time.time()

        image = openai.Image.create(
            api_type = 'openai',
            api_version = '2020-11-07',
            api_base = "https://api.openai.com/v1",
            api_key = os.getenv("OPEN_AI_KEY"),
            prompt=f'{sentence} in the style of {self.image_style}',
            n=1,
            size=f"1024x1024",
        )
        image_url = image["data"][0]["url"]
        self.logger.info("🖌️ generated image from url", image["data"][0]["url"])
        response = requests.get(image_url)
        self.logger.info("🖌️ got image from url", response)
        dalle_stream = io.BytesIO(response.content)
        dalle_im = Image.open(dalle_stream)
        self.logger.info("🖌️ total time", time.time() - start)

        return (image_url, dalle_im)

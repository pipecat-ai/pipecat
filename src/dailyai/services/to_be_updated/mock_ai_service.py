import io
import requests
import time
from PIL import Image
from services.ai_service import AIService

class MockAIService(AIService):
    def __init__(self):
        super().__init__()

    def run_tts(self, sentence):
        print("running tts", sentence)
        time.sleep(2)

    def run_image_gen(self, sentence):
        image_url = "https://d3d00swyhr67nd.cloudfront.net/w800h800/collection/ASH/ASHM/ASH_ASHM_WA1940_2_22-001.jpg"
        response = requests.get(image_url)
        image_stream = io.BytesIO(response.content)
        image = Image.open(image_stream)
        time.sleep(1)
        return (image_url, image)

    def run_llm(self, messages, latest_user_message=None, stream = True):
        for i in range(5):
            time.sleep(1)
            yield({"choices": [{"delta": {"content": f"hello {i}!"}}]})


import requests
import os
from services.ai_service import AIService

# Note that Cloudflare's AI workers are still in beta.
# https://developers.cloudflare.com/workers-ai/
class CloudflareAIService(AIService):
    def __init__(self):
        super().__init__()
        self.cloudflare_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.cloudflare_api_token = os.getenv("CLOUDFLARE_API_TOKEN")

        self.api_base_url = f'https://api.cloudflare.com/client/v4/accounts/{self.cloudflare_account_id}/ai/run/'
        self.headers = {"Authorization": f'Bearer {self.cloudflare_api_token}'}

    # base endpoint, used by the others
    def run(self, model, input):
        response = requests.post(f"{self.api_base_url}{model}", headers=self.headers, json=input)
        return response.json()

    # https://developers.cloudflare.com/workers-ai/models/llm/
    def run_llm(self, messages, latest_user_message=None, stream = True):
        input = {
            "messages": [
                { "role": "system", "content": "You are a friendly assistant" },
                { "role": "user", "content": sentence }
            ]
        }

        return self.run("@cf/meta/llama-2-7b-chat-int8", input)

    # https://developers.cloudflare.com/workers-ai/models/translation/
    def run_text_translation(self, sentence, source_language, target_language):
        return self.run('@cf/meta/m2m100-1.2b', {
            "text": sentence,
            "source_lang": source_language,
            "target_lang": target_language
        })

    # https://developers.cloudflare.com/workers-ai/models/sentiment-analysis/
    def run_text_sentiment(self, sentence):
        return self.run("@cf/huggingface/distilbert-sst-2-int8", {"text": sentence})

    # https://developers.cloudflare.com/workers-ai/models/image-classification/
    def run_image_classification(self, image_url):
        response = requests.get(image_url)

        if response.status_code != 200:
            return {"error": "There was a problem downloading the image."}

        if response.status_code == 200:
            data = response.content
            inputs = {"image": list(data)}

        return self.run("@cf/microsoft/resnet-50", inputs)

    # https://developers.cloudflare.com/workers-ai/models/embedding/
    def run_embeddings(self, texts, size="medium"):
        models = {
            "small": "@cf/baai/bge-small-en-v1.5", # 384 output dimensions
            "medium": "@cf/baai/bge-base-en-v1.5", # 768 output dimensions
            "large": "@cf/baai/bge-large-en-v1.5" #1024 output dimensions
        }

        return self.run(models[size], {"text": texts})

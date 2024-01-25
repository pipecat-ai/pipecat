from services.ai_service import AIService
from transformers import pipeline

# These functions are just intended for testing, not production use. If
# you'd like to use HuggingFace, you should use your own models, or do
# some research into the specific models that will work best for your use
# case.


class HuggingFaceAIService(AIService):
    def __init__(self):
        super().__init__()

    def run_text_sentiment(self, sentence):
        classifier = pipeline("sentiment-analysis")
        return classifier(sentence)

    # available models at https://huggingface.co/Helsinki-NLP (**not all
    # models use 2-character language codes**)
    def run_text_translation(self, sentence, source_language, target_language):
        translator = pipeline(
            f"translation",
            model=f"Helsinki-NLP/opus-mt-{source_language}-{target_language}")

        return translator(sentence)[0]["translation_text"]

    def run_text_summarization(self, sentence):
        summarizer = pipeline("summarization")
        return summarizer(sentence)

    def run_image_classification(self, image_path):
        classifier = pipeline("image-classification")
        return classifier(image_path)

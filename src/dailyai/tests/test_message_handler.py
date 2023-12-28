import time
import unittest

from unittest.mock import MagicMock, call

from dailyai.message_handler.message_handler import MessageHandler, IndexingMessageHandler
from dailyai.services.ai_services import (
    AIServiceConfig,
    TTSService,
    LLMService,
    ImageGenService,
)
from ..storage.search import SearchIndexer


class TestMessageHandler(unittest.TestCase):
    def test_simple_intro(self):
        message_handler = MessageHandler("Hello world")
        self.assertEqual(
            message_handler.get_llm_messages(),
            [{"role": "system", "content": "Hello world"}],
        )

    def test_simple_user_message(self):
        message_handler = MessageHandler("System prompt")
        message_handler.add_user_message("User message")
        self.assertEqual(
            message_handler.get_llm_messages(),
            [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User message"},
            ],
        )

    def test_simple_user_and_assistant_message(self):
        message_handler = MessageHandler("System prompt")
        message_handler.add_user_message("User message")
        message_handler.add_assistant_message("Assistant message")
        self.assertEqual(
            message_handler.get_llm_messages(),
            [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": "Assistant message"},
            ],
        )

    def test_user_message_overwrite(self):
        message_handler = MessageHandler("System prompt")
        message_handler.add_user_message("User message")
        message_handler.add_assistant_message("Assistant message")
        message_handler.add_user_message("plus something else")
        self.assertEqual(
            message_handler.get_llm_messages(),
            [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User message plus something else"},
            ],
        )

    def test_user_message_after_assistant(self):
        message_handler = MessageHandler("System prompt")
        message_handler.add_user_message("User message")
        message_handler.add_assistant_message("Assistant message")
        message_handler.finalize_user_message()
        message_handler.add_user_message("other user message")
        self.assertEqual(
            message_handler.get_llm_messages(),
            [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User message"},
                {"role": "assistant", "content": "Assistant message"},
                {"role": "user", "content": "other user message"},
            ],
        )


class MockTTSService(TTSService):
    def run_tts(self, sentence):
        for word in sentence.split(" "):
            time.sleep(0.1)
            yield bytes(word, "utf-8")


class MockLLMService(LLMService):
    def run_llm(self, messages) -> str:
        return "Parsed user message."

class MockImageService(ImageGenService):
    def run_image_gen(self, sentence) -> None:
        return None


class TestStorageMessageHandler(unittest.TestCase):
    def test_user_message_finalized(self):
        mock_tts_service = MockTTSService()
        mock_llm_service = MockLLMService()
        mock_image_service = MockImageService()

        service_config = AIServiceConfig(
            tts=mock_tts_service, llm=mock_llm_service, image=mock_image_service
        )

        mock_indexer = MagicMock(spec=SearchIndexer)

        message_handler = IndexingMessageHandler(
            "Hello world", service_config, mock_indexer
        )
        message_handler.cleanup_user_message = MagicMock(return_value="Parsed user message.")
        message_handler.add_user_message("User message")
        message_handler.add_assistant_message("Assistant message will be ignored")
        message_handler.add_user_message("plus something else")
        message_handler.finalize_user_message()
        message_handler.add_assistant_message(
            "New assistant message will not be ignored"
        )
        message_handler.add_user_message("User message second time")
        message_handler.add_assistant_message("Assistant message second time")
        message_handler.write_messages_to_storage()

        time.sleep(0.5)
        message_handler.cleanup_user_message.assert_called_with("User message plus something else")
        self.assertEqual(
            mock_indexer.mock_calls,
            [
                call.index_text('"Parsed user message."'),
                call.index_text("New assistant message will not be ignored"),
            ],
        )

        mock_indexer.reset_mock()

        message_handler.finalize_user_message()

        time.sleep(0.5)

        self.assertEqual(
            mock_indexer.mock_calls,
            [
                call.index_text('"Parsed user message."'),
                call.index_text("Assistant message second time"),
            ],
        )


if __name__ == "__main__":
    unittest.main()

import time
import unittest

from queue import Queue, Empty
from threading import Thread, Event
from typing import Generator

from dailyai.services.ai_services import (
    AIServiceConfig,
    ImageGenService,
    LLMService,
    TTSService
)
from dailyai.message_handler.message_handler import MessageHandler
from dailyai.async_processor.async_processor import (
    AsyncProcessor,
    AsyncProcessorState,
    Response,
)

class MockTTSService(TTSService):
    def run_tts(self, sentence):
        for word in sentence.split(' '):
            time.sleep(0.1)
            yield bytes(word, "utf-8")

class MockLLMService(LLMService):
    def run_llm_async(self, messages) -> Generator[str, None, None]:
        for i in ["Hello ", "there.", "How are ", "you?", "I ", "hope ", "you ", "are ", "well."]:
            time.sleep(0.1)
            yield i

class MockImageService(ImageGenService):
    def run_image_gen(self, sentence) -> None:
        return None

class TestResponse(unittest.TestCase):
    def test_base_state_transitions(self):
        mock_tts_service = MockTTSService()
        mock_llm_service = MockLLMService()
        mock_image_service = MockImageService()
        processor = AsyncProcessor(AIServiceConfig(tts=mock_tts_service, llm=mock_llm_service, image=mock_image_service))
        processor.prepare()
        processor.play()
        processor.finalize()
        self.assertEqual(processor.state, AsyncProcessorState.FINALIZED)

    def test_state_transitions(self):
        output_queue = Queue()
        mock_tts_service = MockTTSService()
        mock_llm_service = MockLLMService()
        mock_image_service = MockImageService()
        message_handler = MessageHandler("Hello World")
        processor = Response(
            AIServiceConfig(
                tts=mock_tts_service, llm=mock_llm_service, image=mock_image_service
            ),
            message_handler,
            output_queue,
        )
        processor.prepare()
        processor.play()

        # Consume the output from the output queue. It's necessary to mark these tasks as done for the
        # play function to return.
        expected_words = ["Hello", "there.", "How", "are", "you?", "I", "hope", "you", "are", "well."]

        # remove the "start_stream" message from the queue
        output_queue.get()
        output_queue.task_done()

        while expected_words:
            actual_word = output_queue.get()
            word = expected_words.pop(0)
            self.assertEqual(actual_word['type'], 'audio_frame')
            self.assertEqual(actual_word['data'], bytes(word, "utf-8"))
            output_queue.task_done()

        processor.finalize()

        self.assertEqual(processor.state, AsyncProcessorState.FINALIZED)

    def test_interrupt_preparation(self):
        output_queue = Queue()
        mock_tts_service = MockTTSService()
        mock_llm_service = MockLLMService()
        mock_image_service = MockImageService()
        message_handler = MessageHandler("System Message")
        processor = Response(
            AIServiceConfig(
                tts=mock_tts_service, llm=mock_llm_service, image=mock_image_service
            ),
            message_handler,
            output_queue,
        )
        processor.prepare()
        interrupt_request_at = time.perf_counter()
        processor.interrupt()
        processor.finalize()
        finalized_at = time.perf_counter()
        self.assertTrue(0.1 < finalized_at - interrupt_request_at < 0.2)
        print(f"delta: {interrupt_request_at, finalized_at}")
        self.assertEqual(processor.state, AsyncProcessorState.FINALIZED)

    def test_interrupt_play(self):
        output_queue = Queue()
        mock_tts_service = MockTTSService()
        mock_llm_service = MockLLMService()
        mock_image_service = MockImageService()
        message_handler = MessageHandler("System Message")
        processor = Response(
            AIServiceConfig(
                tts=mock_tts_service, llm=mock_llm_service, image=mock_image_service
            ),
            message_handler,
            output_queue,
        )
        processor.prepare()
        processor.play()

        stop_processing_output_queue = Event()
        def process_output_queue_async():
            # Consume the output from the output queue. It's necessary to mark these tasks as done for the
            # play function to return.
            time.sleep(0.1)
            expected_words = ["Hello", "there.", "How", "are", "you?", "I", "hope", "you", "are", "well."]
            while expected_words and not stop_processing_output_queue.is_set():
                try:
                    actual_word = output_queue.get_nowait()
                    if actual_word['type'] == 'audio_frame':
                        time.sleep(0.1)
                        word = expected_words.pop(0)
                        self.assertEqual(actual_word['type'], 'audio_frame')
                        self.assertEqual(actual_word['data'], bytes(word, "utf-8"))
                    output_queue.task_done()
                except Empty:
                    pass

        process_output_queue = Thread(target=process_output_queue_async, daemon=True)
        process_output_queue.start()

        time.sleep(0.5)
        processor.interrupt()

        stop_processing_output_queue.set()
        process_output_queue.join()

        processor.finalize()
        self.assertEqual(processor.state, AsyncProcessorState.FINALIZED)

    def test_statechange_callback(self):
        mock_tts_service = MockTTSService()
        mock_llm_service = MockLLMService()
        mock_image_service = MockImageService()
        processor = AsyncProcessor(
            AIServiceConfig(
                tts=mock_tts_service, llm=mock_llm_service, image=mock_image_service
            )
        )
        is_finalized = False
        def set_is_finalized(async_processor:AsyncProcessor):
            nonlocal is_finalized
            is_finalized = True

        processor.set_state_callback(
            AsyncProcessorState.FINALIZED, set_is_finalized
        )
        processor.prepare()
        self.assertFalse(is_finalized)
        processor.play()
        self.assertFalse(is_finalized)
        processor.finalize()
        self.assertTrue(is_finalized)
        self.assertEqual(processor.state, AsyncProcessorState.FINALIZED)


if __name__ == '__main__':
    unittest.main()

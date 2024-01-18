from typing import AsyncGenerator

from dailyai.queue_frame import FrameType, QueueFrame
from dailyai.services.ai_services import AIService

class SentenceAggregator(AIService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_sentence = ""

    def allowed_input_frame_types(self) -> set[FrameType]:
        return set([FrameType.TEXT_CHUNK, FrameType.SENTENCE])

    def possible_output_frame_types(self) -> set[FrameType]:
        return set([FrameType.SENTENCE])

    async def process_frame(
        self, requested_frame_types: set[FrameType], frame: QueueFrame
    ) -> AsyncGenerator[QueueFrame, None]:
        if not FrameType.SENTENCE in requested_frame_types:
            return

        if frame.frame_type == FrameType.TEXT_CHUNK:
            if type(frame.frame_data) != str:
                raise Exception(
                    "Sentence aggregator requires a string for the data field"
                )

            self.current_sentence += frame.frame_data
            if self.current_sentence.endswith((".", "?", "!")):
                sentence = self.current_sentence
                self.current_sentence = ""
                yield QueueFrame(FrameType.SENTENCE, sentence)
        elif frame.frame_type == FrameType.END_STREAM:
            if self.current_sentence:
                yield QueueFrame(FrameType.SENTENCE, self.current_sentence)
        elif frame.frame_type == FrameType.SENTENCE:
            yield frame


class TranscriptionSentenceAggregator(AIService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_sentence = ""

    def allowed_input_frame_types(self) -> set[FrameType]:
        return set([FrameType.TEXT_CHUNK, FrameType.SENTENCE])

    def possible_output_frame_types(self) -> set[FrameType]:
        return set([FrameType.SENTENCE])

    async def process_frame(
        self, requested_frame_types: set[FrameType], frame: QueueFrame
    ) -> AsyncGenerator[QueueFrame, None]:
        if not FrameType.SENTENCE in requested_frame_types:
            return

        if frame.frame_type == FrameType.TEXT_CHUNK:
            if type(frame.frame_data) != str:
                raise Exception(
                    "Sentence aggregator requires a string for the data field"
                )

            self.current_sentence += frame.frame_data
            if self.current_sentence.endswith((".", "?", "!")):
                sentence = self.current_sentence
                self.current_sentence = ""
                yield QueueFrame(FrameType.SENTENCE, sentence)
        elif frame.frame_type == FrameType.END_STREAM:
            if self.current_sentence:
                yield QueueFrame(FrameType.SENTENCE, self.current_sentence)
        elif frame.frame_type == FrameType.SENTENCE:
            yield frame

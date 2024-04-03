from typing import AsyncGenerator, Callable
from dailyai.pipeline.frame_processor import FrameProcessor
from dailyai.pipeline.frames import (
    Frame,
    LLMResponseEndFrame,
    LLMResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from dailyai.pipeline.openai_frames import OpenAILLMContextFrame
from dailyai.services.openai_llm_context import OpenAILLMContext

try:
    from openai.types.chat import ChatCompletionRole
except ModuleNotFoundError as e:
    print(f"Exception: {e}")
    print(
        "In order to use OpenAI, you need to `pip install dailyai[openai]`. Also, set `OPENAI_API_KEY` environment variable.")
    raise Exception(f"Missing module: {e}")


class OpenAIContextAggregator(FrameProcessor):

    def __init__(
        self,
        context: OpenAILLMContext,
        aggregator: Callable[[Frame, str | None], str | None],
        role: ChatCompletionRole,
        start_frame: type,
        end_frame: type,
        accumulator_frame: type,
        pass_through=True,
    ):
        if not (
            issubclass(start_frame, Frame)
            and issubclass(end_frame, Frame)
            and issubclass(accumulator_frame, Frame)
        ):
            raise TypeError(
                "start_frame, end_frame and accumulator_frame must be instances of Frame"
            )

        self._context: OpenAILLMContext = context
        self._aggregator: Callable[[Frame, str | None], None] = aggregator
        self._role: ChatCompletionRole = role
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._accumulator_frame = accumulator_frame
        self._pass_through = pass_through

        self._aggregating = False
        self._aggregation = None

    async def process_frame(self, frame: Frame) -> AsyncGenerator[Frame, None]:
        if isinstance(frame, self._start_frame):
            self._aggregating = True
        elif isinstance(frame, self._end_frame):
            self._aggregating = False
            if self._aggregation:
                self._context.add_message(
                    {
                        "role": self._role,
                        "content": self._aggregation,
                        "name": self._role,
                    }  # type: ignore
                )
            self._aggregation = None
            yield OpenAILLMContextFrame(self._context)
        elif isinstance(frame, self._accumulator_frame) and self._aggregating:
            self._aggregation = self._aggregator(frame, self._aggregation)
            if self._pass_through:
                yield frame
        else:
            yield frame

    def string_aggregator(
            self,
            frame: Frame,
            aggregation: str | None) -> str | None:
        if not isinstance(frame, TextFrame):
            raise TypeError(
                "Frame must be a TextFrame instance to be aggregated by a string aggregator."
            )
        if not aggregation:
            aggregation = ""
        return " ".join([aggregation, frame.text])


class OpenAIUserContextAggregator(OpenAIContextAggregator):
    def __init__(self, context: OpenAILLMContext):
        super().__init__(
            context=context,
            aggregator=self.string_aggregator,
            role="user",
            start_frame=UserStartedSpeakingFrame,
            end_frame=UserStoppedSpeakingFrame,
            accumulator_frame=TranscriptionFrame,
            pass_through=False,
        )


class OpenAIAssistantContextAggregator(OpenAIContextAggregator):

    def __init__(self, context: OpenAILLMContext):
        super().__init__(
            context,
            aggregator=self.string_aggregator,
            role="assistant",
            start_frame=LLMResponseStartFrame,
            end_frame=LLMResponseEndFrame,
            accumulator_frame=TextFrame,
            pass_through=True,
        )

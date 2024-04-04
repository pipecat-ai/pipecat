from dataclasses import dataclass

from dailyai.pipeline.frames import Frame
from dailyai.services.openai_llm_context import OpenAILLMContext


@dataclass()
class OpenAILLMContextFrame(Frame):
    """Like an LLMMessagesFrame, but with extra context specific to the
    OpenAI API. The context in this message is also mutable, and will be
    changed by the OpenAIContextAggregator frame processor."""
    context: OpenAILLMContext

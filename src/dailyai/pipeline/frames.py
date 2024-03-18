from dataclasses import dataclass
from typing import Any, List

from dailyai.services.openai_llm_context import OpenAILLMContext


class Frame:
    def __str__(self):
        return f"{self.__class__.__name__}"


class ControlFrame(Frame):
    # Control frames should contain no instance data, so
    # equality is based solely on the class.
    def __eq__(self, other):
        return type(other) == self.__class__


class StartFrame(ControlFrame):
    pass


class EndFrame(ControlFrame):
    pass


class EndPipeFrame(ControlFrame):
    pass


class PipelineStartedFrame(ControlFrame):
    """
    Used by the transport to indicate that execution of a pipeline is starting
    (or restarting). It should be the first frame your app receives when it
    starts, or when an interruptible pipeline has been interrupted.
    """

    pass


class LLMResponseStartFrame(ControlFrame):
    pass


class LLMResponseEndFrame(ControlFrame):
    pass


@dataclass()
class AudioFrame(Frame):
    data: bytes

    def __str__(self):
        return f"{self.__class__.__name__}, size: {len(self.data)} B"


@dataclass()
class ImageFrame(Frame):
    url: str | None
    image: bytes

    def __str__(self):
        return f"{self.__class__.__name__}, url: {self.url}, image size: {len(self.image)} B"


@dataclass()
class SpriteFrame(Frame):
    images: list[bytes]

    def __str__(self):
        return f"{self.__class__.name__}, list size: {len(self.images)}"


@dataclass()
class TextFrame(Frame):
    text: str

    def __str__(self):
        return f'{self.__class__.__name__}: "{self.text}"'


@dataclass()
class TranscriptionQueueFrame(TextFrame):
    participantId: str
    timestamp: str


@dataclass()
class LLMMessagesQueueFrame(Frame):
    messages: List[dict]


@dataclass()
class OpenAILLMContextFrame(Frame):
    context: OpenAILLMContext


@dataclass()
class ReceivedAppMessageFrame(Frame):
    message: Any
    sender: str

    def __str__(self):
        return f"ReceivedAppMessageFrame: sender: {self.sender}, message: {self.message}"


@dataclass()
class SendAppMessageFrame(Frame):
    message: Any
    participantId: str | None

    def __str__(self):
        return f"SendAppMessageFrame: participantId: {self.participantId}, message: {self.message}"


class UserStartedSpeakingFrame(Frame):
    pass


class UserStoppedSpeakingFrame(Frame):
    pass


class BotStartedSpeakingFrame(Frame):
    pass


class BotStoppedSpeakingFrame(Frame):
    pass


@dataclass()
class LLMFunctionStartFrame(Frame):
    function_name: str


@dataclass()
class LLMFunctionCallFrame(Frame):
    function_name: str
    arguments: str

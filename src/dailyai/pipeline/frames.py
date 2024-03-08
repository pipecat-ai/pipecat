from dataclasses import dataclass
from typing import Any


class Frame:
    pass

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


class LLMResponseStartFrame(ControlFrame):
    pass


class LLMResponseEndFrame(ControlFrame):
    pass


@dataclass()
class AudioFrame(Frame):
    data: bytes


@dataclass()
class ImageFrame(Frame):
    url: str | None
    image: bytes


@dataclass()
class SpriteFrame(Frame):
    images: list[bytes]


@dataclass()
class TextFrame(Frame):
    text: str


@dataclass()
class TranscriptionQueueFrame(TextFrame):
    participantId: str
    timestamp: str


@dataclass()
class LLMMessagesQueueFrame(Frame):
    messages: list[dict[str, str]]  # TODO: define this more concretely!


class AppMessageQueueFrame(Frame):
    message: Any
    participantId: str

class UserStartedSpeakingFrame(Frame):
    pass

class UserStoppedSpeakingFrame(Frame):
    pass

@dataclass()
class LLMFunctionStartFrame(Frame):
    function_name: str
@dataclass()
class LLMFunctionCallFrame(Frame):
    function_name: str
    arguments: str
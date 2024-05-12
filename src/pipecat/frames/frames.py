#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Any, List

from pipecat.utils.utils import obj_count, obj_id


class Frame:
    def __init__(self, data=None):
        self.id: int = obj_id()
        self.data: Any = data
        self.metadata = {}
        self.name: str = f"{self.__class__.__name__}#{obj_count(self)}"

    def __str__(self):
        return self.name


class DataFrame(Frame):
    def __init__(self, data):
        super().__init__(data)


class AudioRawFrame(DataFrame):
    def __init__(self, data, sample_rate: int, num_channels: int):
        super().__init__(data)
        self.metadata["sample_rate"] = sample_rate
        self.metadata["num_channels"] = num_channels
        self.metadata["num_frames"] = int(len(data) / (num_channels * 2))

    @property
    def num_frames(self) -> int:
        return self.metadata["num_frames"]

    @property
    def sample_rate(self) -> int:
        return self.metadata["sample_rate"]

    @property
    def num_channels(self) -> int:
        return self.metadata["num_channels"]

    def __str__(self):
        return f"{self.name}(frames: {self.num_frames}, sample_rate: {self.sample_rate}, channels: {self.num_channels})"


class ImageRawFrame(DataFrame):
    def __init__(self, data, size: tuple[int, int], format: str):
        super().__init__(data)
        self.metadata["size"] = size
        self.metadata["format"] = format

    @property
    def image(self) -> bytes:
        return self.data

    @property
    def size(self) -> tuple[int, int]:
        return self.metadata["size"]

    @property
    def format(self) -> str:
        return self.metadata["format"]

    def __str__(self):
        return f"{self.name}(size: {self.size}, format: {self.format})"


class URLImageRawFrame(ImageRawFrame):
    def __init__(self, url: str, data, size: tuple[int, int], format: str):
        super().__init__(data, size, format)
        self.metadata["url"] = url

    @property
    def url(self) -> str:
        return self.metadata["url"]

    def __str__(self):
        return f"{self.name}(url: {self.url}, size: {self.size}, format: {self.format})"


class VisionImageRawFrame(ImageRawFrame):
    def __init__(self, text: str, data, size: tuple[int, int], format: str):
        super().__init__(data, size, format)
        self.metadata["text"] = text

    @property
    def text(self) -> str:
        return self.metadata["text"]

    def __str__(self):
        return f"{self.name}(text: {self.text}, size: {self.size}, format: {self.format})"


class UserImageRawFrame(ImageRawFrame):
    def __init__(self, user_id: str, data, size: tuple[int, int], format: str):
        super().__init__(data, size, format)
        self.metadata["user_id"] = user_id

    @property
    def user_id(self) -> str:
        return self.metadata["user_id"]

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, size: {self.size}, format: {self.format})"


class SpriteFrame(Frame):
    def __init__(self, data):
        super().__init__(data)

    @property
    def images(self) -> List[ImageRawFrame]:
        return self.data

    def __str__(self):
        return f"{self.name}(size: {len(self.images)})"


class TextFrame(DataFrame):
    def __init__(self, data):
        super().__init__(data)

    @property
    def text(self) -> str:
        return self.data


class TranscriptionFrame(TextFrame):
    def __init__(self, data, user_id: str, timestamp: int):
        super().__init__(data)
        self.metadata["user_id"] = user_id
        self.metadata["timestamp"] = timestamp

    @property
    def user_id(self) -> str:
        return self.metadata["user_id"]

    @property
    def timestamp(self) -> str:
        return self.metadata["timestamp"]

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, timestamp: {self.timestamp})"


class InterimTranscriptionFrame(TextFrame):
    def __init__(self, data, user_id: str, timestamp: int):
        super().__init__(data)
        self.metadata["user_id"] = user_id
        self.metadata["timestamp"] = timestamp

    @property
    def user_id(self) -> str:
        return self.metadata["user_id"]

    @property
    def timestamp(self) -> str:
        return self.metadata["timestamp"]

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, timestamp: {self.timestamp})"


class LLMMessagesFrame(DataFrame):
    """A frame containing a list of LLM messages. Used to signal that an LLM
    service should run a chat completion and emit an LLM started response event,
    text frames and an LLM stopped response event.
    """

    def __init__(self, messages):
        super().__init__(messages)

#
# App frames. Application user-defined frames.
#


class AppFrame(Frame):
    def __init__(self, data=None):
        super().__init__(data)


#
# System frames
#

class SystemFrame(Frame):
    def __init__(self, data=None):
        super().__init__(data)


class StartFrame(SystemFrame):
    def __init__(self):
        super().__init__()


class CancelFrame(SystemFrame):
    def __init__(self):
        super().__init__()


class ErrorFrame(SystemFrame):
    def __init__(self, data):
        super().__init__(data)
        self.metadata["error"] = data

    @property
    def error(self) -> str:
        return self.metadata["error"]

    def __str__(self):
        return f"{self.name}(error: {self.error})"

#
# Control frames
#


class ControlFrame(Frame):
    def __init__(self, data=None):
        super().__init__(data)


class EndFrame(ControlFrame):
    def __init__(self):
        super().__init__()


class LLMResponseStartFrame(ControlFrame):
    """Used to indicate the beginning of an LLM response. Following TextFrames
    are part of the LLM response until an LLMResponseEndFrame"""

    def __init__(self):
        super().__init__()


class LLMResponseEndFrame(ControlFrame):
    """Indicates the end of an LLM response."""

    def __init__(self):
        super().__init__()


class UserStartedSpeakingFrame(ControlFrame):
    def __init__(self):
        super().__init__()


class UserStoppedSpeakingFrame(ControlFrame):
    def __init__(self):
        super().__init__()


class TTSStartedFrame(ControlFrame):
    def __init__(self):
        super().__init__()


class TTSStoppedFrame(ControlFrame):
    def __init__(self):
        super().__init__()


class UserImageRequestFrame(ControlFrame):
    def __init__(self, user_id):
        super().__init__()
        self.metadata["user_id"] = user_id

    @property
    def user_id(self) -> str:
        return self.metadata["user_id"]

    def __str__(self):
        return f"{self.name}, user: {self.user_id}"


# class StartFrame(ControlFrame):
#     """Used (but not required) to start a pipeline, and is also used to
#     indicate that an interruption has ended and the transport should start
#     processing frames again."""
#     pass


# class EndFrame(ControlFrame):
#     """Indicates that a pipeline has ended and frame processors and pipelines
#     should be shut down. If the transport receives this frame, it will stop
#     sending frames to its output channel(s) and close all its threads."""
#     pass


# class EndPipeFrame(ControlFrame):
#     """Indicates that a pipeline has ended but that the transport should
#     continue processing. This frame is used in parallel pipelines and other
#     sub-pipelines."""
#     pass


# class PipelineStartedFrame(ControlFrame):
#     """
#     Used by the transport to indicate that execution of a pipeline is starting
#     (or restarting). It should be the first frame your app receives when it
#     starts, or when an interruptible pipeline has been interrupted.
#     """

#     pass


# @dataclass()
# class URLImageFrame(ImageFrame):
#     """An image with an associated URL. Will be shown by the transport if the
#     transport's camera is enabled.

#     """
#     url: str | None

#     def __init__(self, url, image, size):
#         super().__init__(image, size)
#         self.url = url

#     def __str__(self):
# return f"{self.__class__.__name__}, url: {self.url}, image size:
# {self.size[0]}x{self.size[1]}, buffer size: {len(self.image)} B"


# @dataclass()
# class VisionImageFrame(ImageFrame):
#     """An image with an associated text to ask for a description of it. Will be shown by the
#     transport if the transport's camera is enabled.

#     """
#     text: str | None

#     def __init__(self, text, image, size):
#         super().__init__(image, size)
#         self.text = text

#     def __str__(self):
# return f"{self.__class__.__name__}, text: {self.text}, image size:
# {self.size[0]}x{self.size[1]}, buffer size: {len(self.image)} B"


# @dataclass()
# class UserImageFrame(ImageFrame):
#     """An image associated to a user. Will be shown by the transport if the transport's camera is
#     enabled."""
#     user_id: str

#     def __init__(self, user_id, image, size):
#         super().__init__(image, size)
#         self.user_id = user_id

#     def __str__(self):
# return f"{self.__class__.__name__}, user: {self.user_id}, image size:
# {self.size[0]}x{self.size[1]}, buffer size: {len(self.image)} B"


# @dataclass()
# class UserImageRequestFrame(Frame):
#     """A frame user to request an image from the given user."""
#     user_id: str

#     def __str__(self):
#         return f"{self.__class__.__name__}, user: {self.user_id}"


# @dataclass()
# class SpriteFrame(Frame):
#     """An animated sprite. Will be shown by the transport if the transport's
#     camera is enabled. Will play at the framerate specified in the transport's
#     `fps` constructor parameter."""
#     images: list[bytes]

#     def __str__(self):
#         return f"{self.__class__.__name__}, list size: {len(self.images)}"


# @dataclass()
# class TextFrame(Frame):
#     """A chunk of text. Emitted by LLM services, consumed by TTS services, can
#     be used to send text through pipelines."""
#     text: str

#     def __str__(self):
#         return f'{self.__class__.__name__}: "{self.text}"'


# class TTSStartFrame(ControlFrame):
#     """Used to indicate the beginning of a TTS response. Following AudioFrames
#     are part of the TTS response until an TTEndFrame. These frames can be used
#     for aggregating audio frames in a transport to optimize the size of frames
#     sent to the session, without needing to control this in the TTS service."""
#     pass


# class TTSEndFrame(ControlFrame):
#     """Indicates the end of a TTS response."""
#     pass


# @dataclass()
# class LLMMessagesFrame(Frame):
#     """A frame containing a list of LLM messages. Used to signal that an LLM
#     service should run a chat completion and emit an LLMStartFrames, TextFrames
#     and an LLMEndFrame.
#     Note that the messages property on this class is mutable, and will be
#     be updated by various ResponseAggregator frame processors."""
#     messages: List[dict]


# @dataclass()
# class ReceivedAppMessageFrame(Frame):
#     message: Any
#     sender: str

#     def __str__(self):
#         return f"ReceivedAppMessageFrame: sender: {self.sender}, message: {self.message}"


# @dataclass()
# class SendAppMessageFrame(Frame):
#     message: Any
#     participant_id: str | None

#     def __str__(self):
#         return f"SendAppMessageFrame: participant: {self.participant_id}, message: {self.message}"


# class UserStartedSpeakingFrame(Frame):
#     """Emitted by VAD to indicate that a participant has started speaking.
#     This can be used for interruptions or other times when detecting that
#     someone is speaking is more important than knowing what they're saying
#     (as you will with a TranscriptionFrame)"""
#     pass


# class UserStoppedSpeakingFrame(Frame):
#     """Emitted by the VAD to indicate that a user stopped speaking."""
#     pass


# class BotStartedSpeakingFrame(Frame):
#     pass


# class BotStoppedSpeakingFrame(Frame):
#     pass


# @dataclass()
# class LLMFunctionStartFrame(Frame):
#     """Emitted when the LLM receives the beginning of a function call
#     completion. A frame processor can use this frame to indicate that it should
#     start preparing to make a function call, if it can do so in the absence of
#     any arguments."""
#     function_name: str


# @dataclass()
# class LLMFunctionCallFrame(Frame):
#     """Emitted when the LLM has received an entire function call completion."""
#     function_name: str
#     arguments: str

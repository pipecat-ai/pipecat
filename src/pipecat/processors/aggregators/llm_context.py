#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.frames.frames import Frame, InterimTranscriptionFrame, LLMMessagesFrame, TextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class LLMContextAggregator(FrameProcessor):
    def __init__(
        self,
        messages: list[dict],
        role: str,
        complete_sentences=True,
        pass_through=True,
    ):
        super().__init__()
        self._messages = messages
        self._role = role
        self._sentence = ""
        self._complete_sentences = complete_sentences
        self._pass_through = pass_through

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # We don't do anything with non-text frames, pass it along to next in
        # the pipeline.
        if not isinstance(frame, TextFrame):
            await self.push_frame(frame, direction)
            return

        # If we get interim results, we ignore them.
        if isinstance(frame, InterimTranscriptionFrame):
            return

        # The common case for "pass through" is receiving frames from the LLM that we'll
        # use to update the "assistant" LLM messages, but also passing the text frames
        # along to a TTS service to be spoken to the user.
        if self._pass_through:
            await self.push_frame(frame, direction)

        # TODO: split up transcription by participant
        if self._complete_sentences:
            # type: ignore -- the linter thinks this isn't a TextFrame, even
            # though we check it above
            self._sentence += frame.text
            if self._sentence.endswith((".", "?", "!")):
                self._messages.append(
                    {"role": self._role, "content": self._sentence})
                self._sentence = ""
                await self.push_frame(LLMMessagesFrame(self._messages))
        else:
            # type: ignore -- the linter thinks this isn't a TextFrame, even
            # though we check it above
            self._messages.append({"role": self._role, "content": frame.text})
            await self.push_frame(LLMMessagesFrame(self._messages))


class LLMUserContextAggregator(LLMContextAggregator):
    def __init__(
            self,
            messages: list[dict],
            complete_sentences=True):
        super().__init__(
            messages,
            "user",
            complete_sentences,
            pass_through=False)


class LLMAssistantContextAggregator(LLMContextAggregator):
    def __init__(
            self,
            messages: list[dict],
            complete_sentences=True):
        super().__init__(
            messages,
            "assistant",
            complete_sentences,
            pass_through=True,
        )

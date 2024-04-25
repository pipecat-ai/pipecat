#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.frames.frames import Frame


class OpenAILLMContextFrame(Frame):
    """Like an LLMMessagesFrame, but with extra context specific to the
    OpenAI API."""

    def __init__(self, data):
        super().__init__(data)

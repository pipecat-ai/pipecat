#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import io
import unittest

from loguru import logger

from pipecat.frames.frames import InputAudioRawFrame, TextFrame
from pipecat.processors.logger import FrameLogger
from pipecat.tests.utils import run_test


class TestFrameLogger(unittest.IsolatedAsyncioTestCase):
    async def _logged(self, frame_logger: FrameLogger, frames) -> str:
        sink = io.StringIO()
        handler_id = logger.add(sink, level="DEBUG", format="{message}")
        try:
            await run_test(frame_logger, frames_to_send=frames)
        finally:
            logger.remove(handler_id)
        return sink.getvalue()

    async def test_default_ignores_audio_frames(self):
        audio = InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)

        logged = await self._logged(FrameLogger(), [TextFrame(text="hello"), audio])

        self.assertIn("Frame: TextFrame", logged)
        self.assertNotIn("InputAudioRawFrame", logged)

    async def test_empty_ignored_frame_types_logs_every_frame(self):
        audio = InputAudioRawFrame(audio=b"\x00\x00", sample_rate=16000, num_channels=1)

        logged = await self._logged(
            FrameLogger(ignored_frame_types=()), [TextFrame(text="hello"), audio]
        )

        self.assertIn("Frame: TextFrame", logged)
        self.assertIn("Frame: InputAudioRawFrame", logged)

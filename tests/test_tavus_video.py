#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for TavusVideoService."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tavus.video import TavusVideoService
from pipecat.tests.utils import run_test


def _fake_tavus_client(*args, **kwargs) -> MagicMock:
    client = MagicMock()
    for method in ("setup", "join", "cleanup", "start_send_task", "cancel_send_task", "stop"):
        setattr(client, method, AsyncMock())
    return client


class TestTavusVideoService(unittest.IsolatedAsyncioTestCase):
    async def test_bot_speaking_frames_pass_upstream(self):
        with patch("pipecat.services.tavus.video.TavusTransportClient", _fake_tavus_client):
            service = TavusVideoService(api_key="key", replica_id="replica", session=MagicMock())
            await run_test(
                service,
                frames_to_send=[BotStartedSpeakingFrame(), BotStoppedSpeakingFrame()],
                frames_to_send_direction=FrameDirection.UPSTREAM,
                expected_up_frames=[BotStartedSpeakingFrame, BotStoppedSpeakingFrame],
            )


if __name__ == "__main__":
    unittest.main()

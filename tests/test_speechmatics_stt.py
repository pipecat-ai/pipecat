#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from unittest.mock import MagicMock

import pytest

from pipecat.frames.frames import VADUserStoppedSpeakingFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.services.stt_service import STTService


@pytest.mark.asyncio
async def test_vad_user_stopped_defers_finalize(monkeypatch):
    """``finalize()`` must be deferred on ``VADUserStoppedSpeakingFrame``, not called immediately.

    Speechmatics' server may still be delivering the final tokens of a turn when VAD reports the
    user stopped. Calling ``finalize()`` right away truncates those trailing tokens (issue #4484),
    so the service schedules the call after a short aggregation delay instead.
    """
    service = SpeechmaticsSTTService(api_key="test-key")
    service._enable_vad = False
    service._client = MagicMock()  # the real VoiceAgentClient.finalize() is synchronous
    service._finalize_task = None

    scheduled = []

    def fake_create_task(coro, name=None):
        scheduled.append(coro)
        return MagicMock()

    # The VAD branch is all we exercise here; skip the full FrameProcessor pipeline machinery.
    async def noop_process_frame(self, frame, direction):
        pass

    monkeypatch.setattr(STTService, "process_frame", noop_process_frame, raising=False)
    monkeypatch.setattr(service, "create_task", fake_create_task)
    monkeypatch.setattr(service, "request_finalize", MagicMock())

    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    # The fix: finalize() is scheduled on a task, not invoked synchronously.
    service._client.finalize.assert_not_called()
    assert len(scheduled) == 1

    # Running the deferred task waits the aggregation delay, then finalizes the turn.
    await scheduled[0]
    service._client.finalize.assert_called_once()

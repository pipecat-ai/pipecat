from unittest.mock import AsyncMock, patch

import pytest

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi import RTVIServerMessageFrame

from ..processors.latency import LatencyProcessor


@pytest.mark.asyncio
async def test_latency_emitted_only_after_user_and_bot_frames():
    # Arrange: create processor with dummy transport/rtvi
    transport = AsyncMock()
    rtvi = AsyncMock()
    processor = LatencyProcessor(transport, rtvi)

    # Spy on push_frame to capture what it emits
    processor.push_frame = AsyncMock()

    # control time.monotonic(): user stops at t=2.0, bot starts at t=2.3
    with patch("time.monotonic", side_effect=[2.0, 2.3]):
        # Act: feed in the two frames
        await processor.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await processor.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    # Assert: among the calls to push_frame there is exactly one RTVIServerMessageFrame
    calls = [c.args[0] for c in processor.push_frame.call_args_list]
    # find any RTVIServerMessageFrame
    rtvi_msgs = [f for f in calls if isinstance(f, RTVIServerMessageFrame)]
    assert len(rtvi_msgs) == 1, "Expected exactly one RTVIServerMessageFrame"
    latency_frame = rtvi_msgs[0]
    # check the latency_ms ≈ 300.0
    assert pytest.approx(latency_frame.data["latency_ms"], rel=1e-3) == 300.0


@pytest.mark.asyncio
async def test_no_latency_if_bot_before_user():
    transport = AsyncMock()
    rtvi = AsyncMock()
    processor = LatencyProcessor(transport, rtvi)
    processor.push_frame = AsyncMock()

    # If BotStartedSpeakingFrame arrives first, no latency should be emitted.
    with patch("time.monotonic", return_value=5.0):
        await processor.process_frame(BotStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    # Assert: no RTVIServerMessageFrame in any push_frame call
    calls = [c.args[0] for c in processor.push_frame.call_args_list]
    assert not any(isinstance(f, RTVIServerMessageFrame) for f in calls)

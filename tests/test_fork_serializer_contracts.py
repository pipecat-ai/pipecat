#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Contract tests for Dograh-owned telephony serializers."""

from types import SimpleNamespace

import pytest

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import OutputTransportMessageFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.serializers.asterisk import AsteriskFrameSerializer
from pipecat.serializers.vobiz import VobizFrameSerializer
from pipecat.utils.asyncio.task_manager import TaskManager


def _setup(*, sample_rate: int = 16000) -> FrameProcessorSetup:
    return FrameProcessorSetup(
        audio_in_sample_rate=sample_rate,
        clock=SystemClock(),
        task_manager=TaskManager(),
        pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_vobiz_uses_v18_setup_and_filters_rtvi_messages():
    serializer = VobizFrameSerializer(
        stream_id="stream",
        params=VobizFrameSerializer.InputParams(auto_hang_up=False),
    )

    await serializer.setup(_setup())

    assert serializer._sample_rate == 16000
    assert (
        await serializer.serialize(
            OutputTransportMessageFrame(message={"label": "rtvi-ai", "type": "test"})
        )
        is None
    )


@pytest.mark.asyncio
async def test_asterisk_uses_v18_setup_contract():
    serializer = AsteriskFrameSerializer(
        channel_id="channel",
        ari_endpoint="http://asterisk.invalid",
        app_name="app",
        app_password="secret",
        params=AsteriskFrameSerializer.InputParams(sample_rate=24000),
    )

    await serializer.setup(_setup(sample_rate=16000))

    assert serializer._sample_rate == 24000

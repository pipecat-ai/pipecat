#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for shared telephony serializer lifecycle behavior."""

import asyncio
import json
from collections.abc import Callable
from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import CancelFrame, EndFrame, Frame, InterruptionFrame, TextFrame
from pipecat.serializers.base_telephony import AutoHangupFrameSerializer
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.serializers.twilio import TwilioFrameSerializer


class _TestAutoHangupFrameSerializer(AutoHangupFrameSerializer):
    """Minimal serializer used to exercise automatic hang-up behavior."""

    def __init__(self):
        super().__init__()
        self.hang_up = AsyncMock()

    async def serialize(self, frame: Frame) -> str | bytes | None:
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        return None

    async def _hang_up_call(self) -> None:
        await self.hang_up()


@pytest.mark.asyncio
async def test_auto_hangup_ignores_non_terminal_frame():
    serializer = _TestAutoHangupFrameSerializer()

    handled = await serializer._maybe_hang_up(TextFrame(text="hello"), enabled=True)

    assert handled is False
    serializer.hang_up.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("frame", [EndFrame(), CancelFrame()])
async def test_auto_hangup_ignores_terminal_frame_when_disabled(frame: Frame):
    serializer = _TestAutoHangupFrameSerializer()

    handled = await serializer._maybe_hang_up(frame, enabled=False)

    assert handled is False
    serializer.hang_up.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("frame", [EndFrame(), CancelFrame()])
async def test_auto_hangup_handles_terminal_frame(frame: Frame):
    serializer = _TestAutoHangupFrameSerializer()

    handled = await serializer._maybe_hang_up(frame, enabled=True)

    assert handled is True
    serializer.hang_up.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_auto_hangup_attempts_once_for_mixed_terminal_frames():
    serializer = _TestAutoHangupFrameSerializer()

    first_handled = await serializer._maybe_hang_up(EndFrame(), enabled=True)
    second_handled = await serializer._maybe_hang_up(CancelFrame(), enabled=True)

    assert first_handled is True
    assert second_handled is True
    serializer.hang_up.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_auto_hangup_marks_attempt_before_awaiting_provider_io():
    serializer = _TestAutoHangupFrameSerializer()
    started = asyncio.Event()
    finish = asyncio.Event()

    async def delayed_hang_up():
        started.set()
        await finish.wait()

    serializer.hang_up.side_effect = delayed_hang_up
    first = asyncio.create_task(serializer._maybe_hang_up(EndFrame(), enabled=True))
    await started.wait()

    second_handled = await serializer._maybe_hang_up(CancelFrame(), enabled=True)
    finish.set()
    first_handled = await first

    assert first_handled is True
    assert second_handled is True
    serializer.hang_up.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_auto_hangup_does_not_retry_after_provider_error():
    serializer = _TestAutoHangupFrameSerializer()
    serializer.hang_up.side_effect = RuntimeError("provider error")

    with pytest.raises(RuntimeError, match="provider error"):
        await serializer._maybe_hang_up(EndFrame(), enabled=True)

    handled = await serializer._maybe_hang_up(CancelFrame(), enabled=True)

    assert handled is True
    serializer.hang_up.assert_awaited_once_with()


def _twilio_serializer(*, auto_hang_up: bool = True) -> TwilioFrameSerializer:
    params = TwilioFrameSerializer.InputParams(auto_hang_up=auto_hang_up)
    return TwilioFrameSerializer(
        "stream-id",
        call_sid="call-id",
        account_sid="account-id",
        auth_token="auth-token",
        params=params,
    )


def _telnyx_serializer(*, auto_hang_up: bool = True) -> TelnyxFrameSerializer:
    params = TelnyxFrameSerializer.InputParams(auto_hang_up=auto_hang_up)
    return TelnyxFrameSerializer(
        "stream-id",
        "PCMU",
        "PCMU",
        call_control_id="call-control-id",
        api_key="api-key",
        params=params,
    )


def _plivo_serializer(*, auto_hang_up: bool = True) -> PlivoFrameSerializer:
    params = PlivoFrameSerializer.InputParams(auto_hang_up=auto_hang_up)
    return PlivoFrameSerializer(
        "stream-id",
        call_id="call-id",
        auth_id="auth-id",
        auth_token="auth-token",
        params=params,
    )


SerializerFactory = Callable[..., AutoHangupFrameSerializer]


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [_twilio_serializer, _telnyx_serializer, _plivo_serializer])
@pytest.mark.parametrize("frame", [EndFrame(), CancelFrame()])
async def test_provider_auto_hangup_handles_terminal_frame(
    factory: SerializerFactory, frame: Frame
):
    serializer = factory()
    serializer._hang_up_call = AsyncMock()

    result = await serializer.serialize(frame)

    assert result is None
    serializer._hang_up_call.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [_twilio_serializer, _telnyx_serializer, _plivo_serializer])
async def test_provider_auto_hangup_attempts_once(factory: SerializerFactory):
    serializer = factory()
    serializer._hang_up_call = AsyncMock()

    await serializer.serialize(EndFrame())
    await serializer.serialize(CancelFrame())

    serializer._hang_up_call.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [_twilio_serializer, _telnyx_serializer, _plivo_serializer])
async def test_provider_auto_hangup_can_be_disabled(factory: SerializerFactory):
    serializer = factory(auto_hang_up=False)
    serializer._hang_up_call = AsyncMock()

    result = await serializer.serialize(EndFrame())

    assert result is None
    serializer._hang_up_call.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (_twilio_serializer, {"event": "clear", "streamSid": "stream-id"}),
        (_telnyx_serializer, {"event": "clear"}),
        (_plivo_serializer, {"event": "clearAudio", "streamId": "stream-id"}),
    ],
)
async def test_provider_interruption_serialization_is_unchanged(
    factory: SerializerFactory, expected: dict[str, str]
):
    serializer = factory()

    result = await serializer.serialize(InterruptionFrame())

    assert isinstance(result, str)
    assert json.loads(result) == expected

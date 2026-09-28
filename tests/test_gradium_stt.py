#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import (
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.gradium.stt import GradiumSTTService, _TurnPhase
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies


def _service(*, enable_turn_detection: bool = True, **kwargs) -> GradiumSTTService:
    service = GradiumSTTService(
        api_key="test-key",
        sample_rate=16000,
        enable_turn_detection=enable_turn_detection,
        **kwargs,
    )
    service.broadcast_frame = AsyncMock()
    service.broadcast_interruption = AsyncMock()
    service.push_frame = AsyncMock()
    return service


def _step(inactivity: float, horizon: float = 3.0) -> dict:
    return {"type": "step", "vad": [{"horizon_s": horizon, "inactivity_prob": inactivity}]}


def test_gradium_recommends_external_strategies_only_with_turn_detection():
    assert isinstance(
        _service().service_metadata_frame().user_turn_strategies, ExternalUserTurnStrategies
    )
    assert (
        _service(enable_turn_detection=False).service_metadata_frame().user_turn_strategies is None
    )


def test_gradium_turn_detection_settings_are_set_only_with_turn_detection():
    on = _service()._settings
    off = _service(enable_turn_detection=False)._settings

    assert (on.eot_horizon_s, on.eot_threshold, on.post_flush_cooldown_frames) == (3.0, 0.5, 8)
    assert (off.eot_horizon_s, off.eot_threshold, off.post_flush_cooldown_frames) == (
        None,
        None,
        None,
    )


def test_gradium_turn_detection_settings_passed_in_win_over_the_defaults():
    settings = _service(settings=GradiumSTTService.Settings(eot_threshold=0.7))._settings

    assert settings.eot_threshold == 0.7
    assert settings.eot_horizon_s == 3.0


@pytest.mark.asyncio
async def test_gradium_update_reconnects_only_for_connection_bound_settings():
    for delta, reconnects in (
        (GradiumSTTService.Settings(eot_threshold=0.7), False),
        (GradiumSTTService.Settings(delay_in_frames=8), True),
    ):
        service = _service()
        service._websocket = object()
        service._disconnect = AsyncMock()
        service._connect = AsyncMock()

        await service._update_settings(delta)

        assert service._disconnect.await_count == (1 if reconnects else 0)


def test_gradium_supports_ttfs_only_without_turn_detection():
    assert not _service().supports_ttfs
    assert _service(enable_turn_detection=False).supports_ttfs


@pytest.mark.asyncio
async def test_gradium_a_vad_stop_flushes_only_without_turn_detection():
    for enable_turn_detection, flushes in ((False, 1), (True, 0)):
        service = _service(enable_turn_detection=enable_turn_detection)
        service._send_flush = AsyncMock()

        await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

        assert service._send_flush.await_count == flushes


@pytest.mark.asyncio
async def test_gradium_turn_detection_the_signal_falling_below_the_threshold_proposes_a_turn_start():
    service = _service()

    await service._handle_step(_step(0.9))
    await service._handle_step(_step(0.2))

    service.broadcast_frame.assert_awaited_once_with(ProposedUserStartedSpeakingFrame)
    # The strategies own the interruption; the service only proposes.
    service.broadcast_interruption.assert_not_awaited()
    assert service._turn_phase is _TurnPhase.OPEN


@pytest.mark.asyncio
async def test_gradium_turn_detection_a_signal_that_starts_low_opens_nothing():
    # With nothing heard yet the estimate starts low and climbs; that is not
    # speech. Only a dip after the signal has read inactive is.
    service = _service()

    for inactivity in (0.1, 0.3, 0.45):
        await service._handle_step(_step(inactivity))

    service.broadcast_frame.assert_not_awaited()
    assert service._turn_phase is not _TurnPhase.OPEN


@pytest.mark.asyncio
async def test_gradium_turn_detection_watches_the_horizon_closest_to_the_setting():
    service = _service(settings=GradiumSTTService.Settings(eot_horizon_s=1.0))

    await service._handle_step(
        {
            "type": "step",
            "vad": [
                {"horizon_s": 1.0, "inactivity_prob": 0.9},
                {"horizon_s": 3.0, "inactivity_prob": 0.1},
            ],
        }
    )

    assert service._turn_phase is not _TurnPhase.OPEN


@pytest.mark.asyncio
async def test_gradium_turn_detection_a_step_at_the_threshold_ends_an_open_turn_with_a_flush():
    service = _service()
    service._send_flush = AsyncMock()
    await service._handle_step(_step(0.9))
    await service._handle_step(_step(0.2))

    await service._handle_step(_step(0.5))

    service._send_flush.assert_awaited_once()
    assert service.broadcast_frame.await_args.args == (ProposedUserStoppedSpeakingFrame,)
    assert service._turn_phase is _TurnPhase.ENDING


@pytest.mark.asyncio
async def test_gradium_turn_detection_the_flush_ack_ends_the_ending_phase_and_starts_the_cooldown():
    service = _service()
    service._turn_phase = _TurnPhase.ENDING
    service._handle_flushed = AsyncMock()

    async def messages():
        yield json.dumps({"type": "flushed"})

    service._get_websocket = messages

    await service._receive_messages()

    assert service._turn_phase is _TurnPhase.IDLE
    assert service._flush_cooldown == 8
    service._handle_flushed.assert_awaited_once()


@pytest.mark.asyncio
async def test_gradium_turn_detection_the_transcript_finalizes_on_the_spot_when_the_flush_cannot_be_sent():
    # No socket, so no "flushed" acknowledgment is coming to finalize it.
    service = _service()
    service.emit_stt_usage_metrics = AsyncMock()
    service._trace_transcription = AsyncMock()
    service._turn_phase = _TurnPhase.OPEN
    service._accumulated_text = ["so far"]

    await service._end_turn()

    assert service._turn_phase is _TurnPhase.IDLE
    assert service.push_frame.await_args.args[0].text == "so far"
    service.broadcast_frame.assert_awaited_once_with(ProposedUserStoppedSpeakingFrame)


@pytest.mark.asyncio
async def test_gradium_turn_detection_ignores_the_signal_during_the_post_flush_cooldown():
    # The steps right after a flush can read as speech resuming; none of them
    # may open a turn, and the first one past the cooldown may.
    service = _service(settings=GradiumSTTService.Settings(post_flush_cooldown_frames=2))
    service._flush_cooldown = 2

    # Neither an inactive reading nor the dip after it counts while cooling down.
    await service._handle_step(_step(0.9))
    await service._handle_step(_step(0.9))
    await service._handle_step(_step(0.2))
    service.broadcast_frame.assert_not_awaited()

    await service._handle_step(_step(0.9))
    await service._handle_step(_step(0.2))
    service.broadcast_frame.assert_awaited_once_with(ProposedUserStartedSpeakingFrame)


@pytest.mark.asyncio
async def test_gradium_turn_detection_no_new_start_while_the_flush_is_unacknowledged():
    service = _service()
    service._turn_phase = _TurnPhase.ENDING

    await service._handle_step(_step(0.2))

    service.broadcast_frame.assert_not_awaited()
    assert service._turn_phase is not _TurnPhase.OPEN

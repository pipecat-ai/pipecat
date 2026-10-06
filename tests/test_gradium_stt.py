#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipecat.frames.frames import (
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.gradium import stt as gradium_stt
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


def _record_order(service: GradiumSTTService) -> list[str]:
    order = []

    async def push(frame, *args, **kwargs):
        order.append(type(frame).__name__)

    async def broadcast(frame_cls, **kwargs):
        order.append(frame_cls.__name__)

    service.push_frame = AsyncMock(side_effect=push)
    service.broadcast_frame = AsyncMock(side_effect=broadcast)
    return order


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
    # The stop is proposed once the flush's transcript is out.
    service.broadcast_frame.assert_awaited_once_with(ProposedUserStartedSpeakingFrame)
    assert service._turn_phase is _TurnPhase.ENDING


@pytest.mark.asyncio
async def test_gradium_turn_detection_the_stop_is_proposed_after_the_transcript(monkeypatch):
    # With the transcript already pushed, the turn strategies close the turn
    # on the stop proposal instead of waiting out their hold.
    monkeypatch.setattr(gradium_stt, "TRANSCRIPT_AGGREGATION_DELAY", 0)
    service = _service(settings=GradiumSTTService.Settings(post_flush_cooldown_frames=0))
    service.emit_stt_usage_metrics = AsyncMock()
    service._trace_transcription = AsyncMock()
    order = _record_order(service)
    service._handle_flushed = AsyncMock()
    service._turn_phase = _TurnPhase.ENDING
    service._accumulated_text = ["book a table"]

    async def messages():
        yield json.dumps({"type": "flushed"})
        yield json.dumps(_step(0.9))

    service._get_websocket = messages
    await service._receive_messages()
    await service._transcript_aggregation_handler()

    assert order == ["TranscriptionFrame", "ProposedUserStoppedSpeakingFrame"]
    assert service._turn_phase is _TurnPhase.ARMED


@pytest.mark.asyncio
async def test_gradium_turn_detection_the_flush_ack_starts_the_cooldown_and_holds_the_ending_phase():
    # The turn's transcript is still aggregating, so no new turn may start yet.
    service = _service()
    service._turn_phase = _TurnPhase.ENDING
    service._handle_flushed = AsyncMock()

    async def messages():
        yield json.dumps({"type": "flushed"})

    service._get_websocket = messages

    await service._receive_messages()

    assert service._turn_phase is _TurnPhase.ENDING
    assert service._flush_cooldown == 8
    service._handle_flushed.assert_awaited_once()


@pytest.mark.asyncio
async def test_gradium_turn_detection_speech_continuing_through_a_flush_keeps_the_turn_open(
    monkeypatch,
):
    # After a flush the signal follows the audio: a user who keeps talking
    # reads low, so the turn reopens instead of closing and the continuation
    # lands in the same user turn.
    monkeypatch.setattr(gradium_stt, "TRANSCRIPT_AGGREGATION_DELAY", 0)
    service = _service(settings=GradiumSTTService.Settings(post_flush_cooldown_frames=2))
    service.emit_stt_usage_metrics = AsyncMock()
    service._trace_transcription = AsyncMock()
    service._send_flush = AsyncMock(return_value=True)
    service._handle_flushed = AsyncMock()
    service._turn_phase = _TurnPhase.ENDING
    service._accumulated_text = ["I'd like to book a"]

    async def messages():
        yield json.dumps({"type": "flushed"})
        for _ in range(3):
            yield json.dumps(_step(0.1))

    service._get_websocket = messages
    await service._receive_messages()
    await service._transcript_aggregation_handler()

    assert service.push_frame.await_args.args[0].text == "I'd like to book a"
    service.broadcast_frame.assert_not_awaited()
    assert service._turn_phase is _TurnPhase.OPEN

    # The reopened turn ends on the signal like any other.
    await service._handle_step(_step(0.9))

    service._send_flush.assert_awaited_once()
    assert service._turn_phase is _TurnPhase.ENDING


@pytest.mark.asyncio
async def test_gradium_turn_detection_an_empty_turn_still_proposes_its_stop(monkeypatch):
    monkeypatch.setattr(gradium_stt, "TRANSCRIPT_AGGREGATION_DELAY", 0)
    service = _service()
    service._turn_phase = _TurnPhase.ENDING
    service._steps_since_flush = 9
    service._last_inactive = True  # a quiet reading after the post-flush cooldown

    await service._transcript_aggregation_handler()

    service.push_frame.assert_not_awaited()
    service.broadcast_frame.assert_awaited_once_with(ProposedUserStoppedSpeakingFrame)
    assert service._turn_phase is _TurnPhase.ARMED


@pytest.mark.asyncio
async def test_gradium_turn_detection_no_start_before_the_previous_transcript_is_pushed(
    monkeypatch,
):
    # With no cooldown, steps after the flush ack are read at once; the turn's
    # transcript and stop must still go out before the next start proposal.
    monkeypatch.setattr(gradium_stt, "TRANSCRIPT_AGGREGATION_DELAY", 0)
    service = _service(settings=GradiumSTTService.Settings(post_flush_cooldown_frames=0))
    service.emit_stt_usage_metrics = AsyncMock()
    service._trace_transcription = AsyncMock()
    service._handle_flushed = AsyncMock()
    service._turn_phase = _TurnPhase.ENDING
    service._accumulated_text = ["first turn"]

    async def messages():
        yield json.dumps({"type": "flushed"})
        yield json.dumps(_step(0.9))

    service._get_websocket = messages
    await service._receive_messages()

    service.broadcast_frame.assert_not_awaited()

    await service._transcript_aggregation_handler()
    await service._handle_step(_step(0.1))

    assert service.push_frame.await_args.args[0].text == "first turn"
    assert [c.args[0] for c in service.broadcast_frame.await_args_list] == [
        ProposedUserStoppedSpeakingFrame,
        ProposedUserStartedSpeakingFrame,
    ]


@pytest.mark.asyncio
async def test_gradium_without_turn_detection_a_flush_ack_leaves_turns_alone(monkeypatch):
    monkeypatch.setattr(gradium_stt, "TRANSCRIPT_AGGREGATION_DELAY", 0)
    service = _service(enable_turn_detection=False)
    service.emit_stt_usage_metrics = AsyncMock()
    service._trace_transcription = AsyncMock()
    service._handle_flushed = service._transcript_aggregation_handler
    service._accumulated_text = ["hello"]

    async def messages():
        yield json.dumps({"type": "flushed"})
        yield json.dumps(_step(0.9))
        yield json.dumps(_step(0.1))

    service._get_websocket = messages
    await service._receive_messages()

    assert service.push_frame.await_args.args[0].text == "hello"
    service.broadcast_frame.assert_not_awaited()
    assert service._turn_phase is _TurnPhase.IDLE
    assert service._flush_cooldown == 0


@pytest.mark.asyncio
async def test_gradium_turn_detection_a_failed_transcript_push_still_arms_the_next_turn(
    monkeypatch,
):
    monkeypatch.setattr(gradium_stt, "TRANSCRIPT_AGGREGATION_DELAY", 0)
    service = _service()
    service._turn_phase = _TurnPhase.ENDING
    service._steps_since_flush = 9
    service._last_inactive = True  # a quiet reading after the post-flush cooldown
    service._finalize_accumulated_text = AsyncMock(side_effect=RuntimeError("push failed"))

    with pytest.raises(RuntimeError):
        await service._transcript_aggregation_handler()

    assert service._turn_phase is _TurnPhase.ARMED


def _without_task_manager(service: GradiumSTTService):
    """Let SETTLING start its timeout without a pipeline's task manager."""

    def create_task(coro, name=None):
        coro.close()
        return None

    service.create_task = create_task
    service.cancel_task = AsyncMock()


@pytest.mark.asyncio
async def test_gradium_turn_detection_settling_waits_for_a_reading_after_the_flush(monkeypatch):
    monkeypatch.setattr(gradium_stt, "TRANSCRIPT_AGGREGATION_DELAY", 0)
    service = _service(settings=GradiumSTTService.Settings(post_flush_cooldown_frames=0))
    _without_task_manager(service)
    service._turn_phase = _TurnPhase.ENDING

    await service._transcript_aggregation_handler()

    assert service._turn_phase is _TurnPhase.SETTLING
    service.broadcast_frame.assert_not_awaited()

    await service._handle_step(_step(0.9))

    service.broadcast_frame.assert_awaited_once_with(ProposedUserStoppedSpeakingFrame)
    assert service._turn_phase is _TurnPhase.ARMED


@pytest.mark.asyncio
async def test_gradium_turn_detection_settling_proposes_the_stop_when_no_reading_comes(
    monkeypatch,
):
    monkeypatch.setattr(gradium_stt, "STOP_SETTLE_TIMEOUT_S", 0)
    service = _service()
    service._turn_phase = _TurnPhase.SETTLING

    await service._settle_timeout_handler()

    service.broadcast_frame.assert_awaited_once_with(ProposedUserStoppedSpeakingFrame)
    assert service._turn_phase is _TurnPhase.ARMED


@pytest.mark.asyncio
async def test_gradium_turn_detection_settling_decides_after_the_post_flush_cooldown(
    monkeypatch,
):
    # Readings inside the cooldown do not decide: a user who starts again a
    # moment after the flush still lands in the same turn.
    monkeypatch.setattr(gradium_stt, "TRANSCRIPT_AGGREGATION_DELAY", 0)
    service = _service(settings=GradiumSTTService.Settings(post_flush_cooldown_frames=2))
    _without_task_manager(service)
    service._turn_phase = _TurnPhase.ENDING

    await service._transcript_aggregation_handler()
    await service._handle_step(_step(0.1))
    await service._handle_step(_step(0.1))

    assert service._turn_phase is _TurnPhase.SETTLING

    await service._handle_step(_step(0.9))

    service.broadcast_frame.assert_awaited_once_with(ProposedUserStoppedSpeakingFrame)
    assert service._turn_phase is _TurnPhase.ARMED


@pytest.mark.asyncio
async def test_gradium_turn_detection_the_transcript_finalizes_on_the_spot_when_the_flush_cannot_be_sent():
    # No socket, so no "flushed" acknowledgment is coming to finalize it.
    service = _service()
    service.emit_stt_usage_metrics = AsyncMock()
    service._trace_transcription = AsyncMock()
    order = _record_order(service)
    service._turn_phase = _TurnPhase.OPEN
    service._accumulated_text = ["so far"]

    await service._end_turn()

    assert service._turn_phase is _TurnPhase.IDLE
    assert service.push_frame.await_args.args[0].text == "so far"
    assert order == ["TranscriptionFrame", "ProposedUserStoppedSpeakingFrame"]


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


def _reconnectable(service: GradiumSTTService) -> GradiumSTTService:
    service.emit_stt_usage_metrics = AsyncMock()
    service._trace_transcription = AsyncMock()
    service._disconnect_websocket = AsyncMock()
    service._connect_websocket = AsyncMock()
    service._verify_connection = AsyncMock(return_value=True)
    return service


@pytest.mark.asyncio
async def test_gradium_a_reconnect_pushes_the_text_so_far_and_keeps_an_open_turn_open():
    # The user may still be speaking; the new connection's signal ends the turn.
    service = _reconnectable(_service())
    service._turn_phase = _TurnPhase.OPEN
    service._accumulated_text = ["book a table"]
    service._flush_counter = 3
    service._flush_cooldown = 2

    assert await service._reconnect_websocket(1)

    frame = service.push_frame.await_args.args[0]
    assert isinstance(frame, TranscriptionFrame) and frame.text == "book a table"
    service.broadcast_frame.assert_not_awaited()
    assert service._turn_phase is _TurnPhase.OPEN
    assert (service._accumulated_text, service._flush_counter, service._flush_cooldown) == (
        [],
        0,
        0,
    )


@pytest.mark.asyncio
async def test_gradium_a_reconnect_while_ending_pushes_the_transcript_then_the_stop():
    service = _reconnectable(_service())
    order = _record_order(service)
    service._turn_phase = _TurnPhase.ENDING
    service._accumulated_text = ["the tail"]

    await service._reconnect_websocket(1)

    assert order == ["TranscriptionFrame", "ProposedUserStoppedSpeakingFrame"]
    assert service._turn_phase is _TurnPhase.IDLE


@pytest.mark.asyncio
async def test_gradium_a_reconnect_while_settling_proposes_the_stop():
    # The transcript is already out; only the stop was waiting on the signal.
    service = _reconnectable(_service())
    service._turn_phase = _TurnPhase.SETTLING

    await service._reconnect_websocket(1)

    service.push_frame.assert_not_awaited()
    service.broadcast_frame.assert_awaited_once_with(ProposedUserStoppedSpeakingFrame)
    assert service._turn_phase is _TurnPhase.IDLE


@pytest.mark.asyncio
async def test_gradium_a_settings_reconnect_while_settling_proposes_the_stop_first():
    service = _service()
    order = _record_order(service)
    service._websocket = object()

    async def disconnect():
        order.append("disconnect")

    service._disconnect = disconnect
    service._connect = AsyncMock()
    service._turn_phase = _TurnPhase.SETTLING

    await service._update_settings(GradiumSTTService.Settings(delay_in_frames=8))

    assert order == ["ProposedUserStoppedSpeakingFrame", "disconnect"]


@pytest.mark.asyncio
async def test_gradium_a_failed_first_connect_still_starts_the_receive_loop():
    # The receive loop is what retries the connection.
    service = _service()
    service._connect_websocket = AsyncMock()
    service.create_task = MagicMock(side_effect=lambda coro, *a, **kw: coro.close())

    await service._connect()

    assert service._websocket is None
    service.create_task.assert_called_once()


@pytest.mark.parametrize(
    "pipeline_rate, sent_rate",
    [(8000, 8000), (16000, 16000), (24000, 24000), (11025, 16000), (22050, 24000), (48000, 24000)],
)
def test_gradium_pcm_input_is_sent_at_a_rate_gradium_accepts(pipeline_rate, sent_rate):
    assert gradium_stt._gradium_pcm_sample_rate(pipeline_rate) == sent_rate


@pytest.mark.asyncio
async def test_gradium_audio_at_an_unsupported_rate_is_resampled_before_chunking():
    service = _service(enable_turn_detection=False)
    service._sample_rate = 48000
    service._send_sample_rate = 24000
    service._chunk_size_bytes = int(80 * 24000 * 2 / 1000)
    websocket = MagicMock()
    websocket.state = gradium_stt.State.OPEN
    websocket.send = AsyncMock()
    service._websocket = websocket

    # One second at 48 kHz is one second at 24 kHz: about 12 chunks of 80 ms.
    async for _ in service.run_stt(b"\x00\x00" * 48000):
        pass

    assert 11 <= websocket.send.await_count <= 12

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Image request completion when the Daily room is left."""

import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from pipecat.frames.frames import CancelFrame, EndFrame, UserImageRequestFrame
from pipecat.transports.base_transport import VideoInSourceParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.utils.asyncio.task_manager import TaskManager


async def _joined_transport():
    with (
        patch("pipecat.transports.daily.transport.Daily"),
        patch("pipecat.transports.daily.transport.CallClient"),
    ):
        transport = DailyTransport(
            "https://mock.daily.co/mock",
            None,
            "bot",
            params=DailyParams(
                video_in_enabled=True,
                audio_in_enabled=False,
                audio_out_enabled=False,
                video_in_sources={"camera": VideoInSourceParams(on_request_only=True)},
            ),
        )
    input_transport = transport.input()
    input_transport.push_frame = AsyncMock()
    input_transport.push_error = AsyncMock()
    transport._call_event_handler = AsyncMock(wraps=transport._call_event_handler)
    client = transport._client
    client._task_manager = MagicMock()
    client._task_manager.get_event_loop.return_value = asyncio.get_running_loop()
    client._client.join.side_effect = lambda *args, completion, **kwargs: completion({}, None)
    client._client.leave.side_effect = lambda *, completion: completion(None)
    client._client.update_subscriptions.side_effect = lambda *, completion, **kwargs: completion(
        None
    )
    # SDK I/O and startup/downstream notifications are isolated. The client
    # join/leave, subscription futures, callback queue and input methods are real.
    client._callbacks.on_joined = AsyncMock()
    await client.join()
    client._client.participants.return_value = {
        participant: {
            "id": participant,
            "media": {"camera": {"state": "playable"}, "screenVideo": {"state": "playable"}},
        }
        for participant in ("p1", "p2")
    }
    for participant, source in (("p1", "camera"), ("p1", "screenVideo"), ("p2", "camera")):
        await input_transport.capture_participant_video(
            participant, video_source=source, on_request_only=True
        )
    return transport


async def _requests(transport):
    results = []
    for participant, source in (
        ("p1", "camera"),
        ("p1", "camera"),
        ("p1", "screenVideo"),
        ("p2", "camera"),
    ):
        result = AsyncMock()
        await transport.input().request_participant_image(
            UserImageRequestFrame(user_id=participant, video_source=source, result_callback=result)
        )
        result.assert_not_awaited()
        results.append(result)
    return results


def _assert_completed(transport, results):
    assert not transport.input()._video_samplers._samplers
    for result in results:
        result.assert_awaited_once_with({"error": "The bot left the room."})


async def _start_audio_input(transport):
    input_transport = transport.input()
    reading = asyncio.Event()

    async def pending_audio_read():
        reading.set()
        await asyncio.Event().wait()

    input_transport._params.audio_in_enabled = True
    input_transport._params.audio_in_user_tracks = False
    input_transport._task_manager = TaskManager()
    transport._client.read_next_audio_frame = pending_audio_read
    await input_transport._start_audio_in_streaming()
    await asyncio.wait_for(reading.wait(), timeout=1)
    assert input_transport._audio_in_task is not None
    return input_transport._audio_in_task


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_frame", [EndFrame, CancelFrame])
async def test_local_leave_completes_requests_while_sdk_left_is_gated(terminal_frame):
    transport = await _joined_transport()
    input_transport = transport.input()
    client = transport._client
    results = await _requests(transport)
    subscription_started = asyncio.Event()
    subscription_done = asyncio.Event()
    completions = []

    def pending_subscription(*, completion, **kwargs):
        completions.append(completion)
        subscription_started.set()

    client._client.update_subscriptions.side_effect = pending_subscription
    client._client.set_video_renderer.side_effect = lambda *args, **kwargs: subscription_done.set()
    client._event_queue = asyncio.Queue()
    consumer = asyncio.create_task(client._callback_task_handler(client._event_queue))
    try:
        await asyncio.to_thread(client.on_participant_joined, {"id": "joining-peer"})
        await asyncio.wait_for(subscription_started.wait(), timeout=1)
        if terminal_frame is EndFrame:
            await input_transport.stop(terminal_frame())
        else:
            await input_transport.cancel(terminal_frame())
        await asyncio.to_thread(client.on_call_state_updated, "left")
        completions[0](None)
        await asyncio.wait_for(subscription_done.wait(), timeout=1)

        assert not client._joined_event.is_set()
        assert client._event_queue.qsize() == 1
        assert not consumer.done()
        client._client.leave.assert_called_once()
        _assert_completed(transport, results)
        transport._call_event_handler.assert_any_await("on_left")
    finally:
        consumer.cancel()
        with suppress(asyncio.CancelledError):
            await consumer


@pytest.mark.asyncio
async def test_only_last_room_owner_completes_pending_requests():
    transport = await _joined_transport()
    client = transport._client
    await client.join()
    results = await _requests(transport)

    await transport.input().stop(EndFrame())

    client._client.leave.assert_not_called()
    assert client._joined_event.is_set()
    for result in results:
        result.assert_not_awaited()

    await client.leave()
    _assert_completed(transport, results)
    await client.leave()
    await client._handle_call_state_updated("left")
    await client._handle_call_state_updated("left")
    client._client.leave.assert_called_once()
    _assert_completed(transport, results)
    assert transport._call_event_handler.await_args_list.count(call("on_left")) == 1


@pytest.mark.asyncio
async def test_passive_sdk_leave_still_completes_requests():
    transport = await _joined_transport()
    client = transport._client
    results = await _requests(transport)
    client._event_queue = asyncio.Queue()
    consumer = asyncio.create_task(client._callback_task_handler(client._event_queue))
    try:
        for _ in range(2):
            await asyncio.to_thread(client.on_call_state_updated, "left")
            await asyncio.wait_for(client._event_queue.join(), timeout=1)
            _assert_completed(transport, results)
        client._client.leave.assert_not_called()
        assert call("on_left") not in transport._call_event_handler.await_args_list
    finally:
        consumer.cancel()
        with suppress(asyncio.CancelledError):
            await consumer


@pytest.mark.asyncio
async def test_failed_sdk_leave_preserves_pending_requests_and_error_notification():
    transport = await _joined_transport()
    client = transport._client
    results = await _requests(transport)
    client._client.leave.side_effect = lambda *, completion: completion("leave failed")

    await transport.input().stop(EndFrame())

    for result in results:
        result.assert_not_awaited()
    assert transport.input()._video_samplers.is_capturing("p1", "camera")
    transport.input().push_error.assert_awaited_once()
    assert "leave failed" in transport.input().push_error.await_args.kwargs["error_msg"]
    assert call("on_left") not in transport._call_event_handler.await_args_list


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_count", [1, 2])
async def test_result_callback_errors_do_not_skip_other_requests_or_left_notification(
    failure_count,
):
    transport = await _joined_transport()
    results = await _requests(transport)
    errors = [RuntimeError("first callback"), ValueError("second callback")][:failure_count]
    for result, error in zip(results, errors, strict=False):
        result.side_effect = error

    with pytest.raises(RuntimeError if failure_count == 1 else ExceptionGroup) as raised:
        await transport.input().stop(EndFrame())

    _assert_completed(transport, results)
    if failure_count == 1:
        assert raised.value is errors[0]
    else:
        assert isinstance(raised.value, ExceptionGroup)
        assert raised.value.exceptions == tuple(errors)
    transport._call_event_handler.assert_any_await("on_left")
    await transport._client._handle_call_state_updated("left")
    _assert_completed(transport, results)


@pytest.mark.asyncio
async def test_concurrent_sdk_left_does_not_repeat_in_progress_result_callbacks():
    transport = await _joined_transport()
    results = await _requests(transport)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def pending_result(result):
        entered.set()
        await release.wait()

    results[0].side_effect = pending_result
    stopping = asyncio.create_task(transport.input().stop(EndFrame()))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        await transport._client._handle_call_state_updated("left")
        release.set()
        await asyncio.wait_for(stopping, timeout=1)
        _assert_completed(transport, results)
    finally:
        stopping.cancel()
        with suppress(asyncio.CancelledError):
            await stopping


@pytest.mark.asyncio
async def test_cancelling_request_completion_propagates_and_still_notifies_left():
    transport = await _joined_transport()
    results = await _requests(transport)
    audio_task = await _start_audio_input(transport)
    entered = asyncio.Event()

    async def pending_result(result):
        entered.set()
        await asyncio.Event().wait()

    results[0].side_effect = pending_result
    stopping = asyncio.create_task(transport.input().stop(EndFrame()))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopping
        results[0].assert_awaited_once()
        for result in results[1:]:
            result.assert_not_awaited()
        transport._call_event_handler.assert_any_await("on_left")
        assert transport.input()._audio_in_task is None
        assert audio_task.cancelled()
    finally:
        stopping.cancel()
        with suppress(asyncio.CancelledError):
            await stopping
        await transport.input()._stop_audio_in_task()


@pytest.mark.asyncio
async def test_output_only_left_notification_needs_no_input_transport():
    transport = await _joined_transport()
    transport._input = None

    await transport._client.leave()

    transport._call_event_handler.assert_any_await("on_left")


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_frame", [EndFrame, CancelFrame])
async def test_result_callback_failure_still_stops_running_audio_input(terminal_frame):
    transport = await _joined_transport()
    input_transport = transport.input()
    results = await _requests(transport)
    failure = RuntimeError("result callback failed")
    results[0].side_effect = failure
    audio_task = await _start_audio_input(transport)
    try:
        with pytest.raises(RuntimeError) as raised:
            if terminal_frame is EndFrame:
                await input_transport.stop(terminal_frame())
            else:
                await input_transport.cancel(terminal_frame())
        assert raised.value is failure
        assert input_transport._audio_in_task is None
        assert audio_task.cancelled()
        _assert_completed(transport, results)
    finally:
        await input_transport._stop_audio_in_task()

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Hermetic audio-send regressions against the actual AssemblyAI service."""

import asyncio
import json
import socket
from unittest.mock import AsyncMock

import pytest
from websockets.protocol import State

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.assemblyai.stt import AssemblyAISTTService
from pipecat.tests.utils import run_test
from pipecat.utils.asyncio.task_manager import TaskManager

CHUNK = b"\x01\x02" * 800  # 50 ms of mono PCM16 at 16 kHz.
SECOND = b"\x03\x04" * 800
THIRD = b"\x05\x06" * 800


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("This regression must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket.socket, "connect_ex", reject)


class FakeWebsocket:
    state = State.OPEN

    def __init__(self, service, *, reject_call=None, block_call=None):
        self.service = service
        self.reject_call = reject_call
        self.block_call = block_call
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.attempts = []
        self.accepted = []

    async def send(self, message):
        if isinstance(message, bytes):
            self.attempts.append(message)
            call = len(self.attempts)
            if call == self.block_call:
                self.entered.set()
                await self.release.wait()
            if call == self.reject_call:
                raise ConnectionError("rejected before accepting any bytes")
        self.accepted.append(message)
        if message == json.dumps({"type": "Terminate"}):
            self.service._termination_event.set()

    async def close(self):
        self.state = State.CLOSED


async def make_service(**socket_options):
    service = AssemblyAISTTService(api_key="test-key", sample_rate=16000)
    service._connect = AsyncMock()
    await service.setup(
        FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=TaskManager(),
            pipeline_worker=None,
            audio_in_sample_rate=16000,
        )
    )
    assert service._chunk_size_bytes == len(CHUNK)
    websocket = FakeWebsocket(service, **socket_options)
    service._websocket = websocket
    return service, websocket


async def send_audio(service, audio):
    assert [frame async for frame in service.run_stt(audio)] == [None]


@pytest.mark.asyncio
@pytest.mark.parametrize("public_processing", [False, True])
async def test_rejected_chunk_is_sent_once_after_later_success(public_processing):
    service, websocket = await make_service(reject_call=1)
    if public_processing:
        await service.process_audio_frame(
            InputAudioRawFrame(CHUNK, 16000, 1), FrameDirection.DOWNSTREAM
        )
    else:
        await send_audio(service, CHUNK)

    assert websocket.accepted == []
    assert bytes(service._audio_buffer) == CHUNK
    websocket.reject_call = None
    await send_audio(service, b"")
    assert websocket.accepted == [CHUNK]
    assert websocket.attempts == [CHUNK, CHUNK]
    assert not service._audio_buffer
    await send_audio(service, b"")
    assert websocket.accepted == [CHUNK]


@pytest.mark.asyncio
async def test_partial_success_preserves_failed_chunk_and_tail_in_order():
    service, websocket = await make_service(reject_call=2)
    tail = THIRD[:100]
    await send_audio(service, CHUNK + SECOND + tail)
    assert websocket.accepted == [CHUNK]
    assert bytes(service._audio_buffer) == SECOND + tail

    websocket.reject_call = None
    await send_audio(service, THIRD[100:])
    assert websocket.accepted == [CHUNK, SECOND, THIRD]
    assert websocket.attempts == [CHUNK, SECOND, SECOND, THIRD]
    assert not service._audio_buffer


@pytest.mark.asyncio
async def test_cancelled_send_preserves_chunk_and_propagates_cancellation():
    service, websocket = await make_service(block_call=1)
    sending = asyncio.create_task(send_audio(service, CHUNK + SECOND))
    await asyncio.wait_for(websocket.entered.wait(), timeout=1)
    sending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sending
    assert websocket.accepted == []
    assert bytes(service._audio_buffer) == CHUNK + SECOND

    websocket.block_call = None
    await send_audio(service, b"")
    assert websocket.accepted == [CHUNK, SECOND]
    assert not service._audio_buffer


@pytest.mark.asyncio
async def test_success_does_not_delete_replacement_audio_after_buffer_clear():
    service, websocket = await make_service(block_call=1)
    sending = asyncio.create_task(send_audio(service, CHUNK + SECOND))
    await asyncio.wait_for(websocket.entered.wait(), timeout=1)
    # Adversarial private mutation: a successful send must not delete new bytes.
    service._audio_buffer.clear()
    service._audio_buffer.extend(THIRD)
    websocket.release.set()
    await asyncio.wait_for(sending, timeout=1)
    assert websocket.accepted == [CHUNK, THIRD]
    assert not service._audio_buffer


@pytest.mark.asyncio
@pytest.mark.parametrize("reject", [False, True])
async def test_concurrent_disconnect_waits_for_inflight_send_before_flushing(reject):
    service, websocket = await make_service(block_call=1, reject_call=1 if reject else None)
    sending = asyncio.create_task(send_audio(service, CHUNK + SECOND))
    await asyncio.wait_for(websocket.entered.wait(), timeout=1)
    started = asyncio.Event()
    original_disconnect = service._disconnect

    async def disconnect():
        started.set()
        await original_disconnect()

    disconnecting = asyncio.create_task(disconnect())
    await started.wait()
    try:
        # _disconnect has run to its first blocking await.
        assert websocket.attempts == [CHUNK]
        assert not disconnecting.done()
    finally:
        websocket.release.set()
        await asyncio.wait_for(asyncio.gather(sending, disconnecting), timeout=1)
    expected = [CHUNK + SECOND] if reject else [CHUNK, SECOND]
    assert websocket.accepted == expected + [json.dumps({"type": "Terminate"})]
    assert not service._audio_buffer
    assert service._websocket is None


@pytest.mark.asyncio
async def test_cancel_then_disconnect_flushes_restored_audio_once():
    service, websocket = await make_service(block_call=1)
    sending = asyncio.create_task(send_audio(service, CHUNK))
    await asyncio.wait_for(websocket.entered.wait(), timeout=1)
    sending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await sending
    websocket.block_call = None
    await asyncio.wait_for(service._disconnect(), timeout=1)
    assert websocket.accepted == [CHUNK, json.dumps({"type": "Terminate"})]
    assert not service._audio_buffer


@pytest.mark.asyncio
async def test_concurrent_audio_calls_do_not_overtake_failed_chunk():
    service, websocket = await make_service(block_call=1, reject_call=1)
    first = asyncio.create_task(send_audio(service, CHUNK))
    await asyncio.wait_for(websocket.entered.wait(), timeout=1)
    started = asyncio.Event()

    async def second_send():
        started.set()
        await send_audio(service, SECOND)

    second = asyncio.create_task(second_send())
    await started.wait()
    try:
        assert websocket.attempts == [CHUNK]
    finally:
        websocket.release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
    assert websocket.accepted == [CHUNK, SECOND]
    assert not service._audio_buffer


@pytest.mark.asyncio
async def test_disconnect_flush_does_not_clear_new_audio_after_await():
    service, websocket = await make_service(block_call=1)
    service._audio_buffer.extend(CHUNK)
    disconnecting = asyncio.create_task(service._disconnect())
    await asyncio.wait_for(websocket.entered.wait(), timeout=1)
    service._audio_buffer.clear()
    service._audio_buffer.extend(SECOND)
    websocket.release.set()
    await asyncio.wait_for(disconnecting, timeout=1)
    assert websocket.accepted == [CHUNK, json.dumps({"type": "Terminate"})]
    assert bytes(service._audio_buffer) == SECOND


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_disconnect_flush_preserves_audio_and_releases_lock(cancel):
    service, websocket = await make_service(
        block_call=1 if cancel else None, reject_call=None if cancel else 1
    )
    service.push_error = AsyncMock()
    service._audio_buffer.extend(CHUNK + SECOND)
    if cancel:
        disconnecting = asyncio.create_task(service._disconnect())
        await asyncio.wait_for(websocket.entered.wait(), timeout=1)
        disconnecting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await disconnecting
    else:
        await asyncio.wait_for(service._disconnect(), timeout=1)
        service.push_error.assert_awaited_once()
    assert websocket.accepted == []
    assert bytes(service._audio_buffer) == CHUNK + SECOND

    recovered_socket = FakeWebsocket(service)
    service._websocket = recovered_socket
    await asyncio.wait_for(service._disconnect(), timeout=1)
    assert recovered_socket.accepted == [CHUNK + SECOND, json.dumps({"type": "Terminate"})]
    assert not service._audio_buffer


@pytest.mark.asyncio
async def test_inflight_send_does_not_drain_tail_into_replaced_socket():
    service, websocket = await make_service(block_call=1)
    sending = asyncio.create_task(send_audio(service, CHUNK + SECOND))
    await asyncio.wait_for(websocket.entered.wait(), timeout=1)
    new_socket = FakeWebsocket(service)
    service._websocket = new_socket
    websocket.release.set()
    await asyncio.wait_for(sending, timeout=1)
    assert websocket.accepted == [CHUNK]
    assert new_socket.accepted == []
    assert bytes(service._audio_buffer) == SECOND
    await send_audio(service, b"")
    assert new_socket.accepted == [SECOND]


class ReceivingWebsocket(FakeWebsocket):
    """Deliver the termination acknowledgment through the actual receive loop."""

    def __init__(self, service, **kwargs):
        super().__init__(service, **kwargs)
        self.incoming = asyncio.Queue()

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.incoming.get()

    async def ping(self):
        pass

    async def send(self, message):
        await super().send(message)
        if message == json.dumps({"type": "Terminate"}):
            self.service._termination_event.clear()
            await self.incoming.put(
                json.dumps(
                    {
                        "type": "Termination",
                        "audio_duration_seconds": 0.1,
                        "session_duration_seconds": 1.0,
                    }
                )
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("recover_with_next_frame", [False, True])
async def test_pipeline_recovers_audio_and_receives_termination(recover_with_next_frame):
    service = AssemblyAISTTService(api_key="test-key", sample_rate=16000)
    websocket = ReceivingWebsocket(service, reject_call=2 if recover_with_next_frame else 1)
    # Only the connection boundary is substituted: setup/connect and receive run normally.
    service._websocket_connect = AsyncMock(return_value=websocket)
    tail = THIRD[:100]
    frames = [
        InputAudioRawFrame(
            CHUNK + SECOND + tail if recover_with_next_frame else CHUNK + tail, 16000, 1
        )
    ]
    if recover_with_next_frame:
        frames.append(InputAudioRawFrame(THIRD[100:], 16000, 1))
    await asyncio.wait_for(
        run_test(
            service, frames_to_send=frames, expected_down_frames=None, expected_up_frames=None
        ),
        timeout=3,
    )
    expected = [CHUNK, SECOND, THIRD] if recover_with_next_frame else [CHUNK + tail]
    assert websocket.accepted == expected + [json.dumps({"type": "Terminate"})]
    assert service._termination_event.is_set()
    assert not service._audio_buffer
    assert service._websocket is None
    assert service._receive_task is None
    assert not service.task_manager.current_tasks()
    service._websocket_connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_repeated_public_audio_failures_preserve_all_bytes_in_order():
    service, websocket = await make_service()
    for call, audio in enumerate((CHUNK, SECOND, THIRD), start=1):
        websocket.reject_call = call
        await service.process_audio_frame(
            InputAudioRawFrame(audio, 16000, 1), FrameDirection.DOWNSTREAM
        )
    assert websocket.attempts == [CHUNK, CHUNK, CHUNK]
    assert websocket.accepted == []
    assert bytes(service._audio_buffer) == CHUNK + SECOND + THIRD
    websocket.reject_call = None
    await service._disconnect()
    assert websocket.accepted == [CHUNK + SECOND + THIRD, json.dumps({"type": "Terminate"})]
    assert not service._audio_buffer


@pytest.mark.asyncio
async def test_cancelling_waiting_caller_does_not_corrupt_inflight_audio():
    service, websocket = await make_service(block_call=1)
    first = asyncio.create_task(send_audio(service, CHUNK))
    await asyncio.wait_for(websocket.entered.wait(), timeout=1)
    entered = asyncio.Event()

    async def queued():
        entered.set()
        await send_audio(service, SECOND)

    second = asyncio.create_task(queued())
    await entered.wait()
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    websocket.release.set()
    await asyncio.wait_for(first, timeout=1)
    await send_audio(service, THIRD)
    assert websocket.accepted == [CHUNK, THIRD]
    assert not service._audio_buffer


@pytest.mark.asyncio
async def test_repeated_disconnect_does_not_send_audio_twice():
    service, websocket = await make_service()
    await send_audio(service, CHUNK[:100])
    await service._disconnect()
    await service._disconnect()
    assert websocket.accepted == [CHUNK[:100], json.dumps({"type": "Terminate"})]
    assert not service._audio_buffer


@pytest.mark.asyncio
async def test_actual_reconnect_path_retains_rejected_audio():
    service, websocket = await make_service(reject_call=1)
    await send_audio(service, CHUNK + SECOND)
    websocket.state = State.CLOSED
    recovered = ReceivingWebsocket(service)
    service._websocket_connect = AsyncMock(return_value=recovered)
    assert await service._reconnect_websocket(1) is True
    await service.process_audio_frame(
        InputAudioRawFrame(THIRD, 16000, 1), FrameDirection.DOWNSTREAM
    )
    assert recovered.accepted == [CHUNK, SECOND, THIRD]
    assert not service._audio_buffer
    # No receive task was started by make_service; use the immediate-ack fake for teardown.
    service._websocket = FakeWebsocket(service)
    await service._disconnect()


@pytest.mark.asyncio
async def test_after_acceptance_failure_documents_duplicate_delivery_policy():
    service, websocket = await make_service()
    original_send = websocket.send

    async def accept_then_raise(message):
        await original_send(message)
        if len(websocket.attempts) == 1:
            raise ConnectionError("simulated failure after accepting bytes")

    websocket.send = accept_then_raise
    await send_audio(service, CHUNK)
    assert websocket.accepted == [CHUNK]
    assert bytes(service._audio_buffer) == CHUNK
    await send_audio(service, b"")
    assert websocket.accepted == [CHUNK, CHUNK]
    assert not service._audio_buffer

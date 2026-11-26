#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Playback stops must preserve interruption reporting for later audio."""

import asyncio

import pytest

pytest.importorskip("rimelabs_sdk")

from pipecat.frames.frames import BotStoppedSpeakingFrame, ErrorFrame, InterruptionFrame
from tests.rime_realtime_peer import conversation, until
from tests.test_rime_realtime_service import sent


def playing(service):
    return [output.ref.response.response_id for output in service._playing]


def gate_writes(output, count):
    """Hold each of the first ``count`` audio writes until its event is set."""
    writing = [asyncio.Event() for _ in range(count)]
    release = [asyncio.Event() for _ in range(count)]
    write_audio = output.write_audio_frame
    writes = 0

    async def gated_write(frame):
        nonlocal writes
        index, writes = writes, writes + 1
        if index < count:
            writing[index].set()
            await release[index].wait()
        return await write_audio(frame)

    output.write_audio_frame = gated_write
    return writing, release


def observe_stops(service):
    stops = []
    process_frame = service.process_frame

    async def observe(frame, direction):
        await process_frame(frame, direction)
        if isinstance(frame, BotStoppedSpeakingFrame):
            stops.append(frame)

    service.process_frame = observe
    return stops


@pytest.mark.asyncio
@pytest.mark.parametrize("generation_ended", [False, True])
async def test_stop_of_an_earlier_output_keeps_the_next_one_truncatable(generation_ended):
    async with conversation() as (peer, service, worker, _, output, upstream, __):
        writing, release = gate_writes(output, 2)
        stops = observe_stops(service)
        try:
            first = await peer.start()
            await peer.message(first, text="First output", samples=480)
            await peer.end(first)
            await asyncio.wait_for(writing[0].wait(), 3)

            second = await peer.start()
            await peer.message(second, text="Second output", samples=2400, done=generation_ended)
            if generation_ended:
                await peer.end(second)
            await until(lambda: playing(service) == [first, second])
            release[0].set()
            await asyncio.wait_for(writing[1].wait(), 3)
            await until(lambda: len(stops) == 1)
            assert playing(service) == [second]

            await worker.queue_frame(InterruptionFrame())
            await until(lambda: bool(sent(peer, "conversation.item.truncate")))
            report = sent(peer, "conversation.item.truncate")[-1]
            assert report["item_id"] == f"message-{second}"
            assert 0 <= report["audio_end_ms"] <= 100
            await until(lambda: not service._requests)
            assert service.is_usable
            assert not any(isinstance(frame, ErrorFrame) for frame in upstream)
        finally:
            for event in release:
                event.set()


@pytest.mark.asyncio
async def test_queued_output_playback_is_timed_from_the_earlier_stop():
    async with conversation() as (peer, service, worker, _, output, upstream, __):
        writing, release = gate_writes(output, 2)
        stops = observe_stops(service)
        try:
            first = await peer.start()
            await peer.message(first, text="Filler", samples=480)
            await peer.end(first)
            await asyncio.wait_for(writing[0].wait(), 3)

            # One second of audio, generated while the filler still plays.
            second = await peer.start()
            await peer.message(second, text="Answer", samples=24000)
            await peer.end(second)
            await until(lambda: playing(service) == [first, second])
            await asyncio.sleep(0.4)
            release[0].set()
            await asyncio.wait_for(writing[1].wait(), 3)
            await until(lambda: len(stops) == 1)

            await worker.queue_frame(InterruptionFrame())
            await until(lambda: bool(sent(peer, "conversation.item.truncate")))
            report = sent(peer, "conversation.item.truncate")[-1]
            assert report["item_id"] == f"message-{second}"
            assert report["audio_end_ms"] < 300
            assert not any(isinstance(frame, ErrorFrame) for frame in upstream)
        finally:
            for event in release:
                event.set()


@pytest.mark.asyncio
async def test_interruption_during_an_earlier_output_truncates_it():
    async with conversation() as (peer, service, worker, _, output, upstream, __):
        writing, release = gate_writes(output, 1)
        try:
            first = await peer.start()
            await peer.message(first, text="Filler", samples=2400)
            await peer.end(first)
            await asyncio.wait_for(writing[0].wait(), 3)
            second = await peer.start()
            await peer.message(second, text="Answer", samples=2400)
            await peer.end(second)
            await until(lambda: playing(service) == [first, second])

            await worker.queue_frame(InterruptionFrame())
            await until(lambda: bool(sent(peer, "conversation.item.truncate")))
            report = sent(peer, "conversation.item.truncate")[-1]
            assert report["item_id"] == f"message-{first}"
            assert not service._playing
            assert not any(isinstance(frame, ErrorFrame) for frame in upstream)
        finally:
            release[0].set()


@pytest.mark.asyncio
async def test_completed_output_is_not_truncated_by_a_later_interruption():
    async with conversation() as (peer, service, worker, _, output, upstream, __):
        response = await peer.start()
        await peer.message(response)
        await peer.end(response)
        await until(lambda: bool(output.writes))
        await until(lambda: not service._playing)
        interrupted = asyncio.Event()
        interrupt = service._interrupt

        async def observe_interruption(output):
            await interrupt(output)
            interrupted.set()

        service._interrupt = observe_interruption
        await worker.queue_frame(InterruptionFrame())
        await asyncio.wait_for(interrupted.wait(), 3)
        assert not sent(peer, "conversation.item.truncate")
        assert not sent(peer, "response.cancel")
        assert not any(isinstance(frame, ErrorFrame) for frame in upstream)


@pytest.mark.asyncio
async def test_playback_stop_waits_for_the_transport_write():
    async with conversation() as (peer, service, _, __, output, ___, ____):
        writing = asyncio.Event()
        release_write = asyncio.Event()
        acknowledged = asyncio.Event()
        write_audio = output.write_audio_frame
        process_frame = service.process_frame

        async def gated_write(frame):
            writing.set()
            await release_write.wait()
            return await write_audio(frame)

        async def observe_acknowledgment(frame, direction):
            await process_frame(frame, direction)
            if isinstance(frame, BotStoppedSpeakingFrame):
                acknowledged.set()

        output.write_audio_frame = gated_write
        service.process_frame = observe_acknowledgment
        try:
            response = await peer.start()
            await peer.message(response)
            await peer.end(response)
            await asyncio.wait_for(writing.wait(), 3)
            await until(lambda: service._responses[response].ended)
            assert not acknowledged.is_set()
            assert playing(service) == [response]

            release_write.set()
            await asyncio.wait_for(acknowledged.wait(), 3)
            assert not service._playing
        finally:
            release_write.set()

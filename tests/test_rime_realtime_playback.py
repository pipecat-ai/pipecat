#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Output completion must preserve interruption reporting for later audio."""

import asyncio

import pytest

pytest.importorskip("rimelabs_sdk")

from pipecat.frames.frames import BotStoppedSpeakingFrame, ErrorFrame, InterruptionFrame
from pipecat.services.rime.realtime._aggregators import _RimeOutputStoppedFrame
from tests.rime_realtime_peer import conversation, until
from tests.test_rime_realtime_service import sent


@pytest.mark.asyncio
@pytest.mark.parametrize("generation_ended", [False, True])
async def test_earlier_playback_stop_preserves_later_interruption(generation_ended):
    async with conversation() as (peer, service, worker, _, output, upstream, __):
        first_write = asyncio.Event()
        release_first = asyncio.Event()
        second_write = asyncio.Event()
        release_second = asyncio.Event()
        stopped = asyncio.Event()
        output_stopped = asyncio.Event()
        write_audio = output.write_audio_frame
        process_frame = service.process_frame
        writes = 0

        async def gated_write(frame):
            nonlocal writes
            writes += 1
            if writes == 1:
                first_write.set()
                await release_first.wait()
            elif writes == 2:
                second_write.set()
                await release_second.wait()
            return await write_audio(frame)

        async def observe_stop(frame, direction):
            await process_frame(frame, direction)
            if isinstance(frame, BotStoppedSpeakingFrame):
                stopped.set()
            elif isinstance(frame, _RimeOutputStoppedFrame):
                output_stopped.set()

        output.write_audio_frame = gated_write
        service.process_frame = observe_stop
        try:
            first = await peer.start()
            await peer.message(first, text="First output", samples=480)
            await peer.end(first)
            await asyncio.wait_for(first_write.wait(), 3)

            second = await peer.start()
            await peer.message(second, text="Second output", samples=2400, done=generation_ended)
            if generation_ended:
                await peer.end(second)
            await until(
                lambda: (
                    service._latest_audio_output is not None
                    and service._latest_audio_output.ref.response.response_id == second
                )
            )
            release_first.set()
            await asyncio.wait_for(second_write.wait(), 3)
            await asyncio.wait_for(stopped.wait(), 3)
            await asyncio.wait_for(output_stopped.wait(), 3)

            await worker.queue_frame(InterruptionFrame())
            await until(lambda: bool(sent(peer, "conversation.item.truncate")))
            report = sent(peer, "conversation.item.truncate")[-1]
            assert report["item_id"] == f"message-{second}"
            assert 0 <= report["audio_end_ms"] <= 100
            if not generation_ended:
                assert sent(peer, "response.cancel")[-1]["response_id"] == second
            await until(lambda: not service._requests)
            assert service.is_usable
            assert not any(isinstance(frame, ErrorFrame) for frame in upstream)
        finally:
            release_first.set()
            release_second.set()


@pytest.mark.asyncio
async def test_completed_output_is_not_truncated_by_a_later_interruption():
    async with conversation() as (peer, service, worker, _, output, upstream, __):
        response = await peer.start()
        await peer.message(response)
        await peer.end(response)
        await until(lambda: bool(output.writes))
        await until(lambda: service._latest_audio_output is None)
        interrupted = asyncio.Event()
        interrupt = service._interrupt

        async def observe_interruption():
            await interrupt()
            interrupted.set()

        service._interrupt = observe_interruption
        await worker.queue_frame(InterruptionFrame())
        await asyncio.wait_for(interrupted.wait(), 3)
        assert not sent(peer, "conversation.item.truncate")
        assert not sent(peer, "response.cancel")
        assert not any(isinstance(frame, ErrorFrame) for frame in upstream)

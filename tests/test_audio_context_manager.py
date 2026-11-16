#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for AudioContextManager."""

import asyncio

import pytest

from pipecat.frames.frames import TextFrame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.utils.context.audio_context_manager import AudioContextFrame, AudioContextManager


def _audio(num_bytes: int) -> TTSAudioRawFrame:
    return TTSAudioRawFrame(b"\x01" * num_bytes, 16000, 1)


@pytest.mark.asyncio
async def test_contexts_and_frames_play_in_queue_order():
    """Contexts and frames come out in the order they were queued, then playback ends."""
    manager = AudioContextManager()
    frame = TextFrame("between")
    await manager.create("a")
    await manager.put_frame(frame)
    await manager.create("b")
    await manager.end_playback()

    first = await manager.next_for_playback()
    assert isinstance(first, AudioContextFrame) and first.context_id == "a"
    assert manager.playing_context_id == "a"
    assert await manager.next_for_playback() is frame
    second = await manager.next_for_playback()
    assert isinstance(second, AudioContextFrame) and second.context_id == "b"
    assert manager.playing_context_id == "b"
    assert await manager.next_for_playback() is None


@pytest.mark.asyncio
async def test_context_items_come_out_in_order_until_the_context_ends():
    """A context's items come out in order, and None ends the context."""
    manager = AudioContextManager()
    await manager.create("a")
    stopped = TTSStoppedFrame(context_id="a")
    await manager.append("a", _audio(4))
    await manager.append("a", stopped)
    await manager.append("a", None)

    assert len((await manager.get("a", timeout=1)).audio) == 4
    assert await manager.get("a", timeout=1) is stopped
    assert await manager.get("a", timeout=1) is None


@pytest.mark.asyncio
async def test_split_sample_is_carried_to_the_next_frame():
    """Audio is trimmed to whole samples, and the split sample joins the next frame."""
    manager = AudioContextManager()
    await manager.create("a")
    await manager.append("a", _audio(3))
    await manager.append("a", _audio(3))

    assert len((await manager.get("a", timeout=1)).audio) == 2
    assert len((await manager.get("a", timeout=1)).audio) == 4


@pytest.mark.asyncio
async def test_split_sample_is_padded_when_the_context_ends():
    """A split sample left when the context ends is padded and played before the end."""
    manager = AudioContextManager()
    await manager.create("a")
    await manager.append("a", _audio(3))
    await manager.append("a", None)

    assert len((await manager.get("a", timeout=1)).audio) == 2
    assert (await manager.get("a", timeout=1)).audio == b"\x01\x00"
    assert await manager.get("a", timeout=1) is None


@pytest.mark.asyncio
async def test_frame_with_less_than_a_sample_is_dropped():
    """A frame without a whole sample is held back, not played."""
    manager = AudioContextManager()
    await manager.create("a")
    await manager.append("a", _audio(1))

    with pytest.raises(TimeoutError):
        await manager.get("a", timeout=0.05)


@pytest.mark.asyncio
async def test_keepalive_restarts_the_wait():
    """A keepalive restarts the wait for the context's next item."""
    manager = AudioContextManager()
    await manager.create("a")

    async def keep_then_append():
        await asyncio.sleep(0.06)
        manager.keepalive("a")
        await asyncio.sleep(0.06)
        await manager.append("a", None)

    task = asyncio.create_task(keep_then_append())
    assert await manager.get("a", timeout=0.1) is None
    await task


@pytest.mark.asyncio
async def test_reset_keeps_only_uninterruptible_contexts_and_frames():
    """Reset drops the interruptible contexts and frames waiting to play."""
    manager = AudioContextManager()
    kept_frame = TextFrame("kept")
    kept_frame.interruptible = False
    await manager.create("a")
    await manager.put_frame(TextFrame("dropped"))
    await manager.put_frame(kept_frame)
    await manager.create("b", interruptible=False)
    manager.reset()
    await manager.end_playback()

    assert await manager.next_for_playback() is kept_frame
    context = await manager.next_for_playback()
    assert isinstance(context, AudioContextFrame) and context.context_id == "b"
    assert await manager.next_for_playback() is None


@pytest.mark.asyncio
async def test_removed_context_is_skipped():
    """A context removed before it plays is skipped in play order."""
    manager = AudioContextManager()
    await manager.create("a", interruptible=False)
    await manager.create("b")
    manager.reset()
    manager.remove("a")
    await manager.create("c")
    await manager.end_playback()

    context = await manager.next_for_playback()
    assert isinstance(context, AudioContextFrame) and context.context_id == "c"
    assert await manager.next_for_playback() is None


@pytest.mark.asyncio
async def test_remove_forgets_the_context_and_its_split_sample():
    """A removed context is no longer available, and its split sample is gone."""
    manager = AudioContextManager()
    await manager.create("a")
    await manager.append("a", _audio(3))
    manager.remove("a")

    assert not manager.available("a")
    assert manager.context_ids == []
    await manager.create("a")
    await manager.append("a", _audio(2))
    assert (await manager.get("a", timeout=1)).audio == b"\x01\x01"

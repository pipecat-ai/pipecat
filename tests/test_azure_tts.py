#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for AzureTTSService cross-thread audio delivery.

Azure's Speech SDK fires its synthesis callbacks from native (non-event-loop)
threads. Those callbacks must deliver to the awaiting ``run_tts`` getter even when
the event loop is otherwise idle — e.g. a headless pipeline with no output transport
pumping audio. A bare ``asyncio.Queue.put_nowait()`` from another thread does NOT
wake an idle selector, so the audio sits unread; the callbacks marshal onto the loop
with ``asyncio.run_coroutine_threadsafe(queue.put(...), self.get_event_loop())``.

These are deterministic regression tests: the getter is parked on an idle loop
*first*, then the callback is fired from a real thread. With the fix the getter wakes
in ~ms; without it the loop stays blocked in ``select()`` until the ``wait_for`` timer
(seconds), so the elapsed assertion fails (or ``wait_for`` raises ``TimeoutError``).
"""

import asyncio
import threading
import time
from unittest.mock import Mock

import pytest

pytest.importorskip("azure.cognitiveservices.speech")

from azure.cognitiveservices.speech import CancellationReason  # noqa: E402

from pipecat.services.azure.tts import AzureTTSService  # noqa: E402

# The thread fires after this delay, by which point the awaiting getter has parked
# the loop in select() — so only the callback itself can wake it.
_FIRE_DELAY = 0.1
# Generous wait so the buggy path blocks on the timer; tight bound so the fix (~ms
# after _FIRE_DELAY) passes while the bug (~_WAIT) fails.
_WAIT = 5.0
_MAX_DELIVERY = 1.0


def _make_service() -> AzureTTSService:
    svc = AzureTTSService(api_key="test-key", region="eastus")
    # The SDK callbacks call get_event_loop(); without a started pipeline there is no
    # task manager, so point it at the running test loop.
    loop = asyncio.get_running_loop()
    svc.get_event_loop = lambda: loop
    return svc


async def _assert_idle_loop_wakeup(get_coro, fire):
    """Park ``get_coro`` on an idle loop, then ``fire()`` the callback from a thread.

    Returns the value the getter received; asserts it arrived promptly (i.e. the
    cross-thread put woke the idle loop rather than waiting for the ``wait_for`` timer).
    """
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(get_coro)
    t0 = loop.time()
    threading.Thread(target=lambda: (time.sleep(_FIRE_DELAY), fire())).start()
    result = await asyncio.wait_for(task, timeout=_WAIT)
    assert loop.time() - t0 < _MAX_DELIVERY
    return result


@pytest.mark.asyncio
async def test_synthesizing_audio_wakes_idle_loop():
    """Audio pushed from an SDK thread reaches a parked getter on an idle loop."""
    svc = _make_service()

    audio = b"\x00\x01" * 256
    evt = Mock()
    evt.result.audio_data = audio

    data = await _assert_idle_loop_wakeup(
        svc._audio_queue.get(), lambda: svc._handle_synthesizing(evt)
    )
    assert data == audio


@pytest.mark.asyncio
async def test_canceled_error_wakes_idle_loop():
    """A non-user cancellation delivers its error to a parked getter on an idle loop."""
    svc = _make_service()

    evt = Mock()
    evt.result.cancellation_details.reason = CancellationReason.Error
    evt.result.cancellation_details.error_details = "boom"

    item = await _assert_idle_loop_wakeup(svc._audio_queue.get(), lambda: svc._handle_canceled(evt))
    assert isinstance(item, Exception)
    assert "boom" in str(item)


@pytest.mark.asyncio
async def test_completion_sentinel_wakes_idle_loop():
    """The completion sentinel reaches the word-boundary getter on an idle loop.

    ``_handle_completed`` routes completion through the word-boundary queue, whose
    getter is the (loop-side) word-processor task; that handoff is also a cross-thread
    put and must wake an idle loop.
    """
    svc = _make_service()

    evt = Mock()
    evt.result.audio_duration = None  # skip duration bookkeeping

    item = await _assert_idle_loop_wakeup(
        svc._word_boundary_queue.get(), lambda: svc._handle_completed(evt)
    )
    assert item is None

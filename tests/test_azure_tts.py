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

from pipecat.frames.frames import InterruptionFrame  # noqa: E402
from pipecat.processors.frame_processor import FrameDirection  # noqa: E402
from pipecat.services.azure.tts import AzureHttpTTSService, AzureTTSService  # noqa: E402

SSML_SERVICE_CLASSES = (AzureTTSService, AzureHttpTTSService)

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


@pytest.mark.parametrize("service_class", SSML_SERVICE_CLASSES)
def test_construct_ssml_default_has_no_lang_element(service_class):
    """Default settings emit no <lang> element anywhere in the output."""
    service = service_class(api_key="test-key", region="eastus")

    ssml = service._construct_ssml("Hello there.")

    assert "<lang" not in ssml
    assert ssml == (
        "<speak version='1.0' xml:lang='en-US' "
        "xmlns='http://www.w3.org/2001/10/synthesis' "
        "xmlns:mstts='http://www.w3.org/2001/mstts'>"
        "<voice name='en-US-SaraNeural'>"
        "<mstts:silence type='Sentenceboundary' value='20ms' />"
        "Hello there."
        "</voice></speak>"
    )


@pytest.mark.parametrize("service_class", SSML_SERVICE_CLASSES)
def test_construct_ssml_force_locale_false_matches_default(service_class):
    """force_locale=False explicitly must match the unset default byte-for-byte."""
    default_service = service_class(api_key="test-key", region="eastus")
    explicit_false_service = service_class(
        api_key="test-key",
        region="eastus",
        settings=service_class.Settings(force_locale=False),
    )

    assert default_service._construct_ssml(
        "Hello there."
    ) == explicit_false_service._construct_ssml("Hello there.")


@pytest.mark.parametrize("service_class", SSML_SERVICE_CLASSES)
def test_construct_ssml_force_locale_wraps_text_in_lang_element(service_class):
    """force_locale=True wraps the text in <lang xml:lang> inside <voice>."""
    service = service_class(
        api_key="test-key",
        region="eastus",
        settings=service_class.Settings(language="en-GB", force_locale=True),
    )

    ssml = service._construct_ssml("Hello there.")

    assert ssml == (
        "<speak version='1.0' xml:lang='en-GB' "
        "xmlns='http://www.w3.org/2001/10/synthesis' "
        "xmlns:mstts='http://www.w3.org/2001/mstts'>"
        "<voice name='en-US-SaraNeural'>"
        "<mstts:silence type='Sentenceboundary' value='20ms' />"
        "<lang xml:lang='en-GB'>"
        "Hello there."
        "</lang>"
        "</voice></speak>"
    )


@pytest.mark.parametrize("service_class", SSML_SERVICE_CLASSES)
def test_construct_ssml_force_locale_wraps_nested_style_and_prosody(service_class):
    """force_locale=True wraps the entire nested style/prosody/emphasis block,
    not just the raw text.
    """
    service = service_class(
        api_key="test-key",
        region="eastus",
        settings=service_class.Settings(
            language="en-GB",
            force_locale=True,
            style="cheerful",
            rate="slow",
            emphasis="strong",
        ),
    )

    ssml = service._construct_ssml("Hello there.")

    assert ssml == (
        "<speak version='1.0' xml:lang='en-GB' "
        "xmlns='http://www.w3.org/2001/10/synthesis' "
        "xmlns:mstts='http://www.w3.org/2001/mstts'>"
        "<voice name='en-US-SaraNeural'>"
        "<mstts:silence type='Sentenceboundary' value='20ms' />"
        "<lang xml:lang='en-GB'>"
        "<mstts:express-as style='cheerful'>"
        "<prosody rate='slow'>"
        "<emphasis level='strong'>"
        "Hello there."
        "</emphasis>"
        "</prosody>"
        "</mstts:express-as>"
        "</lang>"
        "</voice></speak>"
    )


def _make_streaming_service() -> AzureTTSService:
    """AzureTTSService wired for headless use, including _handle_interruption."""
    svc = _make_service()
    # No TaskManager without a started pipeline; skip recreating the audio task.
    svc._create_audio_context_task = lambda: None
    return svc


def _cancel_event(result_id, reason=CancellationReason.CancelledByUser):
    evt = Mock()
    evt.result.result_id = result_id
    evt.result.cancellation_details.reason = reason
    evt.result.cancellation_details.error_details = None
    return evt


@pytest.mark.asyncio
async def test_canceled_marker_of_interrupted_synthesis_is_dropped():
    """#6743: a late synthesis_canceled from an interrupted synthesis must not
    deliver its end-of-stream marker to the drained queue. The next run_tts
    would read it as its own end of stream and produce no audio.
    """
    svc = _make_streaming_service()
    svc._active_result_id = "rid-A"

    await svc._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM)
    svc._handle_canceled(_cancel_event("rid-A"))
    await asyncio.sleep(0.05)

    assert svc._audio_queue.empty()
    # The terminal event also cleans up the tracking entry.
    assert "rid-A" not in svc._cancelled_result_ids


@pytest.mark.asyncio
async def test_stale_user_cancel_without_synthesis_in_flight_is_dropped():
    """#6743: a user-cancel arriving when no synthesis is in flight (e.g. the
    interruption raced synthesis_started, so the result id was never tracked)
    must not queue an end-of-stream marker either.
    """
    svc = _make_streaming_service()

    svc._handle_canceled(_cancel_event("rid-unknown"))
    await asyncio.sleep(0.05)

    assert svc._audio_queue.empty()


@pytest.mark.asyncio
async def test_user_cancel_still_unblocks_awaiting_synthesis():
    """A user-cancel while run_tts awaits audio still delivers the marker."""
    svc = _make_streaming_service()
    svc._synthesis_in_flight = True

    svc._handle_canceled(_cancel_event("rid-A"))

    assert await asyncio.wait_for(svc._audio_queue.get(), timeout=1.0) is None


@pytest.mark.asyncio
async def test_error_cancellation_still_surfaces():
    """Non-user cancellations (errors) are still delivered as errors."""
    svc = _make_streaming_service()
    svc._synthesis_in_flight = True

    svc._handle_canceled(_cancel_event("rid-A", CancellationReason.Error))

    item = await asyncio.wait_for(svc._audio_queue.get(), timeout=1.0)
    assert isinstance(item, Exception)


@pytest.mark.asyncio
async def test_events_of_interrupted_synthesis_are_dropped():
    """#6743: audio chunks, word boundaries and completion events of an
    interrupted synthesis must not reach the queues after the interruption
    drained them.
    """
    svc = _make_streaming_service()
    svc._active_result_id = "rid-A"
    await svc._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM)

    audio_evt = Mock()
    audio_evt.result.result_id = "rid-A"
    audio_evt.result.audio_data = b"\x00\x01" * 64
    svc._handle_synthesizing(audio_evt)

    word_evt = Mock()
    word_evt.result_id = "rid-A"
    word_evt.text = "hello"
    word_evt.audio_offset = 0
    svc._handle_word_boundary(word_evt)

    done_evt = Mock()
    done_evt.result.result_id = "rid-A"
    svc._handle_completed(done_evt)
    await asyncio.sleep(0.05)

    assert svc._audio_queue.empty()
    assert svc._word_boundary_queue.empty()


@pytest.mark.asyncio
async def test_next_synthesis_audio_flows_after_interruption():
    """#6743 end-to-end shape: after the interrupted synthesis's stale marker is
    dropped, the next synthesis's audio still flows through the queue.
    """
    svc = _make_streaming_service()
    svc._active_result_id = "rid-A"
    await svc._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM)
    svc._handle_canceled(_cancel_event("rid-A"))
    await asyncio.sleep(0.05)

    audio = b"\x00\x01" * 64
    audio_evt = Mock()
    audio_evt.result.result_id = "rid-B"
    audio_evt.result.audio_data = audio
    svc._handle_synthesizing(audio_evt)

    assert await asyncio.wait_for(svc._audio_queue.get(), timeout=1.0) == audio

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Service-specific OpenTelemetry tracing decorators for Pipecat.

This module provides specialized decorators that automatically capture
rich information about service execution including configuration,
parameters, and performance metrics.
"""

import functools
import inspect
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

# Type imports for type checking only
if TYPE_CHECKING:
    from opentelemetry import context as context_api
    from opentelemetry import trace

from pipecat.frames.frames import (
    MetricsFrame,
    TranscriptionFrame,
    TTSStoppedFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.processors.aggregators.llm_context import NOT_GIVEN
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.tracing.service_attributes import (
    add_gemini_live_span_attributes,
    add_llm_span_attributes,
    add_openai_realtime_span_attributes,
    add_stt_span_attributes,
    add_tts_span_attributes,
)
from pipecat.utils.tracing.setup import is_tracing_available

if is_tracing_available():
    from opentelemetry import context as context_api
    from opentelemetry import trace

T = TypeVar("T")
R = TypeVar("R")


def _get_model_name(service) -> str:
    """Get the model name from a service instance.

    This is a bit of a mess — there were multiple places a model name could live.
    Soon, self._settings should be the only source of truth about model name.
    In fact...it might already be the case, but juuuuust to be safe, we'll
    check all the places we used to store it.
    """
    return (
        # Some services store an API-response-provided detailed "full" name,
        # which is distinct from the user-provided model name
        getattr(service, "_full_model_name", None)
        or getattr(getattr(service, "_settings", None), "model", None)
        or getattr(service, "model_name", None)
        or getattr(service, "_model_name", None)
        or "unknown"
    )


def _noop_decorator(func):
    """No-op fallback decorator when tracing is unavailable.

    Args:
        func: The function to pass through unchanged.

    Returns:
        The original function unchanged.
    """
    return func


def _get_turn_context(self):
    """Get the current turn's tracing context if available.

    Args:
        self: The service instance.

    Returns:
        The turn context, or None if unavailable.
    """
    tracing_ctx = getattr(self, "_tracing_context", None)
    return tracing_ctx.get_turn_context() if tracing_ctx else None


def _get_parent_service_context(self):
    """Get the parent service span context (internal use only).

    This looks for the service span that was created when the service was initialized,
    or falls back to the conversation context if available.

    Args:
        self: The service instance.

    Returns:
        The parent service context, or None if unavailable.
    """
    if not is_tracing_available():
        return None

    # Use the conversation context set by TurnTraceObserver via TracingContext.
    tracing_ctx = getattr(self, "_tracing_context", None)
    conversation_context = tracing_ctx.get_conversation_context() if tracing_ctx else None
    if conversation_context:
        return conversation_context

    # Last resort: use current context (may create orphan spans)
    return context_api.get_current()


def _add_token_usage_to_span(span, token_usage):
    """Add token usage metrics to a span (internal use only).

    Args:
        span: The span to add token metrics to.
        token_usage: Dictionary or object containing token usage information.
    """
    if not is_tracing_available() or not token_usage:
        return

    if isinstance(token_usage, dict):
        if "prompt_tokens" in token_usage:
            span.set_attribute("gen_ai.usage.input_tokens", token_usage["prompt_tokens"])
        if "completion_tokens" in token_usage:
            span.set_attribute("gen_ai.usage.output_tokens", token_usage["completion_tokens"])
        # Add cached token metrics for dictionary
        if (
            "cache_read_input_tokens" in token_usage
            and token_usage["cache_read_input_tokens"] is not None
        ):
            span.set_attribute(
                "gen_ai.usage.cache_read.input_tokens", token_usage["cache_read_input_tokens"]
            )
        if (
            "cache_creation_input_tokens" in token_usage
            and token_usage["cache_creation_input_tokens"] is not None
        ):
            span.set_attribute(
                "gen_ai.usage.cache_creation.input_tokens",
                token_usage["cache_creation_input_tokens"],
            )
        if "reasoning_tokens" in token_usage and token_usage["reasoning_tokens"] is not None:
            span.set_attribute("gen_ai.usage.reasoning_tokens", token_usage["reasoning_tokens"])
    else:
        # Handle LLMTokenUsage object
        span.set_attribute("gen_ai.usage.input_tokens", getattr(token_usage, "prompt_tokens", 0))
        span.set_attribute(
            "gen_ai.usage.output_tokens", getattr(token_usage, "completion_tokens", 0)
        )

        # Add cached token metrics for LLMTokenUsage object
        cache_read_tokens = getattr(token_usage, "cache_read_input_tokens", None)
        if cache_read_tokens is not None:
            span.set_attribute("gen_ai.usage.cache_read.input_tokens", cache_read_tokens)

        cache_creation_tokens = getattr(token_usage, "cache_creation_input_tokens", None)
        if cache_creation_tokens is not None:
            span.set_attribute("gen_ai.usage.cache_creation.input_tokens", cache_creation_tokens)

        reasoning_tokens = getattr(token_usage, "reasoning_tokens", None)
        if reasoning_tokens is not None:
            span.set_attribute("gen_ai.usage.reasoning_tokens", reasoning_tokens)


def traced_tts(func: Callable | None = None, *, name: str | None = None) -> Callable:
    """Trace TTS service methods with TTS-specific attributes.

    Automatically captures and records:

    - Service name and model information
    - Voice ID and settings
    - Character count and text content
    - Performance metrics like TTFB

    The span is scoped to the full synthesis operation, from
    ``create_audio_context`` until ``TTSStoppedFrame`` (or
    ``remove_audio_context`` as a safety net), so TTFB and any other
    runtime-computed metrics land on the correct span even when audio
    chunks are delivered after ``run_tts`` returns (e.g. WebSocket
    streaming TTS services).

    Works with both async functions and generators.

    Args:
        func: The TTS method to trace.
        name: Custom span name. Defaults to service type and class name.

    Returns:
        Wrapped method with TTS-specific tracing.
    """
    if not is_tracing_available():
        return _noop_decorator if func is None else _noop_decorator(func)

    def decorator(f):
        is_async_generator = inspect.isasyncgenfunction(f)

        def end_tts_span(service, context_id, *, interrupted=False):
            """End the TTS span for ``context_id`` if still open. Idempotent."""
            entry = service._tts_spans.pop(context_id, None)
            if not entry:
                return
            try:
                span = entry["span"]
                if interrupted:
                    span.set_attribute("tts.interrupted", True)
                span.end()
            except Exception as e:
                logging.warning(f"Error closing TTS span: {e}")

        def install_audio_context_patches(service):
            """Install per-instance wrappers on the audio-context methods.

            The wrappers own the lifetime of the TTS span:

            - ``create_audio_context``: opens the span and records
              baseline attributes.
            - ``append_to_audio_context``: ends the span on
              ``TTSStoppedFrame``.
            - ``push_frame``: records ``metrics.ttfb`` from the
              canonical ``TTFBMetricsData`` payload of any
              ``MetricsFrame`` pushed by ``stop_ttfb_metrics``. Reading
              the value from the metrics event (instead of polling
              ``_metrics.ttfb`` when the first audio is queued) avoids
              the ``ttfb`` property's in-progress fallback, which would
              otherwise report an under-estimate whenever a context's
              audio waits behind earlier queued audio before
              ``_handle_audio_context`` actually stops the TTFB
              measurement.
            - ``remove_audio_context``: ends any still-open span as a
              safety net for error and cancellation paths.
            - ``on_audio_context_completed``: ends the span on natural
              completion. Needed because services that rely on the
              base class to auto-push ``TTSStoppedFrame`` (via
              ``push_frame`` in ``_handle_audio_context``) bypass the
              ``append_to_audio_context`` hook entirely.
            - ``reset_active_audio_context``: ends the currently
              playing context's span if still open. Always called from
              ``_handle_interruption``, so this is the interruption
              hook.

            The patches check ``_tracing_enabled`` at invocation time,
            so they are safe to install regardless of whether tracing
            is enabled.
            """
            if getattr(service, "__tts_tracing_patches_installed__", False):
                return
            service.__tts_tracing_patches_installed__ = True
            service._tts_spans = {}

            orig_create = service.create_audio_context
            orig_append = service.append_to_audio_context
            orig_remove = service.remove_audio_context
            orig_completed = service.on_audio_context_completed
            orig_reset_active = service.reset_active_audio_context
            orig_push_frame = service.push_frame

            async def traced_create_audio_context(context_id):
                if getattr(service, "_tracing_enabled", False):
                    try:
                        parent = _get_turn_context(service) or _get_parent_service_context(service)
                        tracer = trace.get_tracer("pipecat")
                        span = tracer.start_span("tts", context=parent)
                        service._tts_spans[context_id] = {"span": span, "ttfb_recorded": False}

                        settings = getattr(service, "_settings", None)
                        add_tts_span_attributes(
                            span=span,
                            service_name=service.__class__.__name__,
                            model=_get_model_name(service),
                            voice_id=getattr(settings, "voice", "unknown"),
                            settings=settings,
                            operation_name="tts",
                        )
                    except Exception as e:
                        logging.warning(f"Error opening TTS span: {e}")
                return await orig_create(context_id)

            async def traced_append_to_audio_context(context_id, frame):
                entry = service._tts_spans.get(context_id)
                if entry and frame is not None:
                    try:
                        if isinstance(frame, TTSStoppedFrame):
                            entry["span"].end()
                            service._tts_spans.pop(context_id, None)
                    except Exception as e:
                        logging.warning(f"Error updating TTS span: {e}")
                return await orig_append(context_id, frame)

            async def traced_push_frame(frame, direction=FrameDirection.DOWNSTREAM):
                await orig_push_frame(frame, direction)
                if not getattr(service, "_tracing_enabled", False):
                    return
                if not isinstance(frame, MetricsFrame):
                    return
                try:
                    playing_id = getattr(service, "_playing_context_id", None)
                    if playing_id is None:
                        return
                    entry = service._tts_spans.get(playing_id)
                    if not entry or entry["ttfb_recorded"]:
                        return
                    for data in frame.data:
                        if isinstance(data, TTFBMetricsData):
                            entry["span"].set_attribute("metrics.ttfb", data.value)
                            entry["ttfb_recorded"] = True
                            break
                except Exception as e:
                    logging.warning(f"Error recording TTS ttfb from MetricsFrame: {e}")

            async def traced_remove_audio_context(context_id):
                entry = service._tts_spans.pop(context_id, None)
                if entry:
                    try:
                        entry["span"].end()
                    except Exception as e:
                        logging.warning(f"Error closing TTS span: {e}")
                return await orig_remove(context_id)

            async def traced_on_audio_context_completed(context_id):
                end_tts_span(service, context_id)
                return await orig_completed(context_id)

            def traced_reset_active_audio_context():
                playing_id = getattr(service, "_playing_context_id", None)
                if playing_id is not None:
                    end_tts_span(service, playing_id, interrupted=True)
                return orig_reset_active()

            service.create_audio_context = traced_create_audio_context
            service.append_to_audio_context = traced_append_to_audio_context
            service.push_frame = traced_push_frame
            service.remove_audio_context = traced_remove_audio_context
            service.on_audio_context_completed = traced_on_audio_context_completed
            service.reset_active_audio_context = traced_reset_active_audio_context

        def patch_setup(owner):
            """Wrap ``owner.setup`` so audio-context patches install per-instance.

            Idempotent: if a parent class has already been wrapped,
            skip. The patches check ``_tracing_enabled`` at invocation
            time, so wrapping is always safe.
            """
            original_setup = owner.setup
            if getattr(original_setup, "__tts_tracing_setup_wrapped__", False):
                return

            @functools.wraps(original_setup)
            async def patched_setup(self, setup):
                await original_setup(self, setup)
                install_audio_context_patches(self)

            setattr(patched_setup, "__tts_tracing_setup_wrapped__", True)
            owner.setup = patched_setup

        def attach_run_tts_attributes(service, text, args, kwargs):
            """Attach text-specific attributes to the in-flight TTS span."""
            if not getattr(service, "_tracing_enabled", False):
                return
            try:
                context_id = args[0] if args else kwargs.get("context_id")
                entry = getattr(service, "_tts_spans", {}).get(context_id)
                if entry and text:
                    span = entry["span"]
                    span.set_attribute("text", text)
                    span.set_attribute("metrics.character_count", len(text))
            except Exception as e:
                logging.warning(f"Error attaching TTS text to span: {e}")

        def make_run_tts_wrapper():
            """Build the wrapper around ``run_tts`` that adds per-call attributes.

            Span lifetime is owned by the audio-context patches. This
            wrapper only attaches the text and character count to the
            span that was opened by ``create_audio_context`` just
            before ``run_tts`` was invoked.
            """
            if is_async_generator:

                @functools.wraps(f)
                async def gen_wrapper(self, text, *args, **kwargs):
                    attach_run_tts_attributes(self, text, args, kwargs)
                    async for item in f(self, text, *args, **kwargs):
                        yield item

                return gen_wrapper

            @functools.wraps(f)
            async def coro_wrapper(self, text, *args, **kwargs):
                attach_run_tts_attributes(self, text, args, kwargs)
                return await f(self, text, *args, **kwargs)

            return coro_wrapper

        class _TracedTTSDescriptor:
            """Class-level descriptor that wires up TTS tracing at class definition time.

            ``__set_name__`` fires when the class body finishes evaluating,
            giving us a chance to wrap the owner's ``setup()`` so that the
            audio-context patches install on every instance before any
            ``create_audio_context`` call (including the very first one).
            """

            def __set_name__(self, owner, attr_name):
                patch_setup(owner)
                setattr(owner, attr_name, make_run_tts_wrapper())

        return _TracedTTSDescriptor()

    if func is not None:
        # ``decorator(func)`` returns a descriptor placeholder that
        # Python replaces with the real wrapped function once
        # ``__set_name__`` runs at class definition time. Pyright sees
        # only the descriptor instance, hence the ignore.
        return decorator(func)  # type: ignore[return-value]
    return decorator


def traced_stt(func: Callable | None = None, *, name: str | None = None) -> Callable:
    """Trace STT service methods with transcription attributes.

    Automatically captures and records:

    - Service name and model information
    - Transcription text and final status
    - Language information
    - Performance metrics like TTFB

    The span is scoped to one STT segment, from
    ``VADUserStartedSpeakingFrame`` (or the first ``TranscriptionFrame``
    when VAD did not fire, e.g. whispered speech) until a finalized
    ``TranscriptionFrame``. Multiple finalized transcripts in a single
    user turn produce multiple sequential spans, each anchored at the
    point speech for that segment began. ``metrics.ttfb`` is read after
    the base ``push_frame`` runs ``stop_ttfb_metrics`` for the
    finalized frame, so the value is correct for the closing span.

    Args:
        func: The STT method to trace.
        name: Custom span name. Defaults to function name.

    Returns:
        The original method unchanged. The decorator's class-definition-
        time work is to install a ``push_frame`` wrapper on the owning
        class that owns the span lifetime.
    """
    if not is_tracing_available():
        return _noop_decorator if func is None else _noop_decorator(func)

    def decorator(f):
        def patch_push_frame(owner):
            """Wrap ``owner.push_frame`` to drive the STT span lifecycle.

            Idempotent: if a parent class has already been wrapped, skip.
            The wrapper checks ``_tracing_enabled`` at invocation time,
            so it is safe to install regardless of whether tracing is
            enabled.
            """
            original_push_frame = owner.push_frame
            if getattr(original_push_frame, "__stt_tracing_push_frame_wrapped__", False):
                return

            def update_transcript(state, new_text):
                """Append or extend the current segment in ``state['segments']``.

                If ``new_text`` starts with the last recorded segment,
                treat it as a continuation (interim accumulation) and
                replace the last segment. Otherwise treat it as a new
                segment and append. Some STT services (Deepgram with
                utterance_end_ms enabled, for example) emit several
                ``TranscriptionFrame``s per turn where each carries a
                different segment rather than a cumulative update —
                without this logic the span's transcript would only
                show the last segment and the beginning would be lost.
                """
                if not new_text:
                    return
                segments = state["segments"]
                if not segments:
                    segments.append(new_text)
                elif new_text.startswith(segments[-1]):
                    segments[-1] = new_text
                else:
                    segments.append(new_text)

            def open_span(service, state):
                """Open the STT span, anchored at ``segment_start_time`` if set."""
                parent = _get_turn_context(service) or _get_parent_service_context(service)
                tracer = trace.get_tracer("pipecat")
                start_time_ns = (
                    int(state["segment_start_time"] * 1e9)
                    if state["segment_start_time"] is not None
                    else None
                )
                span = tracer.start_span("stt", context=parent, start_time=start_time_ns)
                try:
                    settings = getattr(service, "_settings", None)
                    add_stt_span_attributes(
                        span=span,
                        service_name=service.__class__.__name__,
                        model=_get_model_name(service),
                        settings=settings,
                        vad_enabled=getattr(service, "vad_enabled", False),
                    )
                except Exception as e:
                    logging.warning(f"Error setting STT span baseline attributes: {e}")
                state["span"] = span

            def handle_pre_push(service, frame, state):
                """Record speech-start anchor; lazy-open span on first transcript.

                Lazy-opening on ``TranscriptionFrame`` (rather than on
                ``VADUserStartedSpeakingFrame`` or
                ``UserStartedSpeakingFrame``) avoids racing with
                ``TurnTraceObserver._handle_turn_started``, which runs
                in a background task fired by ``_call_event_handler``
                (``base_object.py:232``) and may not have set the new
                turn's context yet — that produces STT spans parented
                to the previous turn. By the time STT actually emits
                a transcript, the turn observer has run.

                Opening happens in pre-push (rather than post-push) so
                that the recursive ``push_frame`` that
                ``STTService.push_frame`` triggers for the
                ``MetricsFrame`` (via ``stop_ttfb_metrics`` at
                ``stt_service.py:465``) sees the span already open and
                can attribute ``metrics.ttfb`` to it.
                """
                if isinstance(frame, VADUserStartedSpeakingFrame):
                    # Anchor the next span at the moment speech began.
                    # Skip if we already have an anchor (intra-turn VAD
                    # re-trigger) or a span open.
                    if state["span"] is None and state["segment_start_time"] is None:
                        state["segment_start_time"] = frame.timestamp - frame.start_secs
                elif isinstance(frame, TranscriptionFrame) and state["span"] is None:
                    open_span(service, state)

            async def handle_post_push(service, frame, state):
                """Attach per-frame attrs; close on finalized; record TTFB from MetricsFrame.

                ``metrics.ttfb`` is read off the ``TTFBMetricsData``
                payload of any ``MetricsFrame`` pushed by
                ``stop_ttfb_metrics`` — the canonical value the rest
                of the system uses — rather than from
                ``_metrics.ttfb``, which has an in-progress fallback
                branch (``frame_processor_metrics.py:48-62``) that
                would return an under-estimate if read at the wrong
                time.

                One STT span per finalized transcript: the span opens
                lazily on the first ``TranscriptionFrame`` (pre-push,
                anchored at speech start via ``segment_start_time``)
                and closes on ``finalized=True``. Multiple finalized
                transcripts in a single turn produce multiple spans.

                For services that never set ``frame.finalized=True``
                (e.g. Deepgram, which only marks it via
                ``confirm_finalize()``), the span closes on
                ``UserStoppedSpeakingFrame``. To capture
                ``metrics.ttfb`` for those spans we force-stop any
                pending TTFB measurement before closing — that pushes
                a ``MetricsFrame``, our post-push attributes the
                value, and ``patched_stop_ttfb_metrics`` closes the
                span. The ``stt.incomplete=true`` flag is only set if
                neither a finalized transcript nor a TTFB measurement
                ever finalized for the span.
                """
                if isinstance(frame, UserStoppedSpeakingFrame):
                    prev_span = state["span"]
                    if prev_span is None:
                        return
                    metrics = getattr(service, "_metrics", None)
                    if metrics is not None and getattr(metrics, "_start_ttfb_time", 0) > 0:
                        last_transcript_time = getattr(service, "_last_transcript_time", 0) or None
                        try:
                            await service.stop_ttfb_metrics(end_time=last_transcript_time)
                        except Exception as e:
                            logging.warning(f"Error force-stopping STT TTFB on user turn end: {e}")
                    # patched_stop_ttfb_metrics may have closed the span
                    # via the timeout path; re-check.
                    if state["span"] is None:
                        state["segments"] = []
                        return
                    state["span"].set_attribute("stt.incomplete", True)
                    state["span"].end()
                    state["span"] = None
                    state["segment_start_time"] = None
                    state["segments"] = []
                elif isinstance(frame, MetricsFrame):
                    span = state["span"]
                    if span is None:
                        return
                    for data in frame.data:
                        if isinstance(data, TTFBMetricsData):
                            span.set_attribute("metrics.ttfb", data.value)
                            break
                elif isinstance(frame, TranscriptionFrame):
                    span = state["span"]
                    if span is None:
                        return
                    if frame.text:
                        update_transcript(state, frame.text)
                        span.set_attribute("transcript", " ".join(state["segments"]).strip())
                    span.set_attribute("is_final", bool(frame.finalized))
                    if frame.language:
                        span.set_attribute("language", str(frame.language))
                    if frame.user_id:
                        span.set_attribute("user_id", frame.user_id)
                    if frame.finalized:
                        span.end()
                        state["span"] = None
                        state["segment_start_time"] = None
                        state["segments"] = []

            @functools.wraps(original_push_frame)
            async def patched_push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
                state = getattr(self, "_stt_span_state", None)
                if state is None:
                    state = {"span": None, "segment_start_time": None, "segments": []}
                    self._stt_span_state = state

                if getattr(self, "_tracing_enabled", False):
                    try:
                        handle_pre_push(self, frame, state)
                    except Exception as e:
                        logging.warning(f"Error in STT pre-push tracing: {e}")

                await original_push_frame(self, frame, direction)

                if getattr(self, "_tracing_enabled", False):
                    try:
                        await handle_post_push(self, frame, state)
                    except Exception as e:
                        logging.warning(f"Error in STT post-push tracing: {e}")

            setattr(patched_push_frame, "__stt_tracing_push_frame_wrapped__", True)
            owner.push_frame = patched_push_frame

        def patch_stop_ttfb_metrics(owner):
            """Wrap ``owner.stop_ttfb_metrics`` to close the span on the timeout path.

            When ``stop_ttfb_metrics`` is invoked with ``end_time`` set,
            that signals the TTFB-timeout handler firing
            (`stt_service.py:566`), or our own force-stop from the
            ``UserStoppedSpeakingFrame`` handler. In either case we
            anchor the span's end at ``end_time``
            (= ``_last_transcript_time``) rather than at whenever the
            coroutine resumed.

            ``metrics.ttfb`` attribution is not done here — the
            ``MetricsFrame`` that ``stop_ttfb_metrics`` pushes flows
            through ``push_frame`` and gets recorded by
            ``handle_post_push``, which reads the canonical
            ``TTFBMetricsData.value`` rather than the in-progress
            ``_metrics.ttfb`` property.
            """
            original_stop = owner.stop_ttfb_metrics
            if getattr(original_stop, "__stt_tracing_stop_ttfb_wrapped__", False):
                return

            @functools.wraps(original_stop)
            async def patched_stop(self, *, end_time=None):
                await original_stop(self, end_time=end_time)
                if end_time is None:
                    return
                if not getattr(self, "_tracing_enabled", False):
                    return
                state = getattr(self, "_stt_span_state", None)
                if not state or state["span"] is None:
                    return
                try:
                    span = state["span"]
                    span.end(end_time=int(end_time * 1e9))
                    state["span"] = None
                    state["segment_start_time"] = None
                    state["segments"] = []
                except Exception as e:
                    logging.warning(f"Error in STT stop_ttfb_metrics tracing: {e}")

            setattr(patched_stop, "__stt_tracing_stop_ttfb_wrapped__", True)
            owner.stop_ttfb_metrics = patched_stop

        class _TracedSTTDescriptor:
            """Class-level descriptor that wires up STT tracing at class definition time.

            ``__set_name__`` fires when the class body finishes evaluating,
            giving us a chance to wrap the owner's ``push_frame`` so that
            VAD, transcription, and finalization events drive the span
            lifecycle, and to wrap ``stop_ttfb_metrics`` so the
            TTFB-timeout path can attach metrics and close the span when
            no finalized transcript ever arrives. The decorated method
            itself runs unchanged.
            """

            def __set_name__(self, owner, attr_name):
                patch_push_frame(owner)
                patch_stop_ttfb_metrics(owner)
                setattr(owner, attr_name, f)

        return _TracedSTTDescriptor()

    if func is not None:
        # ``decorator(func)`` returns a descriptor placeholder that
        # Python replaces with the real wrapped function once
        # ``__set_name__`` runs at class definition time. Pyright sees
        # only the descriptor instance, hence the ignore.
        return decorator(func)  # type: ignore[return-value]
    return decorator


def traced_llm(func: Callable | None = None, *, name: str | None = None) -> Callable:
    """Trace LLM service methods with LLM-specific attributes.

    Automatically captures and records:

    - Service name and model information
    - Context content and messages
    - Tool configurations
    - Token usage metrics
    - Performance metrics like TTFB
    - Aggregated output text

    Args:
        func: The LLM method to trace.
        name: Custom span name. Defaults to service type and class name.

    Returns:
        Wrapped method with LLM-specific tracing.
    """
    if not is_tracing_available():
        return _noop_decorator if func is None else _noop_decorator(func)

    def decorator(f):
        @functools.wraps(f)
        async def wrapper(self, context, *args, **kwargs):
            if not getattr(self, "_tracing_enabled", False):
                return await f(self, context, *args, **kwargs)

            fn_called = False
            try:
                service_class_name = self.__class__.__name__
                span_name = "llm"

                # Get the parent context - turn context if available, otherwise service context
                parent_context = _get_turn_context(self) or _get_parent_service_context(self)

                # Create a new span as child of the turn span or service span
                tracer = trace.get_tracer("pipecat")
                with tracer.start_as_current_span(
                    span_name, context=parent_context
                ) as current_span:
                    try:
                        # Store original method and output aggregator
                        original_push_frame = self.push_frame
                        output_text = ""  # Simple string accumulation

                        async def traced_push_frame(frame, direction=None):
                            nonlocal output_text
                            # Capture text from LLMTextFrame during streaming
                            if (
                                hasattr(frame, "__class__")
                                and frame.__class__.__name__ == "LLMTextFrame"
                                and hasattr(frame, "text")
                            ):
                                output_text += frame.text

                            # Call original
                            if direction is not None:
                                return await original_push_frame(frame, direction)
                            else:
                                return await original_push_frame(frame)

                        # For token usage monitoring
                        original_start_llm_usage_metrics = None
                        if hasattr(self, "start_llm_usage_metrics"):
                            original_start_llm_usage_metrics = self.start_llm_usage_metrics

                            # Override the method to capture token usage
                            @functools.wraps(original_start_llm_usage_metrics)
                            async def wrapped_start_llm_usage_metrics(tokens):
                                # Call the original method
                                await original_start_llm_usage_metrics(tokens)

                                # Add token usage to the current span
                                _add_token_usage_to_span(current_span, tokens)

                            # Replace the method temporarily
                            self.start_llm_usage_metrics = wrapped_start_llm_usage_metrics

                        try:
                            # Replace push_frame to capture output
                            self.push_frame = traced_push_frame

                            # Get messages for logging
                            # Use adapter's get_messages_for_logging() which returns
                            # messages in provider's native format with sensitive data sanitized
                            messages = None
                            serialized_messages = None

                            # Use adapter for provider-native format
                            if hasattr(self, "get_llm_adapter"):
                                adapter = self.get_llm_adapter()
                                messages = adapter.get_messages_for_logging(context)

                            # Serialize messages if available
                            if messages:
                                serialized_messages = json.dumps(messages)

                            # Get tools
                            # Use adapter's from_standard_tools() to convert ToolsSchema
                            tools = None
                            serialized_tools = None
                            tool_count = 0

                            # Use adapter to convert ToolsSchema
                            if hasattr(self, "get_llm_adapter") and hasattr(context, "tools"):
                                adapter = self.get_llm_adapter()
                                tools = adapter.from_standard_tools(context.tools)

                            # Serialize and count tools if available
                            # Check if tools is not None and not NOT_GIVEN
                            if tools is not None and tools is not NOT_GIVEN:
                                serialized_tools = json.dumps(tools)
                                tool_count = len(tools) if isinstance(tools, list) else 1

                            # Handle system message for different services
                            # settings.system_instruction takes priority (matches service behavior)
                            system_message = None
                            if hasattr(self, "_settings") and getattr(
                                self._settings, "system_instruction", None
                            ):
                                system_message = self._settings.system_instruction
                            else:
                                # Fall back to extracting from context messages
                                ctx_messages = context.get_messages()
                                if ctx_messages:
                                    first = ctx_messages[0]
                                    if isinstance(first, dict) and first.get("role") == "system":
                                        content = first.get("content")
                                        if isinstance(content, str):
                                            system_message = content
                                        elif isinstance(content, list):
                                            system_message = " ".join(
                                                part.get("text", "")
                                                for part in content
                                                if isinstance(part, dict)
                                                and part.get("type") == "text"
                                            )

                            # Use given_fields() defensively in case a service doesn't
                            # initialize all settings.
                            params = {}
                            if hasattr(self, "_settings"):
                                for key, value in self._settings.given_fields().items():
                                    # system_instruction is already captured as the
                                    # "system_instructions" span attribute above.
                                    if key == "system_instruction":
                                        continue
                                    if isinstance(value, (int, float, bool, str)):
                                        params[key] = value
                                    elif value is None:
                                        params[key] = "NOT_GIVEN"

                            # Add all available attributes to the span
                            attribute_kwargs = {
                                "service_name": service_class_name,
                                "model": _get_model_name(self),
                                "stream": True,  # Most LLM services use streaming
                                "parameters": params,
                            }

                            # Add optional attributes only if they exist
                            if serialized_messages:
                                attribute_kwargs["messages"] = serialized_messages
                            if serialized_tools:
                                attribute_kwargs["tools"] = serialized_tools
                                attribute_kwargs["tool_count"] = tool_count
                            if system_message:
                                attribute_kwargs["system_instructions"] = system_message

                            # Add all gathered attributes to the span
                            add_llm_span_attributes(span=current_span, **attribute_kwargs)

                        except Exception as e:
                            logging.warning(f"Error setting up LLM tracing: {e}")
                            # Don't raise - let the function execute anyway

                        # Run function with modified push_frame to capture the output
                        fn_called = True
                        result = await f(self, context, *args, **kwargs)

                        return result

                    finally:
                        # Always restore the original methods
                        self.push_frame = original_push_frame

                        if (
                            "original_start_llm_usage_metrics" in locals()
                            and original_start_llm_usage_metrics
                        ):
                            self.start_llm_usage_metrics = original_start_llm_usage_metrics

                        # Attach whatever output text we accumulated so
                        # far. Doing this in finally captures partial
                        # output when ``f`` is cancelled or raises mid-
                        # stream (e.g. interruption during LLM
                        # generation), rather than only on clean
                        # completion.
                        if output_text:
                            current_span.set_attribute("output", output_text)

                        # Update TTFB metric
                        ttfb: float | None = getattr(getattr(self, "_metrics", None), "ttfb", None)
                        if ttfb is not None:
                            current_span.set_attribute("metrics.ttfb", ttfb)
            except Exception as e:
                if fn_called:
                    raise
                logging.error(f"Error in LLM tracing (continuing without tracing): {e}")
                return await f(self, context, *args, **kwargs)

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def traced_gemini_live(operation: str) -> Callable:
    """Trace Gemini Live service methods with operation-specific attributes.

    This decorator automatically captures relevant information based on the operation type:

    - llm_setup: Configuration, tools definitions, and system instructions
    - llm_tool_call: Function call information
    - llm_tool_result: Function execution results
    - llm_response: Complete LLM response with usage and output

    Args:
        operation: The operation name (matches the event type being handled).

    Returns:
        Wrapped method with Gemini Live specific tracing.
    """
    if not is_tracing_available():
        return _noop_decorator

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not getattr(self, "_tracing_enabled", False):
                return await func(self, *args, **kwargs)

            fn_called = False
            try:
                service_class_name = self.__class__.__name__
                span_name = f"{operation}"

                # Get the parent context - turn context if available, otherwise service context
                parent_context = _get_turn_context(self) or _get_parent_service_context(self)

                # Create a new span as child of the turn span or service span
                tracer = trace.get_tracer("pipecat")
                with tracer.start_as_current_span(
                    span_name, context=parent_context
                ) as current_span:
                    try:
                        # Base service attributes
                        model_name = _get_model_name(self)
                        voice_id = getattr(self, "_voice_id", None)
                        language_code = getattr(self, "_language_code", None)
                        settings = getattr(self, "_settings", None)

                        # Get modalities if available
                        modalities = None
                        if settings and hasattr(settings, "modalities"):
                            modality_obj = settings.modalities
                            if hasattr(modality_obj, "value"):
                                modalities = modality_obj.value
                            else:
                                modalities = str(modality_obj)

                        # Operation-specific attribute collection
                        operation_attrs = {}

                        if operation == "llm_setup":
                            # Capture detailed tool information
                            tools = getattr(self, "_tools", None)
                            if tools:
                                # Handle different tool formats
                                tools_list = []
                                tools_serialized = None

                                try:
                                    if hasattr(tools, "standard_tools"):
                                        # ToolsSchema object
                                        tools_list = tools.standard_tools
                                        # Serialize the tools for detailed inspection
                                        tools_serialized = json.dumps(
                                            [
                                                {
                                                    "name": tool.name
                                                    if hasattr(tool, "name")
                                                    else tool.get("name", "unknown"),
                                                    "description": tool.description
                                                    if hasattr(tool, "description")
                                                    else tool.get("description", ""),
                                                    "properties": tool.properties
                                                    if hasattr(tool, "properties")
                                                    else tool.get("properties", {}),
                                                    "required": tool.required
                                                    if hasattr(tool, "required")
                                                    else tool.get("required", []),
                                                }
                                                for tool in tools_list
                                            ]
                                        )
                                    elif isinstance(tools, list):
                                        # List of tool dictionaries or objects
                                        tools_list = tools
                                        tools_serialized = json.dumps(
                                            [
                                                {
                                                    "name": tool.get("name", "unknown")
                                                    if isinstance(tool, dict)
                                                    else getattr(tool, "name", "unknown"),
                                                    "description": tool.get("description", "")
                                                    if isinstance(tool, dict)
                                                    else getattr(tool, "description", ""),
                                                    "properties": tool.get("properties", {})
                                                    if isinstance(tool, dict)
                                                    else getattr(tool, "properties", {}),
                                                    "required": tool.get("required", [])
                                                    if isinstance(tool, dict)
                                                    else getattr(tool, "required", []),
                                                }
                                                for tool in tools_list
                                            ]
                                        )

                                    if tools_list:
                                        operation_attrs["tools"] = tools_list
                                        operation_attrs["tools_serialized"] = tools_serialized

                                except Exception as e:
                                    logging.warning(f"Error serializing tools for tracing: {e}")
                                    # Fallback to basic tool count
                                    if tools_list:
                                        operation_attrs["tools"] = tools_list

                            # Capture system instruction information
                            system_instruction = getattr(self, "_system_instruction", None)
                            if system_instruction:
                                operation_attrs["system_instruction"] = system_instruction[
                                    :500
                                ]  # Truncate if very long

                            # Capture context system instructions if available
                            if hasattr(self, "_context") and self._context:
                                try:
                                    context_system = self._context.extract_system_instructions()
                                    if context_system:
                                        operation_attrs["context_system_instruction"] = (
                                            context_system[:500]
                                        )  # Truncate if very long
                                except Exception as e:
                                    logging.warning(
                                        f"Error extracting context system instructions: {e}"
                                    )

                        elif operation == "llm_tool_call" and args:
                            # Extract tool call information
                            msg = args[0] if args else None
                            if msg and hasattr(msg, "tool_call") and msg.tool_call.function_calls:
                                function_calls = msg.tool_call.function_calls
                                if function_calls:
                                    # Add information about the first function call
                                    call = function_calls[0]
                                    operation_attrs["tool.function_name"] = call.name
                                    operation_attrs["tool.call_id"] = call.id
                                    operation_attrs["tool.calls_count"] = len(function_calls)

                                    # Add all function names being called
                                    all_function_names = [c.name for c in function_calls]
                                    operation_attrs["tool.all_function_names"] = ",".join(
                                        all_function_names
                                    )

                                    # Add arguments for the first call (truncated if too long)
                                    try:
                                        args_str = json.dumps(call.args) if call.args else "{}"
                                        if len(args_str) > 1000:
                                            args_str = args_str[:1000] + "..."
                                        operation_attrs["tool.arguments"] = args_str
                                    except Exception:
                                        operation_attrs["tool.arguments"] = str(call.args)[:1000]

                        elif operation == "llm_tool_result" and args:
                            # Extract tool result information
                            tool_result_message = args[0] if args else None
                            if tool_result_message and isinstance(tool_result_message, dict):
                                # Extract the tool call information
                                tool_call_id = tool_result_message.get("tool_call_id")
                                tool_call_name = tool_result_message.get("tool_call_name")
                                result_content = tool_result_message.get("content")

                                if tool_call_id:
                                    operation_attrs["tool.call_id"] = tool_call_id
                                if tool_call_name:
                                    operation_attrs["tool.function_name"] = tool_call_name

                                # Parse and capture the result
                                if result_content:
                                    try:
                                        result = json.loads(result_content)
                                        # Serialize the result, truncating if too long
                                        result_str = json.dumps(result)
                                        if len(result_str) > 2000:  # Larger limit for results
                                            result_str = result_str[:2000] + "..."
                                        operation_attrs["tool.result"] = result_str

                                        # Add result status/success indicator if present
                                        if isinstance(result, dict):
                                            if "error" in result:
                                                operation_attrs["tool.result_status"] = "error"
                                            elif "success" in result:
                                                operation_attrs["tool.result_status"] = "success"
                                            else:
                                                operation_attrs["tool.result_status"] = "completed"

                                    except json.JSONDecodeError:
                                        operation_attrs["tool.result"] = (
                                            f"Invalid JSON: {str(result_content)[:500]}"
                                        )
                                        operation_attrs["tool.result_status"] = "parse_error"
                                    except Exception as e:
                                        operation_attrs["tool.result"] = (
                                            f"Error processing result: {str(e)}"
                                        )
                                        operation_attrs["tool.result_status"] = "processing_error"

                        elif operation == "llm_response" and args:
                            # Extract usage and response metadata from turn complete event
                            msg = args[0] if args else None
                            if msg and hasattr(msg, "usage_metadata") and msg.usage_metadata:
                                usage = msg.usage_metadata

                                # Token usage - basic attributes for span visibility
                                if hasattr(usage, "prompt_token_count"):
                                    operation_attrs["tokens.prompt"] = usage.prompt_token_count or 0
                                if hasattr(usage, "response_token_count"):
                                    operation_attrs["tokens.completion"] = (
                                        usage.response_token_count or 0
                                    )
                                if hasattr(usage, "total_token_count"):
                                    operation_attrs["tokens.total"] = usage.total_token_count or 0

                            # Get output text and modality from service state
                            text = getattr(self, "_bot_text_buffer", "")
                            audio_text = getattr(self, "_llm_output_buffer", "")

                            if text:
                                # TEXT modality
                                operation_attrs["output"] = text
                                operation_attrs["output_modality"] = "TEXT"
                            elif audio_text:
                                # AUDIO modality
                                operation_attrs["output"] = audio_text
                                operation_attrs["output_modality"] = "AUDIO"

                            # Add turn completion status
                            if (
                                msg
                                and hasattr(msg, "server_content")
                                and msg.server_content.turn_complete
                            ):
                                operation_attrs["turn_complete"] = True

                        # Add all attributes to the span
                        add_gemini_live_span_attributes(
                            span=current_span,
                            service_name=service_class_name,
                            model=model_name,
                            operation_name=operation,
                            voice_id=voice_id,
                            language=language_code,
                            modalities=modalities,
                            settings=settings,
                            **operation_attrs,
                        )

                        # For llm_response operation, also handle token usage metrics
                        if operation == "llm_response" and hasattr(self, "start_llm_usage_metrics"):
                            msg = args[0] if args else None
                            if msg and hasattr(msg, "usage_metadata") and msg.usage_metadata:
                                usage = msg.usage_metadata
                                # Create LLMTokenUsage object
                                from pipecat.metrics.metrics import LLMTokenUsage

                                tokens = LLMTokenUsage(
                                    prompt_tokens=usage.prompt_token_count or 0,
                                    completion_tokens=usage.response_token_count or 0,
                                    total_tokens=usage.total_token_count or 0,
                                )
                                _add_token_usage_to_span(current_span, tokens)

                        # Capture TTFB metric if available
                        ttfb = getattr(getattr(self, "_metrics", None), "ttfb", None)
                        if ttfb is not None:
                            current_span.set_attribute("metrics.ttfb", ttfb)

                        # Run the original function
                        fn_called = True
                        result = await func(self, *args, **kwargs)

                        return result

                    except Exception as e:
                        current_span.record_exception(e)
                        current_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        raise

            except Exception as e:
                if fn_called:
                    raise
                logging.error(f"Error in Gemini Live tracing (continuing without tracing): {e}")
                return await func(self, *args, **kwargs)

        return wrapper

    return decorator


def traced_openai_realtime(operation: str) -> Callable:
    """Trace OpenAI Realtime service methods with operation-specific attributes.

    This decorator automatically captures relevant information based on the operation type:

    - llm_setup: Session configuration and tools
    - llm_request: Context and input messages
    - llm_response: Usage metadata, output, and function calls

    Args:
        operation: The operation name (matches the event type being handled).

    Returns:
        Wrapped method with OpenAI Realtime specific tracing.
    """
    if not is_tracing_available():
        return _noop_decorator

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not getattr(self, "_tracing_enabled", False):
                return await func(self, *args, **kwargs)

            fn_called = False
            try:
                service_class_name = self.__class__.__name__
                span_name = f"{operation}"

                # Get the parent context - turn context if available, otherwise service context
                parent_context = _get_turn_context(self) or _get_parent_service_context(self)

                # Create a new span as child of the turn span or service span
                tracer = trace.get_tracer("pipecat")
                with tracer.start_as_current_span(
                    span_name, context=parent_context
                ) as current_span:
                    try:
                        # Base service attributes
                        model_name = _get_model_name(self)

                        # Operation-specific attribute collection
                        operation_attrs = {}

                        if operation == "llm_setup":
                            # Capture session properties and tools
                            session_properties = getattr(self, "_session_properties", None)
                            if session_properties:
                                try:
                                    # Convert to dict for easier processing
                                    if hasattr(session_properties, "model_dump"):
                                        props_dict = session_properties.model_dump()
                                    elif hasattr(session_properties, "__dict__"):
                                        props_dict = session_properties.__dict__
                                    else:
                                        props_dict = {}

                                    operation_attrs["session_properties"] = props_dict

                                    # Extract tools if available
                                    tools = props_dict.get("tools")
                                    if tools:
                                        operation_attrs["tools"] = tools
                                        try:
                                            operation_attrs["tools_serialized"] = json.dumps(tools)
                                        except Exception as e:
                                            logging.warning(f"Error serializing OpenAI tools: {e}")

                                    # Extract instructions
                                    instructions = props_dict.get("instructions")
                                    if instructions:
                                        operation_attrs["instructions"] = instructions[:500]

                                except Exception as e:
                                    logging.warning(f"Error processing session properties: {e}")

                            # Also check context for tools
                            if hasattr(self, "_context") and self._context:
                                try:
                                    context_tools = getattr(self._context, "tools", None)
                                    if context_tools and not operation_attrs.get("tools"):
                                        operation_attrs["tools"] = context_tools
                                        operation_attrs["tools_serialized"] = json.dumps(
                                            context_tools
                                        )
                                except Exception as e:
                                    logging.warning(f"Error extracting context tools: {e}")

                        elif operation == "llm_request":
                            # Capture context messages being sent
                            if hasattr(self, "_context") and self._context:
                                try:
                                    messages = self.get_llm_adapter().get_messages_for_logging(
                                        self._context
                                    )
                                    if messages:
                                        operation_attrs["context_messages"] = json.dumps(messages)
                                except Exception as e:
                                    logging.warning(f"Error getting context messages: {e}")

                        elif operation == "llm_response" and args:
                            # Extract usage and response metadata
                            evt = args[0] if args else None
                            if evt and hasattr(evt, "response"):
                                response = evt.response

                                # Token usage - basic attributes for span visibility
                                if hasattr(response, "usage"):
                                    usage = response.usage
                                    if hasattr(usage, "input_tokens"):
                                        operation_attrs["tokens.prompt"] = usage.input_tokens
                                    if hasattr(usage, "output_tokens"):
                                        operation_attrs["tokens.completion"] = usage.output_tokens
                                    if hasattr(usage, "total_tokens"):
                                        operation_attrs["tokens.total"] = usage.total_tokens

                                # Response status and metadata
                                if hasattr(response, "status"):
                                    operation_attrs["response.status"] = response.status

                                if hasattr(response, "id"):
                                    operation_attrs["response.id"] = response.id

                                # Output items and extract transcript for output field
                                if hasattr(response, "output") and response.output:
                                    operation_attrs["response.output_items"] = len(response.output)

                                    # Extract assistant transcript and function calls
                                    assistant_transcript = ""
                                    function_calls = []

                                    for item in response.output:
                                        if (
                                            hasattr(item, "content")
                                            and item.content
                                            and hasattr(item, "role")
                                            and item.role == "assistant"
                                        ):
                                            for content in item.content:
                                                if (
                                                    hasattr(content, "transcript")
                                                    and content.transcript
                                                ):
                                                    assistant_transcript += content.transcript + " "

                                        elif hasattr(item, "type") and item.type == "function_call":
                                            function_call_info = {
                                                "name": getattr(item, "name", "unknown"),
                                                "call_id": getattr(item, "call_id", "unknown"),
                                            }
                                            if hasattr(item, "arguments"):
                                                args_str = item.arguments
                                                if len(args_str) > 500:
                                                    args_str = args_str[:500] + "..."
                                                function_call_info["arguments"] = args_str
                                            function_calls.append(function_call_info)

                                    if assistant_transcript.strip():
                                        operation_attrs["output"] = assistant_transcript.strip()

                                    if function_calls:
                                        operation_attrs["function_calls"] = function_calls
                                        operation_attrs["function_calls.count"] = len(
                                            function_calls
                                        )
                                        all_names = [call["name"] for call in function_calls]
                                        operation_attrs["function_calls.all_names"] = ",".join(
                                            all_names
                                        )

                        # Add all attributes to the span
                        add_openai_realtime_span_attributes(
                            span=current_span,
                            service_name=service_class_name,
                            model=model_name,
                            operation_name=operation,
                            **operation_attrs,
                        )

                        # For llm_response operation, also handle token usage metrics
                        if operation == "llm_response" and hasattr(self, "start_llm_usage_metrics"):
                            evt = args[0] if args else None
                            if evt and hasattr(evt, "response") and hasattr(evt.response, "usage"):
                                usage = evt.response.usage
                                # Create LLMTokenUsage object
                                from pipecat.metrics.metrics import LLMTokenUsage

                                tokens = LLMTokenUsage(
                                    prompt_tokens=getattr(usage, "input_tokens", 0),
                                    completion_tokens=getattr(usage, "output_tokens", 0),
                                    total_tokens=getattr(usage, "total_tokens", 0),
                                )
                                _add_token_usage_to_span(current_span, tokens)

                            # Capture TTFB metric if available
                            ttfb = getattr(getattr(self, "_metrics", None), "ttfb", None)
                            if ttfb is not None:
                                current_span.set_attribute("metrics.ttfb", ttfb)

                        # Run the original function
                        fn_called = True
                        result = await func(self, *args, **kwargs)

                        return result

                    except Exception as e:
                        current_span.record_exception(e)
                        current_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                        raise

            except Exception as e:
                if fn_called:
                    raise
                logging.error(f"Error in OpenAI Realtime tracing (continuing without tracing): {e}")
                return await func(self, *args, **kwargs)

        return wrapper

    return decorator

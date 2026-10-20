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

import asyncio
import functools
import inspect
import json
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

# Type imports for type checking only
if TYPE_CHECKING:
    from opentelemetry import context as context_api
    from opentelemetry import trace

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    ProposedUserStartedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, TTFBMetricsData
from pipecat.processors.aggregators.llm_context import NOT_GIVEN
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.tracing.service_attributes import (
    _get_provider_name_from_service_name,
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
P = TypeVar("P", bound=type[FrameProcessor])


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


def _get_system_instruction(service, context) -> str | None:
    """Resolve the effective system instruction for tracing (internal use only).

    System instructions can live on the LLM service (``settings.system_instruction``,
    the norm — and the composed source of truth, see
    ``LLMService._compose_system_instruction``) or be supplied as an initial
    system message in the context. The service setting takes priority,
    matching service behavior.

    Args:
        service: The LLM service instance.
        context: The universal ``LLMContext``, or None.

    Returns:
        The system instruction text, or None if none is present.
    """
    system_instruction = getattr(getattr(service, "settings", None), "system_instruction", None)
    if system_instruction:
        return system_instruction

    if not context:
        return None

    # Fall back to extracting from context messages
    ctx_messages = context.get_messages()
    if ctx_messages:
        first = ctx_messages[0]
        if isinstance(first, dict) and first.get("role") == "system":
            content = first.get("content")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                return " ".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
    return None


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
            span.set_attribute(
                "gen_ai.usage.reasoning.output_tokens", token_usage["reasoning_tokens"]
            )
        if "input_audio_tokens" in token_usage and token_usage["input_audio_tokens"] is not None:
            span.set_attribute("gen_ai.usage.audio.input_tokens", token_usage["input_audio_tokens"])
        if "output_audio_tokens" in token_usage and token_usage["output_audio_tokens"] is not None:
            span.set_attribute(
                "gen_ai.usage.audio.output_tokens", token_usage["output_audio_tokens"]
            )
        if (
            "cache_read_input_audio_tokens" in token_usage
            and token_usage["cache_read_input_audio_tokens"] is not None
        ):
            span.set_attribute(
                "gen_ai.usage.audio.cache_read.input_tokens",
                token_usage["cache_read_input_audio_tokens"],
            )
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
            span.set_attribute("gen_ai.usage.reasoning.output_tokens", reasoning_tokens)

        input_audio_tokens = getattr(token_usage, "input_audio_tokens", None)
        if input_audio_tokens is not None:
            span.set_attribute("gen_ai.usage.audio.input_tokens", input_audio_tokens)

        output_audio_tokens = getattr(token_usage, "output_audio_tokens", None)
        if output_audio_tokens is not None:
            span.set_attribute("gen_ai.usage.audio.output_tokens", output_audio_tokens)

        cache_read_input_audio_tokens = getattr(token_usage, "cache_read_input_audio_tokens", None)
        if cache_read_input_audio_tokens is not None:
            span.set_attribute(
                "gen_ai.usage.audio.cache_read.input_tokens", cache_read_input_audio_tokens
            )


def _add_stt_usage_to_span(span, usage):
    """Add STT usage metrics to a span (internal use only).

    Args:
        span: The span to add usage metrics to.
        usage: ``STTUsage`` object with audio seconds.
    """
    if not is_tracing_available() or usage is None:
        return

    span.set_attribute("metrics.audio_seconds", usage.audio_seconds)


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
            getattr(service, "_tts_texts", {}).pop(context_id, None)
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

        def apply_text_attributes(span, texts):
            """Set the accumulated text attributes on a TTS span.

            One audio context (and thus one span) typically receives
            several ``run_tts`` calls — one per aggregated sentence — so
            the attributes always reflect the full accumulated text, not
            just the latest call.
            """
            span.set_attribute("text", " ".join(texts))
            span.set_attribute("metrics.character_count", sum(len(t) for t in texts))

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
            service._tts_texts = {}

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

                        # Apply any text buffered by run_tts before the span
                        # existed — some services (e.g. the ElevenLabs
                        # websocket path) create the audio context inside
                        # run_tts, after the text-attach hook has run.
                        texts = service._tts_texts.get(context_id)
                        if texts:
                            apply_text_attributes(span, texts)
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
                            service._tts_texts.pop(context_id, None)
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
                    playing_id = service._audio_context_manager.playing_context_id
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
                service._tts_texts.pop(context_id, None)
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
                playing_id = service._audio_context_manager.playing_context_id
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

            setattr(patched_setup, "__tts_tracing_setup_wrapped__", True)  # noqa: B010
            owner.setup = patched_setup

        def attach_run_tts_attributes(service, text, args, kwargs):
            """Attach text-specific attributes to the in-flight TTS span.

            Texts accumulate per audio context: with sentence aggregation,
            one span covers several ``run_tts`` calls. The text may also
            arrive before the span exists — some services create the audio
            context inside ``run_tts`` — in which case it is buffered here
            and applied by ``create_audio_context`` once the span opens.
            """
            if not getattr(service, "_tracing_enabled", False):
                return
            try:
                if not text:
                    return
                buffers = getattr(service, "_tts_texts", None)
                if buffers is None:
                    return
                context_id = args[0] if args else kwargs.get("context_id")
                texts = buffers.setdefault(context_id, [])
                texts.append(text)
                entry = getattr(service, "_tts_spans", {}).get(context_id)
                if entry:
                    apply_text_attributes(entry["span"], texts)
            except Exception as e:
                logging.warning(f"Error attaching TTS text to span: {e}")

        def make_run_tts_wrapper():
            """Build the wrapper around ``run_tts`` that adds per-call attributes.

            Span lifetime is owned by the audio-context patches. This
            wrapper only records the call's text against the audio
            context, accumulating across calls; the attributes land on
            the span opened by ``create_audio_context`` whether that
            happened before this call or happens inside it.
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
    - Usage metrics (audio seconds) when the service reports them via
      ``start_stt_usage_metrics``

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
        def attach_pending_usage(state):
            """Attach accumulated usage to the open span and clear it.

            Called at every span-close site so usage can't bleed into a
            later span. Usage is accumulated (rather than attached as it
            arrives) because services emit it just before pushing the
            finalized transcript — at which point the span may not be
            open yet — and a single span may cover several usage events.
            """
            usage = state.get("pending_usage")
            if usage is not None:
                _add_stt_usage_to_span(state["span"], usage)
                state["pending_usage"] = None

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
                    attach_pending_usage(state)
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
                        attach_pending_usage(state)
                        span.end()
                        state["span"] = None
                        state["segment_start_time"] = None
                        state["segments"] = []

            @functools.wraps(original_push_frame)
            async def patched_push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
                state = getattr(self, "_stt_span_state", None)
                if state is None:
                    state = {
                        "span": None,
                        "segment_start_time": None,
                        "segments": [],
                        "pending_usage": None,
                    }
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

            setattr(patched_push_frame, "__stt_tracing_push_frame_wrapped__", True)  # noqa: B010
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
                    attach_pending_usage(state)
                    span.end(end_time=int(end_time * 1e9))
                    state["span"] = None
                    state["segment_start_time"] = None
                    state["segments"] = []
                except Exception as e:
                    logging.warning(f"Error in STT stop_ttfb_metrics tracing: {e}")

            setattr(patched_stop, "__stt_tracing_stop_ttfb_wrapped__", True)  # noqa: B010
            owner.stop_ttfb_metrics = patched_stop

        def patch_start_stt_usage_metrics(owner):
            """Wrap ``owner.start_stt_usage_metrics`` to accumulate usage for the span.

            Usage is merged into the span state's ``pending_usage`` and
            attached when the span closes (see ``attach_pending_usage``).
            Values are summed across events.
            """
            original_start = owner.start_stt_usage_metrics
            if getattr(original_start, "__stt_tracing_usage_wrapped__", False):
                return

            @functools.wraps(original_start)
            async def patched_start(self, usage):
                await original_start(self, usage)
                if not getattr(self, "_tracing_enabled", False):
                    return
                try:
                    state = getattr(self, "_stt_span_state", None)
                    if state is None:
                        state = {
                            "span": None,
                            "segment_start_time": None,
                            "segments": [],
                            "pending_usage": None,
                        }
                        self._stt_span_state = state
                    pending = state.get("pending_usage")
                    if pending is None:
                        state["pending_usage"] = usage.model_copy()
                    else:
                        pending.audio_seconds += usage.audio_seconds
                except Exception as e:
                    logging.warning(f"Error in STT usage tracing: {e}")

            setattr(patched_start, "__stt_tracing_usage_wrapped__", True)  # noqa: B010
            owner.start_stt_usage_metrics = patched_start

        class _TracedSTTDescriptor:
            """Class-level descriptor that wires up STT tracing at class definition time.

            ``__set_name__`` fires when the class body finishes evaluating,
            giving us a chance to wrap the owner's ``push_frame`` so that
            VAD, transcription, and finalization events drive the span
            lifecycle, to wrap ``stop_ttfb_metrics`` so the TTFB-timeout
            path can attach metrics and close the span when no finalized
            transcript ever arrives, and to wrap ``start_stt_usage_metrics``
            so usage lands on the span at close. The decorated method
            itself runs unchanged.
            """

            def __set_name__(self, owner, attr_name):
                patch_push_frame(owner)
                patch_stop_ttfb_metrics(owner)
                patch_start_stt_usage_metrics(owner)
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
                            system_message = _get_system_instruction(self, context)

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

    The only operation left is ``llm_setup``: configuration, tool definitions
    and the system instruction, recorded when the session is set up. Replies
    and tool calls are traced by :func:`traced_realtime_llm`.

    Args:
        operation: The operation name. Only ``"llm_setup"`` is meaningful.

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

                            # Capture the effective system instruction:
                            # settings.system_instruction takes priority, else
                            # an initial system message from the context.
                            system_instruction = _get_system_instruction(
                                self, getattr(self, "_context", None)
                            )
                            if system_instruction:
                                operation_attrs["gen_ai.system_instructions"] = system_instruction[
                                    :500
                                ]  # Truncate if very long

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

    The only operation left is ``llm_setup``: session configuration, tools and
    instructions, recorded when the session is created. Replies and tool calls
    are traced by :func:`traced_realtime_llm`.

    Args:
        operation: The operation name. Only ``"llm_setup"`` is meaningful.

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
                                        tools = None
                                        if hasattr(self, "get_llm_adapter"):
                                            adapter = self.get_llm_adapter()
                                            tools = adapter.from_standard_tools(context_tools)

                                        if tools is not None and tools is not NOT_GIVEN:
                                            operation_attrs["tools"] = tools
                                            try:
                                                operation_attrs["tools_serialized"] = json.dumps(
                                                    tools
                                                )
                                            except Exception as e:
                                                logging.warning(
                                                    f"Error serializing OpenAI Realtime tools: {e}"
                                                )
                                except Exception as e:
                                    logging.warning(f"Error extracting context tools: {e}")

                        # Add all attributes to the span
                        add_openai_realtime_span_attributes(
                            span=current_span,
                            service_name=service_class_name,
                            model=model_name,
                            operation_name=operation,
                            **operation_attrs,
                        )

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


#
# Realtime (speech-to-speech) LLM tracing
#

_REALTIME_TOOL_ATTR_MAX_CHARS = 2000
_REALTIME_REPLY_EXPORT_GRACE_SECS = 2.0


def _sum_token_usage(total: LLMTokenUsage | None, usage: LLMTokenUsage) -> LLMTokenUsage:
    """Add a usage report to a running total, field by field.

    Services that report usage in parts (Nova Sonic's partial ``usageEvent``
    reports, Gemini's per-call reports on a held thinking turn) are summed
    into one figure for the reply. A field that neither report sets stays
    ``None``, so "not reported" is kept apart from zero.
    """
    if total is None:
        return usage.model_copy()

    def add(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return a + b

    return LLMTokenUsage(
        prompt_tokens=total.prompt_tokens + usage.prompt_tokens,
        completion_tokens=total.completion_tokens + usage.completion_tokens,
        total_tokens=total.total_tokens + usage.total_tokens,
        cache_read_input_tokens=add(total.cache_read_input_tokens, usage.cache_read_input_tokens),
        cache_creation_input_tokens=add(
            total.cache_creation_input_tokens, usage.cache_creation_input_tokens
        ),
        reasoning_tokens=add(total.reasoning_tokens, usage.reasoning_tokens),
        input_audio_tokens=add(total.input_audio_tokens, usage.input_audio_tokens),
        output_audio_tokens=add(total.output_audio_tokens, usage.output_audio_tokens),
        cache_read_input_audio_tokens=add(
            total.cache_read_input_audio_tokens, usage.cache_read_input_audio_tokens
        ),
    )


def _join_text_pieces(pieces: list[tuple[str, bool]]) -> str:
    """Join text pieces, adding a space between pieces that don't carry their own."""
    out = ""
    for text, includes_spaces in pieces:
        if not text:
            continue
        if includes_spaces or not out:
            out += text
        else:
            out += " " + text
    return out.strip()


def _json_truncated(value: Any) -> str:
    """Serialize ``value`` as JSON for a span attribute, truncated to a sane size."""
    try:
        text = json.dumps(value, default=str)
    except Exception:
        text = str(value)
    if len(text) > _REALTIME_TOOL_ATTR_MAX_CHARS:
        text = text[:_REALTIME_TOOL_ATTR_MAX_CHARS] + "..."
    return text


class _RealtimeReply:
    """One model reply of a realtime service, from end of user speech to reply end.

    The span is opened when the reply starts and exported later, once the data
    that arrives after the reply ends (usage, the user transcript) is in, or
    the grace period runs out.
    """

    def __init__(self, span, start_ns: int, answers_speech: bool, seq: int):
        self.span = span
        self.start_ns = start_ns
        self.end_ns: int | None = None
        self.answers_speech = answers_speech
        self.seq = seq
        self.interrupted = False
        self.input_pieces: list[str] = []
        self.llm_text: list[tuple[str, bool]] = []
        self.tts_text: list[tuple[str, bool]] = []
        self.had_audio = False
        self.ttfb: float | None = None
        self.usage: LLMTokenUsage | None = None
        self.exported = False

    @property
    def complete(self) -> bool:
        """Whether the reply can be exported before the grace period runs out.

        Usage is reported once, so a reply that answers no speech is done when
        it has usage. A reply that answers speech is never complete early: the
        user's transcript comes in pieces (Gemini Live sends one frame per
        sentence, plus a trailing fragment on a timeout), and nothing marks
        the last one. It waits for the grace period or the next reply.
        """
        return self.usage is not None and not self.answers_speech


class _RealtimeLLMTraceState:
    """Per-service state for :func:`traced_realtime_llm`.

    Everything here changes synchronously, from the frame handlers and the
    usage hook. Frames reach a realtime service's ``push_frame`` from several
    of its tasks, so no state change may ``await`` partway through.
    """

    def __init__(self):
        self.reply: _RealtimeReply | None = None
        self.pending: _RealtimeReply | None = None
        self.seq = 0
        # Speech end since the previous reply opened, in seconds, plus the turn
        # context that was current when it ended.
        self.speech_end: float | None = None
        self.speech_end_turn_ctx = None
        # Earliest speech onset since the last speech end, in seconds.
        self.speech_onset: float | None = None
        # Whether the user has spoken since the last reply that answered them
        # opened. Speech not yet answered keeps its transcripts out of the
        # replies around it.
        self.user_started_since_reply = True
        # Data that arrived while no reply was open, kept for the next one.
        self.held_ttfb: float | None = None
        self.held_text: list[tuple[str, bool]] = []
        self.held_tts_text: list[tuple[str, bool]] = []
        self.held_transcript: list[str] = []
        self.tool_spans: dict[str, Any] = {}


def traced_realtime_llm(cls: P) -> P:
    """Trace a realtime (speech-to-speech) LLM service, one span per reply.

    A realtime model takes audio in and sends audio out; there is no separate
    speech-to-text stage to measure. So instead of ``stt`` and per-handler
    ``llm_response`` spans, the service gets:

    - one ``llm_response`` span per reply, from the end of the user's speech to
      the end of the reply (or the interruption), nested inside its ``turn``
      span. The user's transcript and the reply text are attributes on it,
      along with token usage, TTFB and the modalities.
    - one ``llm_tool_call`` span per function call, beside the reply.

    A reply opens on ``LLMFullResponseStartFrame``, or on the first
    ``FunctionCallInProgressFrame`` when no Start came first. It ends on
    ``LLMFullResponseEndFrame`` or ``InterruptionFrame`` and is exported once
    its late data (usage, the user's transcript) is in: at once for a reply
    that answers no speech, otherwise when the next reply opens or after a
    short grace period.

    The decorator wraps the class's ``push_frame`` (reading each frame after
    the original push), ``start_llm_usage_metrics`` and ``cleanup``. It checks
    ``_tracing_enabled`` when called and is idempotent, so subclasses inherit
    one wrapping. Only frames the service itself pushes drive the spans:
    downstream frames, plus the ``TranscriptionFrame`` realtime services push
    upstream.

    Args:
        cls: The realtime LLM service class.

    Returns:
        The same class, with the tracing wrappers installed.
    """
    if not is_tracing_available():
        return cls

    if getattr(cls, "__realtime_llm_tracing_wrapped__", False):
        return cls

    def get_state(service) -> _RealtimeLLMTraceState:
        state = getattr(service, "_realtime_llm_trace_state", None)
        if state is None:
            state = _RealtimeLLMTraceState()
            service._realtime_llm_trace_state = state
        return state

    def turn_context(service):
        return _get_turn_context(service) or _get_parent_service_context(service)

    def discard_held(state: _RealtimeLLMTraceState):
        state.held_ttfb = None
        state.held_text = []
        state.held_tts_text = []

    def export_reply(service, state: _RealtimeLLMTraceState, reply: _RealtimeReply):
        """Set the reply's attributes and end its span at the recorded end time."""
        if reply.exported:
            return
        reply.exported = True
        if state.pending is reply:
            state.pending = None
        if state.reply is reply:
            state.reply = None
        span = reply.span
        try:
            span.set_attribute("gen_ai.output.type", "speech" if reply.had_audio else "text")
            output = _join_text_pieces(reply.llm_text or reply.tts_text)
            if output:
                span.set_attribute("output", output)
            span.set_attribute("output_modality", "AUDIO" if reply.had_audio else "TEXT")
            if reply.input_pieces:
                span.set_attribute("input", " ".join(p.strip() for p in reply.input_pieces))
                span.set_attribute("input_modality", "AUDIO")
            if reply.ttfb is not None:
                span.set_attribute("metrics.ttfb", reply.ttfb)
                span.set_attribute("gen_ai.response.time_to_first_chunk", reply.ttfb)
            if reply.usage is not None:
                _add_token_usage_to_span(span, reply.usage)
            span.set_attribute("llm.interrupted", reply.interrupted)
        except Exception as e:
            logging.warning(f"Error setting realtime llm_response attributes: {e}")
        end_ns = reply.end_ns if reply.end_ns is not None else time.time_ns()
        span.end(end_time=max(end_ns, reply.start_ns))

    def export_if_complete(service, state: _RealtimeLLMTraceState):
        pending = state.pending
        if pending is not None and pending.complete:
            export_reply(service, state, pending)

    def schedule_grace_export(service, state: _RealtimeLLMTraceState, reply: _RealtimeReply):
        """Export the pending reply after the grace period, unless it was already exported.

        The timer is never cancelled. It checks whether its reply is still the
        one waiting and does nothing otherwise, so no state change has to
        ``await`` it.
        """

        async def grace():
            await asyncio.sleep(_REALTIME_REPLY_EXPORT_GRACE_SECS)
            if state.pending is reply and not reply.exported:
                export_reply(service, state, reply)

        try:
            create_task = getattr(service, "create_task", None)
            if create_task is not None:
                create_task(grace(), name="realtime-llm-trace-export")
            else:
                asyncio.get_running_loop().create_task(grace())
        except Exception as e:
            logging.warning(f"Error scheduling realtime llm_response export: {e}")
            export_reply(service, state, reply)

    def end_reply(service, state: _RealtimeLLMTraceState, end_ns: int, interrupted: bool):
        """Record the end of the open reply and move it to pending."""
        reply = state.reply
        if reply is None:
            return
        reply.end_ns = max(end_ns, reply.start_ns)
        reply.interrupted = interrupted
        state.reply = None
        if state.pending is not None and state.pending is not reply:
            export_reply(service, state, state.pending)
        state.pending = reply
        if reply.complete:
            export_reply(service, state, reply)
        else:
            schedule_grace_export(service, state, reply)

    def open_reply(service, state: _RealtimeLLMTraceState):
        if state.pending is not None:
            export_reply(service, state, state.pending)
        speech_end = state.speech_end
        answers_speech = speech_end is not None
        start_secs = speech_end if speech_end is not None else time.time()
        parent = (state.speech_end_turn_ctx if answers_speech else None) or turn_context(service)
        tracer = trace.get_tracer("pipecat")
        start_ns = int(start_secs * 1e9)
        span = tracer.start_span("llm_response", context=parent, start_time=start_ns)
        state.seq += 1
        reply = _RealtimeReply(span, start_ns, answers_speech, state.seq)
        try:
            span.set_attribute(
                "gen_ai.provider.name",
                _get_provider_name_from_service_name(service.__class__.__name__),
            )
            span.set_attribute("gen_ai.request.model", _get_model_name(service))
            span.set_attribute("gen_ai.operation.name", "chat")
        except Exception as e:
            logging.warning(f"Error setting realtime llm_response attributes: {e}")
        # Data that arrived before the start frame belongs to this reply.
        reply.ttfb = state.held_ttfb
        reply.llm_text = list(state.held_text)
        reply.tts_text = list(state.held_tts_text)
        if answers_speech and state.held_transcript:
            reply.input_pieces = list(state.held_transcript)
            state.held_transcript = []
        discard_held(state)
        state.speech_end = None
        state.speech_end_turn_ctx = None
        # speech_onset is kept: a user already speaking when the reply starts
        # ends it at their onset, clamped to the reply's own start.
        if answers_speech:
            # This reply answers the user's speech; what they say from now on
            # is theirs. A reply that answers no speech (a tool follow-up)
            # leaves an utterance already under way to the reply that will
            # answer it.
            state.user_started_since_reply = False
        state.reply = reply

    def mark_pending_interrupted(state: _RealtimeLLMTraceState):
        """Mark a reply the service ended itself on interruption.

        The OpenAI family pushes ``LLMFullResponseEndFrame`` from its
        interruption handler, so the reply has already ended when the
        ``InterruptionFrame`` passes. The user's speech onset falling inside
        the reply tells that End apart from one that ended the reply normally
        before the user spoke.
        """
        pending = state.pending
        onset = state.speech_onset
        if pending is None or pending.exported or pending.end_ns is None or onset is None:
            return
        onset_ns = int(onset * 1e9)
        if onset_ns < pending.end_ns:
            pending.interrupted = True
            pending.end_ns = max(onset_ns, pending.start_ns)

    def interrupted_end_ns(state: _RealtimeLLMTraceState) -> int:
        """When an interrupted reply ends: the user's speech onset, if known."""
        if state.speech_onset is not None:
            return int(state.speech_onset * 1e9)
        return time.time_ns()

    def note_speech_onset(state: _RealtimeLLMTraceState, onset: float):
        if state.speech_onset is None or onset < state.speech_onset:
            state.speech_onset = onset
        state.user_started_since_reply = True
        discard_held(state)

    def note_speech_end(service, state: _RealtimeLLMTraceState, end: float):
        state.speech_end = end
        state.speech_end_turn_ctx = _get_turn_context(service)
        state.speech_onset = None
        discard_held(state)

    def handle_transcript(service, state: _RealtimeLLMTraceState, text: str):
        """Route a user transcript to the reply it answers, or hold it for the next."""
        if not text:
            return
        target = state.reply or state.pending
        if target is not None and not state.user_started_since_reply:
            target.input_pieces.append(text)
        else:
            state.held_transcript.append(text)

    def close_all(service, state: _RealtimeLLMTraceState):
        """Close everything still open: pipeline end or service cleanup."""
        if state.reply is not None:
            end_reply(service, state, time.time_ns(), interrupted=True)
        if state.pending is not None:
            export_reply(service, state, state.pending)
        for span in list(state.tool_spans.values()):
            try:
                span.set_attribute("tool.cancelled", True)
                span.end()
            except Exception as e:
                logging.warning(f"Error ending realtime llm_tool_call span: {e}")
        state.tool_spans = {}

    def handle_tool_frame(service, state: _RealtimeLLMTraceState, frame):
        if isinstance(frame, FunctionCallInProgressFrame):
            if frame.tool_call_id in state.tool_spans:
                return
            tracer = trace.get_tracer("pipecat")
            span = tracer.start_span("llm_tool_call", context=turn_context(service))
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("gen_ai.tool.name", frame.function_name)
            span.set_attribute("gen_ai.tool.call.id", frame.tool_call_id)
            span.set_attribute("gen_ai.tool.call.arguments", _json_truncated(frame.arguments))
            state.tool_spans[frame.tool_call_id] = span
        elif isinstance(frame, FunctionCallResultFrame):
            properties = frame.properties
            if properties is not None and not properties.is_final:
                return
            span = state.tool_spans.pop(frame.tool_call_id, None)
            if span is None:
                return
            span.set_attribute("gen_ai.tool.call.result", _json_truncated(frame.result))
            if frame.error:
                span.set_attribute("error.type", "tool_error")
                span.set_status(trace.Status(trace.StatusCode.ERROR, frame.error))
            span.end()
        elif isinstance(frame, FunctionCallCancelFrame):
            span = state.tool_spans.pop(frame.tool_call_id, None)
            if span is None:
                return
            span.set_attribute("tool.cancelled", True)
            span.end()

    def handle_frame(service, state: _RealtimeLLMTraceState, frame, direction):
        if direction == FrameDirection.UPSTREAM:
            if isinstance(frame, TranscriptionFrame):
                handle_transcript(service, state, frame.text)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            if state.reply is None:
                open_reply(service, state)
        elif isinstance(frame, LLMFullResponseEndFrame):
            end_reply(service, state, time.time_ns(), interrupted=False)
        elif isinstance(frame, InterruptionFrame):
            if state.reply is not None:
                end_reply(service, state, interrupted_end_ns(state), interrupted=True)
            else:
                mark_pending_interrupted(state)
            discard_held(state)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            note_speech_end(service, state, frame.timestamp - frame.stop_secs)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            note_speech_onset(state, frame.timestamp - frame.start_secs)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # A turn strategy can end the user's turn after a service using its
            # own VAD has started replying; that stop is not the next reply's
            # start. Local VAD, when present, is the better measure.
            if state.reply is None and state.speech_end is None:
                note_speech_end(service, state, time.time())
            else:
                discard_held(state)
        elif isinstance(frame, UserStartedSpeakingFrame):
            note_speech_onset(state, time.time())
        elif isinstance(frame, ProposedUserStartedSpeakingFrame):
            note_speech_onset(state, time.time())
        elif isinstance(frame, LLMTextFrame):
            piece = (frame.text, frame.includes_inter_frame_spaces)
            if state.reply is not None:
                state.reply.llm_text.append(piece)
            else:
                state.held_text.append(piece)
        elif isinstance(frame, TTSTextFrame):
            piece = (frame.text, frame.includes_inter_frame_spaces)
            if state.reply is not None:
                state.reply.tts_text.append(piece)
            else:
                state.held_tts_text.append(piece)
        elif isinstance(frame, TTSAudioRawFrame):
            if state.reply is not None:
                state.reply.had_audio = True
        elif isinstance(frame, MetricsFrame):
            for data in frame.data:
                if isinstance(data, TTFBMetricsData) and data.processor == service.name:
                    if state.reply is not None:
                        if state.reply.ttfb is None:
                            state.reply.ttfb = data.value
                    elif state.held_ttfb is None:
                        state.held_ttfb = data.value
                    break
        elif isinstance(
            frame, (FunctionCallInProgressFrame, FunctionCallResultFrame, FunctionCallCancelFrame)
        ):
            if isinstance(frame, FunctionCallInProgressFrame) and state.reply is None:
                # A function call is model output, so a reply is under way even
                # when no Start frame announced it: OpenAI Realtime pushes none
                # for a response that only calls tools.
                open_reply(service, state)
            handle_tool_frame(service, state, frame)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            close_all(service, state)

    original_push_frame = cls.push_frame

    @functools.wraps(original_push_frame)
    async def patched_push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        await original_push_frame(self, frame, direction)
        if not getattr(self, "_tracing_enabled", False):
            return
        try:
            handle_frame(self, get_state(self), frame, direction)
        except Exception as e:
            logging.warning(f"Error in realtime LLM tracing: {e}")

    original_start_llm_usage_metrics = cls.start_llm_usage_metrics

    @functools.wraps(original_start_llm_usage_metrics)
    async def patched_start_llm_usage_metrics(self, tokens):
        await original_start_llm_usage_metrics(self, tokens)
        if not getattr(self, "_tracing_enabled", False):
            return
        try:
            state = get_state(self)
            target = state.reply or state.pending
            if target is None:
                return
            target.usage = _sum_token_usage(target.usage, tokens)
            if target is state.pending:
                export_if_complete(self, state)
        except Exception as e:
            logging.warning(f"Error in realtime LLM usage tracing: {e}")

    original_cleanup = cls.cleanup

    @functools.wraps(original_cleanup)
    async def patched_cleanup(self):
        await original_cleanup(self)
        if not getattr(self, "_tracing_enabled", False):
            return
        try:
            close_all(self, get_state(self))
        except Exception as e:
            logging.warning(f"Error in realtime LLM cleanup tracing: {e}")

    cls.push_frame = patched_push_frame  # type: ignore[method-assign]
    cls.start_llm_usage_metrics = patched_start_llm_usage_metrics  # type: ignore[method-assign]
    cls.cleanup = patched_cleanup  # type: ignore[method-assign]
    cls.__realtime_llm_tracing_wrapped__ = True  # type: ignore[attr-defined]
    return cls

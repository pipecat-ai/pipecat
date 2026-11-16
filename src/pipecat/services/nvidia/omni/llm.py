#
# Copyright (c) 2026, Daily
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""NVIDIA Nemotron Omni LLM service implementation.

Nemotron Omni models accept speech, images, and video as well as text, and reply
with text. This service extends ``NvidiaLLMService``, so reasoning frames and
every OpenAI-compatible behavior come from there unchanged, and adds audio turns:
in audio input mode, Omni buffers the user's speech between turn boundaries and
transcribes and answers it in a single request, replacing the STT stage of a
cascaded pipeline.

Media in the context, built with ``LLMContext.create_audio_message()``,
``create_image_message()``, or ``create_file_message()`` for audio and video
files, is converted by ``NvidiaOmniLLMAdapter`` to the shape Omni reads.

Refer to the model card for supported inputs and deployment options:
https://build.nvidia.com/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
"""

import asyncio
import base64
import contextlib
import time
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, cast

from loguru import logger
from openai import (
    APIStatusError,
    AsyncOpenAI,
    BadRequestError,
    DefaultAsyncHttpxClient,
    UnprocessableEntityError,
)
from openai._types import NotGiven as OpenAINotGiven

from pipecat.adapters.llm.base_llm_adapter import LLMContextConversionError
from pipecat.adapters.llm.services.nvidia_omni_adapter import NvidiaOmniLLMAdapter
from pipecat.adapters.llm.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.audio.utils import pcm_to_wav
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMServiceMetadataFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMStandardMessage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.nvidia.llm import NvidiaLLMService, NvidiaLLMSettings
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.http import TIMEOUT_EXCEPTIONS, connection_limits
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

DEFAULT_AUDIO_RESPONSE_INSTRUCTION = (
    "Listen to the user's speech in the attached audio. First write exactly what the user "
    "said inside <transcript>...</transcript>, then write your spoken reply inside "
    "<response>...</response>."
)

_TOOL_RESULT_IN_PROGRESS = "IN_PROGRESS"

_CompletionTrigger = Literal["user", "tool"]

_TRANSCRIPT_OPEN = "<transcript>"
_TRANSCRIPT_CLOSE = "</transcript>"
_RESPONSE_OPEN = "<response>"
_RESPONSE_CLOSE = "</response>"


@dataclass
class NvidiaOmniLLMSettings(NvidiaLLMSettings):
    """Settings for NvidiaOmniLLMService.

    Parameters:
        audio_response_instruction: Instruction sent with the user's speech in
            audio input mode. ``None`` uses ``DEFAULT_AUDIO_RESPONSE_INSTRUCTION``.
            A custom instruction should keep asking for the ``<transcript>`` and
            ``<response>`` sections; a reply without them still reaches TTS, but
            the user's turn is then missing from the context.
        min_user_audio_secs: Shortest buffered utterance that starts a turn.
        pre_speech_buffer_secs: Audio retained from before the user started
            speaking, so the start of an utterance is not clipped.
    """

    audio_response_instruction: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    min_user_audio_secs: float | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pre_speech_buffer_secs: float | NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass(frozen=True)
class NvidiaOmniInferenceResult:
    """Result of :meth:`NvidiaOmniLLMService.run_multimodal_inference`.

    Parameters:
        text: The response text.
        reasoning: The reasoning the model produced before responding.
        finish_reason: Why generation stopped (e.g. ``"stop"`` or ``"length"``).
    """

    text: str = ""
    reasoning: str = ""
    finish_reason: str = ""


class NvidiaOmniLLMService(NvidiaLLMService):
    """NVIDIA Nemotron Omni LLM service with text and speech input.

    ``audio_input`` decides the pipeline architecture. In audio input mode
    (the default), Omni hears the user directly, replacing the STT stage::

        transport.input() -> user_aggregator -> llm
        -> tts -> transport.output() -> assistant_aggregator

    With ``audio_input=False``, it is a text LLM in a cascaded pipeline::

        transport.input() -> stt -> user_aggregator -> llm
        -> tts -> transport.output() -> assistant_aggregator

    Every turn runs through the inherited completion path, so tool calling,
    streaming, reasoning frames, and metrics behave as in ``NvidiaLLMService``.
    Media placed in the context is sent in either mode.

    In audio input mode the service behaves like a realtime service that still
    needs a TTS. It:

    - announces itself as a realtime service, which switches the user
      aggregator into realtime mode
    - buffers the user's audio between turn boundaries
    - answers a spoken turn once it ends, after any turn still running
    - answers the speech of a turn interrupted before Omni transcribed it
      together with the user's next utterance
    - asks Omni for a transcript of the user's speech and pushes it upstream as
      a ``TranscriptionFrame``, which the user aggregator writes to the context
    - treats the late context frame carrying that transcript as bookkeeping
      rather than as a turn to answer

    Context frames that need an answer, such as tool results, developer
    messages, and typed user input, still start a completion, so the
    ``LLMRunFrame`` greeting pattern works in both modes.

    As with other realtime services, the user's text is only known once the
    transcript arrives, so read it from ``on_user_turn_message_added`` rather
    than ``on_user_turn_stopped``::

        @user_aggregator.event_handler("on_user_turn_message_added")
        async def on_user_turn_message_added(aggregator, message):
            logger.info(f"User said: {message.content}")
    """

    Settings = NvidiaOmniLLMSettings
    _settings: Settings
    adapter_class = NvidiaOmniLLMAdapter
    supports_developer_role = False

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        audio_input: bool = True,
        context: LLMContext | None = None,
        settings: Settings | None = None,
        request_timeout_secs: float = 120.0,
        **kwargs,
    ):
        """Initialize the NvidiaOmniLLMService.

        Args:
            api_key: NVIDIA API key. Required for the cloud endpoint, not for
                local deployments.
            base_url: OpenAI-compatible endpoint base URL. For local deployments,
                pass the local address (e.g. ``http://localhost:8000/v1``).
            audio_input: Whether Omni hears the user directly, with no STT
                service in the pipeline. ``False`` makes it a text LLM for a
                cascaded pipeline.
            context: Context that audio turns are answered against until the
                first ``LLMContextFrame`` supplies one.
            settings: Runtime-updatable settings.
            request_timeout_secs: HTTP request timeout, in seconds.
            **kwargs: Additional keyword arguments passed to NvidiaLLMService.
        """
        default_settings = self.Settings(
            model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            temperature=0.6,
            top_p=0.95,
            audio_response_instruction=None,
            min_user_audio_secs=0.3,
            pre_speech_buffer_secs=0.2,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        self._request_timeout_secs = request_timeout_secs
        super().__init__(api_key=api_key, base_url=base_url, settings=default_settings, **kwargs)

        self._audio_input = audio_input
        self._context = context

        self._sample_rate = 16000
        self._num_channels = 1
        self._audio_buffer: list[bytes] = []
        self._pre_speech_buffer: list[bytes] = []
        self._audio_user_speaking = False
        self._user_turn_started = False
        self._last_user_stopped_at: float | None = None

        self._turn_tasks: list[asyncio.Task] = []
        self._untranscribed_speech: dict[asyncio.Task, bytes] = {}
        # Set only inside the task running a turn, so that one-shot inferences
        # running concurrently from other tasks never send the turn's audio.
        self._turn_parts_var: ContextVar[list[dict[str, Any]] | None] = ContextVar(
            f"{self}::turn_parts", default=None
        )
        self._transcript_extractor: _TranscriptResponseExtractor | None = None
        self._transcript_emitted = False
        self._last_transcript = ""
        self._completion_tracker = _CompletionTracker()

    def create_client(
        self,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers=None,
        **kwargs,
    ):
        """Create the AsyncOpenAI client with the configured request timeout.

        Args:
            api_key: NVIDIA API key. May be omitted for local deployments.
            base_url: OpenAI-compatible endpoint base URL.
            organization: OpenAI organization ID.
            project: OpenAI project ID.
            default_headers: Additional HTTP headers.
            **kwargs: Additional client configuration arguments.

        Returns:
            Configured AsyncOpenAI client instance.
        """
        return AsyncOpenAI(
            # The client refuses to start without a key, which local
            # deployments don't need.
            api_key=api_key or "not-needed",
            base_url=base_url,
            timeout=self._request_timeout_secs,
            organization=organization,
            project=project,
            http_client=DefaultAsyncHttpxClient(
                limits=connection_limits(
                    max_keepalive_connections=100, max_connections=1000, keepalive_expiry=None
                )
            ),
            default_headers=default_headers,
        )

    def service_metadata_frame(self) -> LLMServiceMetadataFrame:
        """Announce the service as realtime when it accepts audio input.

        With audio input, the user's turn is only known once the model has
        transcribed it. Realtime mode makes the user aggregator write it when
        the assistant response starts, by which point the transcript is known.

        Returns:
            The metadata frame broadcast at pipeline start.
        """
        if not self._audio_input:
            return super().service_metadata_frame()
        return LLMServiceMetadataFrame(service_name=self.name, is_realtime_service=True)

    async def start(self, frame: StartFrame):
        """Start the service.

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        self._reset_session_state()

    async def stop(self, frame: EndFrame):
        """Stop the service once the queued turns finish.

        A response requested before the end frame, such as a goodbye, is
        completed first, as it is when the base service completes context
        frames inline.

        Args:
            frame: The end frame.
        """
        if self._turn_tasks:
            await asyncio.wait(self._turn_tasks)
            self._turn_tasks = []
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the service, cancelling the queued turns.

        Args:
            frame: The cancel frame.
        """
        await self._cancel_turns()
        await super().cancel(frame)

    async def cleanup(self):
        """Clean up the service, cancelling the queued turns."""
        await super().cleanup()
        await self._cancel_turns()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, starting text and audio turns in audio input mode.

        As a text LLM, every frame is handled by ``NvidiaLLMService``. In audio
        input mode, completions run in background tasks, because an audio turn
        starts on a speech boundary rather than on a context frame. Turns run
        in order, as the base service completes context frames, and only an
        ``InterruptionFrame`` cancels them. ``LLMContextFrame`` and
        ``LLMRunFrame`` are consumed; every other frame is forwarded.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        if not self._audio_input:
            await super().process_frame(frame, direction)
            return

        # Skip BaseOpenAILLMService.process_frame, which completes every
        # LLMContextFrame inline.
        await super(BaseOpenAILLMService, self).process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self.stop_all_metrics()
            untranscribed = self._pending_untranscribed_speech()
            await self._cancel_turns()
            if self._audio_user_speaking:
                self._audio_buffer = untranscribed + self._audio_buffer
            else:
                self._audio_buffer = []
                self._pre_speech_buffer = []
        elif isinstance(frame, LLMContextFrame):
            self._context = frame.context
            await self._start_text_turn(frame.context)
            return
        elif isinstance(frame, LLMRunFrame):
            await self._start_text_turn(self._context, force=True)
            return
        elif isinstance(frame, InputAudioRawFrame):
            self._handle_input_audio(frame)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._start_buffering_speech()
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._start_buffering_speech()
            self._user_turn_started = True
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._handle_vad_stopped()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking()

        await self.push_frame(frame, direction)

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build chat completion parameters, appending the current audio turn.

        The user's speech is appended as a trailing user message rather than
        written to the context. Fields left unset are dropped, so only the
        configured token limit is sent.

        Args:
            params_from_context: Parameters derived from the LLM context.

        Returns:
            Dictionary of parameters for the chat completion request.
        """
        params = {
            name: value
            for name, value in super().build_chat_completion_params(params_from_context).items()
            if not isinstance(value, (OpenAINotGiven, NotGiven))
        }
        turn_parts = self._active_turn_parts
        if turn_parts:
            adapter = cast(NvidiaOmniLLMAdapter, self.get_llm_adapter())
            messages = list(params.get("messages") or [])
            messages.append(
                {"role": "user", "content": adapter.to_provider_content_parts(turn_parts)}
            )
            params["messages"] = messages
        return params

    async def run_inference(
        self,
        context: LLMContext,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> str | None:
        """Run a one-shot, out-of-pipeline inference without the current audio turn.

        Args:
            context: The LLM context containing conversation history.
            max_tokens: Optional maximum number of tokens to generate.
            system_instruction: Optional system instruction for this inference.
            response_schema: Optional JSON schema the reply must follow.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        with self._without_turn_parts():
            return await super().run_inference(
                context, max_tokens, system_instruction, response_schema=response_schema
            )

    async def run_multimodal_inference(
        self,
        context: LLMContext,
        *,
        max_tokens: int | None = None,
        reasoning_budget: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
        on_text_delta: Callable[[str], Awaitable[None]] | None = None,
        on_reasoning_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> NvidiaOmniInferenceResult:
        """Run a one-shot, out-of-pipeline inference that also returns the reasoning.

        Extends :meth:`run_inference` with what a worker analyzing media outside
        the pipeline needs: the model's reasoning, text and reasoning deltas as
        they arrive, and a per-call reasoning budget. Media is passed in the
        context, for example with ``LLMContext.create_file_message()``.

        Args:
            context: The LLM context, including any media.
            max_tokens: Overrides the configured token limit.
            reasoning_budget: NVIDIA ``reasoning_budget`` request field, limiting
                how many tokens the model may spend reasoning.
            temperature: Overrides the configured temperature.
            stream: Whether to stream the completion and call the delta callbacks.
            on_text_delta: Called with each response text delta while streaming.
            on_reasoning_delta: Called with each reasoning delta while streaming.

        Returns:
            The generated text, reasoning, and finish reason.
        """
        invocation_params = await self.get_llm_adapter().get_llm_invocation_params(
            context,
            system_instruction=assert_given(self._settings.system_instruction),
            convert_developer_to_user=not self.supports_developer_role,
        )
        with self._without_turn_parts():
            params = self.build_chat_completion_params(invocation_params)
        if max_tokens is not None:
            params.pop("max_completion_tokens", None)
            params["max_tokens"] = max_tokens
        if reasoning_budget is not None:
            params["extra_body"] = {
                **(params.get("extra_body") or {}),
                "reasoning_budget": reasoning_budget,
            }
        if temperature is not None:
            params["temperature"] = temperature

        if not stream:
            params["stream"] = False
            params.pop("stream_options", None)
            completion = await self._client.chat.completions.create(**params)
            choice = completion.choices[0] if completion.choices else None
            message = choice.message if choice else None
            return NvidiaOmniInferenceResult(
                text=_message_text(getattr(message, "content", None)).strip(),
                reasoning=_reasoning_text(message).strip(),
                finish_reason=str(getattr(choice, "finish_reason", None) or ""),
            )

        text = ""
        reasoning = ""
        finish_reason = ""
        response_stream = await self._client.chat.completions.create(**params)
        async for chunk in response_stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = str(choice.finish_reason)
            reasoning_delta = _reasoning_text(choice.delta)
            if reasoning_delta:
                reasoning += reasoning_delta
                if on_reasoning_delta:
                    await on_reasoning_delta(reasoning_delta)
            text_delta = _message_text(getattr(choice.delta, "content", None))
            if text_delta:
                text += text_delta
                if on_text_delta:
                    await on_text_delta(text_delta)
        return NvidiaOmniInferenceResult(
            text=text.strip(), reasoning=reasoning.strip(), finish_reason=finish_reason
        )

    def current_turn_has_user_audio(self) -> bool:
        """Whether the completion in progress carries the user's speech.

        Returns:
            True during an audio turn, False during a text turn or between turns.
        """
        return any(part.get("type") == "input_audio" for part in self._active_turn_parts or ())

    @property
    def _active_turn_parts(self) -> list[dict[str, Any]] | None:
        """Content parts the turn running in the current task adds to its request.

        Returns:
            The parts, or None outside a turn and during one-shot inferences.
        """
        return self._turn_parts_var.get()

    @contextlib.contextmanager
    def _without_turn_parts(self):
        token = self._turn_parts_var.set(None)
        try:
            yield
        finally:
            self._turn_parts_var.reset(token)

    async def _push_llm_text(self, text: str):
        """Split the transcript out of the model's text before it reaches TTS.

        Args:
            text: Visible content from the model, with reasoning removed.
        """
        extractor = self._transcript_extractor
        if extractor is None:
            await super()._push_llm_text(text)
            return
        response_text = extractor.feed(text)
        await self._maybe_push_transcript(extractor)
        if response_text:
            await super()._push_llm_text(response_text)

    async def _finalize_reasoning_state(self, *, flush_buffered_text: bool):
        """Finalize reasoning state, then flush text held by the transcript split.

        Args:
            flush_buffered_text: Whether buffered text may still be forwarded.
                ``False`` when the stream ended early through interruption or
                cancellation.
        """
        await super()._finalize_reasoning_state(flush_buffered_text=flush_buffered_text)
        extractor = self._transcript_extractor
        if extractor is None:
            return
        # The remainder is plain response text, so it must not pass through the
        # extractor again.
        self._transcript_extractor = None
        if not flush_buffered_text:
            return
        remainder = extractor.finalize()
        await self._maybe_push_transcript(extractor)
        if remainder:
            await self._push_llm_text(remainder)

    def _handle_input_audio(self, frame: InputAudioRawFrame):
        self._sample_rate = frame.sample_rate
        self._num_channels = frame.num_channels
        if self._audio_user_speaking:
            self._audio_buffer.append(frame.audio)
            return
        self._pre_speech_buffer.append(frame.audio)
        pre_speech_secs = assert_given(self._settings.pre_speech_buffer_secs)
        max_bytes = int(self._bytes_per_second() * pre_speech_secs)
        total = sum(len(chunk) for chunk in self._pre_speech_buffer)
        while self._pre_speech_buffer and total > max_bytes:
            total -= len(self._pre_speech_buffer.pop(0))

    def _start_buffering_speech(self):
        if self._audio_user_speaking:
            return
        self._audio_user_speaking = True
        self._audio_buffer = self._pre_speech_buffer
        self._pre_speech_buffer = []

    def _handle_vad_stopped(self):
        """Drop speech that ended without starting a user turn, such as a noise."""
        if not self._audio_user_speaking or self._user_turn_started:
            return
        self._audio_user_speaking = False
        self._audio_buffer = []

    async def _handle_user_stopped_speaking(self):
        self._user_turn_started = False
        if not self._audio_user_speaking:
            return
        self._audio_user_speaking = False
        self._last_user_stopped_at = time.time()
        await self._start_audio_turn()

    async def _start_audio_turn(self):
        """Answer the buffered utterance, letting Omni transcribe it."""
        audio = b"".join(self._audio_buffer)
        self._audio_buffer = []
        self._pre_speech_buffer = []
        min_secs = assert_given(self._settings.min_user_audio_secs)
        if len(audio) < int(self._bytes_per_second() * min_secs):
            logger.debug(f"{self}: ignoring utterance shorter than {min_secs:.2f}s")
            return

        user_stopped_at, self._last_user_stopped_at = self._last_user_stopped_at, None
        # An audio-only pipeline may never send a context frame, and the user is
        # still waiting for an answer.
        context = self._context or LLMContext()
        turn_parts = [
            _audio_part(audio, self._sample_rate, self._num_channels),
            {"type": "text", "text": self._audio_response_instruction()},
        ]
        logger.debug(f"{self}: audio turn of {len(audio) / self._bytes_per_second():.2f}s")
        self._completion_tracker.mark_answered(context)
        task = self._queue_turn(
            lambda: self._run_turn(
                context,
                turn_parts=turn_parts,
                expect_transcript=True,
                metrics_start_time=user_stopped_at,
            ),
            name="nvidia-omni-audio-turn",
        )
        self._untranscribed_speech[task] = audio

    async def _start_text_turn(self, context: LLMContext | None, *, force: bool = False):
        """Complete a context, unless everything in it is already answered.

        Args:
            context: The context to complete.
            force: Whether a completion was requested outright, as with
                ``LLMRunFrame``, rather than by a new user turn.
        """
        if context is None:
            logger.warning(f"{self}: completion requested before any context, ignoring")
            return

        tracker = self._completion_tracker
        if force:
            trigger = "user"
        else:
            trigger = tracker.trigger(context)
            records_transcript = tracker.consume_transcript_record(context)
            if records_transcript and trigger == "user":
                logger.debug(f"{self}: context frame records an answered spoken turn, ignoring")
                return
        if trigger is None:
            return

        turn_parts = self._unwritten_spoken_turn(trigger, context)
        tracker.mark_answered(context)
        self._queue_turn(
            lambda: self._run_turn(context, turn_parts=turn_parts),
            name="nvidia-omni-text-turn",
        )

    def _queue_turn(self, run: Callable[[], Awaitable[None]], *, name: str) -> asyncio.Task:
        """Run a turn once the turns queued before it finish."""
        self._turn_tasks = [task for task in self._turn_tasks if not task.done()]
        self._untranscribed_speech = {
            task: speech for task, speech in self._untranscribed_speech.items() if not task.done()
        }
        previous = self._turn_tasks[-1] if self._turn_tasks else None

        async def run_after_previous():
            if previous:
                with contextlib.suppress(Exception):
                    await previous
            await run()

        task = self.create_task(run_after_previous(), name=name)
        self._turn_tasks.append(task)
        return task

    def _pending_untranscribed_speech(self) -> list[bytes]:
        """The speech of unfinished audio turns that Omni has not transcribed yet."""
        return [
            self._untranscribed_speech[task]
            for task in self._turn_tasks
            if task in self._untranscribed_speech and not task.done()
        ]

    def _unwritten_spoken_turn(
        self, trigger: _CompletionTrigger | None, context: LLMContext
    ) -> list[dict[str, Any]] | None:
        """The transcribed spoken turn a tool follow-up needs, if the context lacks it.

        The user aggregator writes a spoken turn when the assistant response
        starts, which can be after the follow-up a tool result asks for, so the
        follow-up carries the transcript itself.
        """
        transcript = self._last_transcript
        if trigger != "tool" or not transcript or _latest_user_text(context) == transcript:
            return None
        return [{"type": "text", "text": transcript}]

    async def _run_turn(
        self,
        context: LLMContext,
        *,
        turn_parts: Sequence[dict[str, Any]] | None = None,
        expect_transcript: bool = False,
        metrics_start_time: float | None = None,
    ):
        """Run one turn through the inherited completion path.

        Follows the frame, metrics, and error contract of
        ``BaseOpenAILLMService.process_frame()``, adding ``turn_parts`` to the
        request.
        """
        turn_parts_token = self._turn_parts_var.set(list(turn_parts) if turn_parts else None)
        self._transcript_extractor = _TranscriptResponseExtractor() if expect_transcript else None
        self._transcript_emitted = False
        if expect_transcript:
            self._last_transcript = ""

        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics(start_time=metrics_start_time)
        try:
            await self._process_context(context)
        except TIMEOUT_EXCEPTIONS as e:
            await self._call_event_handler("on_completion_timeout")
            await self.push_error(error_msg="LLM completion timeout", exception=e)
        except LLMContextConversionError as e:
            await self.push_error(error_msg=str(e), exception=e)
            context.remove_invalid_file_message()
        except Exception as e:
            removed = (
                isinstance(e, (BadRequestError, UnprocessableEntityError))
                or (isinstance(e, APIStatusError) and e.status_code == 413)
            ) and context.remove_invalid_file_message()
            await self.push_error(
                error_msg=f"Error during completion: {e}",
                exception=e,
                category=ErrorCategory.APPLICATION if removed else None,
            )
        finally:
            self._turn_parts_var.reset(turn_parts_token)
            self._transcript_extractor = None
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

    async def _maybe_push_transcript(self, extractor: "_TranscriptResponseExtractor"):
        if self._transcript_emitted or not (extractor.transcript_done and extractor.transcript):
            return
        self._transcript_emitted = True
        await self._emit_user_transcript(extractor.transcript)

    async def _emit_user_transcript(self, transcript: str):
        """Report the user's spoken turn upstream.

        The user aggregator writes the transcript to the context, as it does for
        an STT service. Subclasses that parse their own response format call
        this with the transcript they extract.

        Args:
            transcript: What the user said.
        """
        self._last_transcript = transcript
        self._completion_tracker.transcript_reported(transcript)
        task = asyncio.current_task()
        if task:
            self._untranscribed_speech.pop(task, None)
        await self.push_frame(
            TranscriptionFrame(
                text=transcript,
                user_id="",
                timestamp=time_now_iso8601(),
                result=transcript,
            ),
            FrameDirection.UPSTREAM,
        )

    def _audio_response_instruction(self) -> str:
        return self._settings.audio_response_instruction or DEFAULT_AUDIO_RESPONSE_INSTRUCTION

    def _bytes_per_second(self) -> int:
        return max(self._sample_rate * self._num_channels * 2, 1)

    def _reset_session_state(self):
        self._audio_buffer = []
        self._pre_speech_buffer = []
        self._audio_user_speaking = False
        self._user_turn_started = False
        self._last_user_stopped_at = None
        self._last_transcript = ""
        self._completion_tracker = _CompletionTracker()

    async def _cancel_turns(self):
        tasks, self._turn_tasks = self._turn_tasks, []
        self._untranscribed_speech = {}
        for task in reversed(tasks):
            await self.cancel_task(task)


def _audio_part(pcm: bytes, sample_rate: int, num_channels: int) -> dict[str, Any]:
    """Build a universal ``input_audio`` content part from 16-bit PCM audio."""
    wav = pcm_to_wav(pcm, sample_rate, num_channels)
    return {
        "type": "input_audio",
        "input_audio": {"data": base64.b64encode(wav).decode("ascii"), "format": "wav"},
    }


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part["text"]
            for part in content
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
        )
    return ""


def _reasoning_text(payload: Any) -> str:
    """Reasoning from a completion message or delta, under either field name NIM uses."""
    for name in ("reasoning_content", "reasoning"):
        value = getattr(payload, name, None)
        if isinstance(value, str) and value:
            return value
    model_extra = getattr(payload, "model_extra", None)
    if isinstance(model_extra, dict):
        for name in ("reasoning_content", "reasoning"):
            value = model_extra.get(name)
            if isinstance(value, str) and value:
                return value
    return ""


def _latest_user_text(context: LLMContext) -> str:
    for message in reversed(context.get_messages()):
        if isinstance(message, dict) and message.get("role") == "user":
            return _message_text(message.get("content")).strip()
    return ""


class _CompletionTracker:
    """Decides which context frames still need a completion.

    Context frames share one context, and some arrive after the turns they
    record have been answered: the user aggregator writes a spoken turn once
    its transcript arrives, and a tool result can be visible in more than one
    context frame. This tracks what has been answered, so that each spoken
    turn and each tool result is completed once.
    """

    def __init__(self):
        self._answered_tool_results: dict[str, Any] = {}
        self._unrecorded_transcript = ""

    def transcript_reported(self, transcript: str):
        """Note a spoken turn that was answered before its context frame arrives.

        Args:
            transcript: What the user said.
        """
        self._unrecorded_transcript = transcript

    def trigger(self, context: LLMContext) -> _CompletionTrigger | None:
        """Why the context needs a completion.

        Only messages after the last assistant message count. A tool result
        needs one until it has been completed, unless a tool is still running.

        Args:
            context: The context of the frame.

        Returns:
            ``"tool"`` for a new tool result, ``"user"`` for a user or developer
            message, or None when nothing is left to answer.
        """
        new_tool_result = False
        tool_running = False
        user_turn = False
        for message in reversed(context.get_messages()):
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "assistant":
                break
            if role == "tool":
                content = message.get("content")
                if content == _TOOL_RESULT_IN_PROGRESS:
                    tool_running = True
                elif not self._is_answered(message):
                    new_tool_result = True
            elif role in ("user", "developer") and message.get("content"):
                user_turn = True
        if new_tool_result and not tool_running:
            return "tool"
        return "user" if user_turn else None

    def consume_transcript_record(self, context: LLMContext) -> bool:
        """Consume the reported spoken turn if the frame records it.

        The reported transcript is forgotten once a frame shows it, so later
        frames that repeat it, such as a history update, are still answered.

        Args:
            context: The context of the frame.

        Returns:
            True when the transcript is the last message of the context.
        """
        transcript = self._unrecorded_transcript
        if not transcript or _latest_user_text(context) != transcript:
            return False
        self._unrecorded_transcript = ""
        messages = [message for message in context.get_messages() if isinstance(message, dict)]
        return messages[-1].get("role") == "user"

    def mark_answered(self, context: LLMContext):
        """Record the tool results a completion of the context answers.

        Args:
            context: The context being completed.
        """
        for message in context.get_messages():
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            content = message.get("content")
            if content != _TOOL_RESULT_IN_PROGRESS:
                self._answered_tool_results[message.get("tool_call_id", "")] = content

    def _is_answered(self, message: LLMStandardMessage) -> bool:
        tool_call_id = message.get("tool_call_id", "")
        answered = self._answered_tool_results
        return tool_call_id in answered and answered[tool_call_id] == message.get("content")


class _ExtractorState(StrEnum):
    DETECTING = "detecting"
    TRANSCRIPT = "transcript"
    BETWEEN = "between"
    RESPONSE = "response"
    PASSTHROUGH = "passthrough"
    DONE = "done"


def _match_tag(text: str, tag: str) -> Literal["match", "partial", "none"]:
    if text.startswith(tag):
        return "match"
    if len(text) < len(tag) and tag.startswith(text):
        return "partial"
    return "none"


def _strip_partial_tag(text: str, tag: str) -> str:
    """Drop the start of ``tag`` from the end of ``text``, left by a stream cut mid-tag."""
    start = text.rfind("<")
    if start != -1 and _match_tag(text[start:], tag) == "partial":
        return text[:start]
    return text


class _TranscriptResponseExtractor:
    """Incrementally splits ``<transcript>`` and ``<response>`` sections.

    ``feed()`` returns response text that is ready to stream onward. The
    transcript is exposed through ``transcript`` once ``transcript_done`` is
    set. Output that doesn't open with ``<transcript>`` is treated entirely as
    response text, so an untagged reply still reaches TTS.
    """

    def __init__(self):
        self._state = _ExtractorState.DETECTING
        self._buffer = ""
        self._pending_transcript = ""
        self.transcript = ""
        self.transcript_done = False

    def feed(self, text: str) -> str:
        if self._state == _ExtractorState.DONE:
            return ""
        self._buffer += text
        out: list[str] = []
        while self._buffer:
            if self._state == _ExtractorState.DETECTING:
                stripped = self._buffer.lstrip()
                if not stripped:
                    break
                match = _match_tag(stripped, _TRANSCRIPT_OPEN)
                if match == "partial":
                    break
                if match == "match":
                    self._buffer = stripped[len(_TRANSCRIPT_OPEN) :]
                    self._state = _ExtractorState.TRANSCRIPT
                else:
                    self._buffer = stripped
                    self._state = _ExtractorState.PASSTHROUGH
            elif self._state == _ExtractorState.TRANSCRIPT:
                idx = self._buffer.find(_TRANSCRIPT_CLOSE)
                if idx == -1:
                    safe_end = len(self._buffer) - len(_TRANSCRIPT_CLOSE) + 1
                    if safe_end > 0:
                        self._pending_transcript += self._buffer[:safe_end]
                        self._buffer = self._buffer[safe_end:]
                    break
                self._pending_transcript += self._buffer[:idx]
                self.transcript = self._pending_transcript.strip()
                self.transcript_done = True
                self._buffer = self._buffer[idx + len(_TRANSCRIPT_CLOSE) :]
                self._state = _ExtractorState.BETWEEN
            elif self._state == _ExtractorState.BETWEEN:
                stripped = self._buffer.lstrip()
                if not stripped:
                    self._buffer = ""
                    break
                match = _match_tag(stripped, _RESPONSE_OPEN)
                if match == "partial":
                    self._buffer = stripped
                    break
                if match == "match":
                    self._buffer = stripped[len(_RESPONSE_OPEN) :]
                    self._state = _ExtractorState.RESPONSE
                else:
                    self._buffer = stripped
                    self._state = _ExtractorState.PASSTHROUGH
            elif self._state == _ExtractorState.PASSTHROUGH:
                out.append(self._buffer)
                self._buffer = ""
            else:
                idx = self._buffer.find(_RESPONSE_CLOSE)
                if idx != -1:
                    out.append(self._buffer[:idx])
                    self._buffer = ""
                    self._state = _ExtractorState.DONE
                    break
                safe_end = len(self._buffer) - len(_RESPONSE_CLOSE) + 1
                if safe_end > 0:
                    out.append(self._buffer[:safe_end])
                    self._buffer = self._buffer[safe_end:]
                break
        return "".join(out)

    def finalize(self) -> str:
        """Return response text still held back once the stream has ended."""
        remainder, self._buffer = self._buffer, ""
        if self._state == _ExtractorState.TRANSCRIPT:
            if not self.transcript_done:
                remainder = _strip_partial_tag(remainder, _TRANSCRIPT_CLOSE)
                self.transcript = (self._pending_transcript + remainder).strip()
                self.transcript_done = bool(self.transcript)
            return ""
        if self._state == _ExtractorState.DONE:
            return ""
        return remainder

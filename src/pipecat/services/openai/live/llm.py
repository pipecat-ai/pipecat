#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Live LLM service: full-duplex speech-to-speech over WebSocket."""

import asyncio
import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from loguru import logger
from openai._types import NotGiven as OpenAINotGiven
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed

from pipecat.adapters.services.open_ai_live_adapter import (
    OpenAILiveLLMAdapter,
    OpenAILiveLLMInvocationParams,
)
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AggregationType,
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMServiceMetadataFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    SpeechOutputAudioRawFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext, LLMStandardMessage
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.openai._constants import OPENAI_SAMPLE_RATE
from pipecat.services.openai.responses.llm import (
    OpenAIResponsesLLMSettings,
    OpenAIResponsesReasoningConfig,
)
from pipecat.services.settings import LLMSettings
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given, is_given
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.llm.backend_llm_worker import (
    BackendOutput,
    _delegate_to_backend,
    _render_transcript_request,
)

from . import events

DEFAULT_MODEL = "gpt-live-1-diamond-alpha"

# Quiet time that ends a speaker's turn. The API emits transcript fragments on
# 200 ms frame boundaries, so this has to clear ordinary gaps within speech.
DEFAULT_TURN_GAP_SECS = 0.8

# The server drains delegation and output work for at most 10 seconds after
# `session.close` before emitting `session.closed`.
SESSION_CLOSE_TIMEOUT_SECS = 10.0

# Each context append takes one text part of at most 500 tokens; this keeps
# a chunk comfortably below that.
MAX_CONTEXT_APPEND_CHARS = 1200


@dataclass
class OpenAILiveLLMSettings(LLMSettings):
    """Settings for OpenAILiveLLMService.

    Parameters:
        voice: Output voice name (for example ``marin`` or ``cedar``). ``None``
            leaves the choice to the API. Cannot be changed once the session
            has started.
        transcript_turn_gap_secs: How long a speaker's transcript must stay
            quiet before their turn is treated as over. The API emits timed
            fragments and no turn boundaries, so turns are grouped here; tune
            this against recordings for the languages and pacing you expect.
    """

    voice: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    transcript_turn_gap_secs: float | NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass
class _TurnGrouper:
    """One speaker's in-progress turn, assembled from timed transcript fragments.

    Parameters:
        gap_secs: Quiet time that ends the turn.
        text: Fragments accumulated so far.
        open: Whether a turn is currently open.
        timer: The task that closes the turn once ``gap_secs`` elapses.
    """

    gap_secs: float
    text: str = ""
    open: bool = False
    timer: asyncio.Task | None = None


@dataclass
class ResponsesDelegation:
    """Responses delegation: OpenAI hosts the backend model the live model delegates to.

    Hosted tools run server-side; function calls are executed by this service
    with the handlers registered for the tools in the pipeline's ``LLMContext``
    (which is also where the backend model's ``tools`` and ``tool_choice``
    come from).

    Parameters:
        settings: Request settings for the backend model — the same object
            :class:`~pipecat.services.openai.responses.llm.OpenAIResponsesLLMService`
            takes. ``model`` is required. ``system_instruction`` becomes the
            backend's ``instructions``, ``max_completion_tokens`` its
            ``max_output_tokens``, ``reasoning`` is sent when configured (the
            server default applies otherwise) and ``extra`` is merged into the
            delegation configuration. Fields the Live API doesn't accept for a
            delegated model (``temperature``, ``top_p``, the penalties, ``seed``,
            ``top_k``, ``max_tokens``, and the user-turn-completion fields) are
            dropped with a warning.
        service_tier: Responses API service tier for delegated requests
            (``auto``, ``default``, ``flex`` or ``priority``).
    """

    settings: OpenAIResponsesLLMSettings
    service_tier: str | None = None


@dataclass
class ClientDelegation:
    """Client delegation: a Pipecat worker is the backend the live model delegates to.

    A delegation names no task: the live model signals only that it is handing
    work over. Each one is sent to the backend as a ``run`` job carrying the
    transcript fragments since the previous delegation, and the backend works
    out the request from them.

    What comes back is appended to the live session according to each output's
    ``prefers_spoken`` flag: what it wants heard goes as commentary, for the
    model to relay in its own words, and the rest as thinking — silent context
    it can draw on if the conversation turns that way.

    Parameters:
        backend: The worker that runs delegated tasks — normally a
            :class:`~pipecat.workers.llm.backend_llm_worker.BackendLLMWorker`
            wrapping any LLM service. The service registers it as a child of
            the pipeline worker at setup, so the pipeline must run under a
            ``WorkerRunner``.
        timeout_secs: How long a delegation may take before it is
            abandoned.
    """

    backend: BaseWorker
    timeout_secs: float = 120.0


class OpenAILiveLLMService(LLMService[OpenAILiveLLMAdapter]):
    """OpenAI Live LLM service: full-duplex speech-to-speech over WebSocket.

    The live model (``gpt-live-1``) listens and speaks at the same time. It
    decides on its own when to answer, when to stop talking when the user
    speaks over it, and when to *delegate* work — search, reasoning, tool use
    — to a backend text model while the conversation continues. In the
    two-layer terms used throughout, the live model is the *frontend* (the
    conversational model) and the delegated-to model is the *backend*. There
    is no client-side turn detection or response triggering: the pipeline
    streams audio in and plays audio out.

    Delegation modes, selected with ``delegation``:

    - :class:`ResponsesDelegation` — OpenAI hosts the backend (Responses API)
      model. Function calls it makes are executed here with the handlers
      registered for the pipeline context's tools; results go back to the API
      as soon as they are available.
    - :class:`ClientDelegation` — a
      :class:`~pipecat.workers.llm.backend_llm_worker.BackendLLMWorker` running
      any Pipecat LLM service is the backend. The model does not share its
      conversation with the backend, so the service ships the transcript turns
      since the previous delegation along with each request.
    - ``None`` — client delegation mode with no backend configured; delegated
      requests are declined.

    Turn frames: the service proposes user turn boundaries from the API's
    projected transcript turns, and the recommended
    ``ExternalUserTurnStrategies(enable_interruptions=False)`` resolve them
    into ``UserStartedSpeakingFrame`` / ``UserStoppedSpeakingFrame`` **without
    broadcasting interruptions**: the model handles being talked over itself,
    and a delegation keeps running when that happens (the model is
    expected to disregard results the conversation has moved past). As a
    consequence every tool behaves as ``cancel_on_interruption=False``; use
    ``cancellable_by_llm=True`` for tools the model should be able to cancel on
    request.

    The context aggregators record both sides of the conversation from the
    transcript frames. Unlike the Realtime services, this one is not flagged
    as a realtime service for the aggregators: a user turn's final transcript
    arrives with its ``turn.done``, so the user aggregator writes each user
    turn to the context as it ends — before any tool calls it triggers —
    rather than when the assistant starts to respond.

    The session starts on the first ``LLMContextFrame`` (typically queued as an
    ``LLMRunFrame``): the context's leading system message (or
    ``Settings.system_instruction``) becomes the model's instructions and the
    remaining text messages seed the session as prior conversation.

    Event handlers available:

    - on_session_started: Called with the session resource once the session is
      ready for audio.
    - on_delegation_created: Called with the delegation item when the model
      delegates work.

    Example::

        llm = OpenAILiveLLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILiveLLMService.Settings(system_instruction="..."),
            delegation=OpenAILiveLLMService.ResponsesDelegation(
                settings=OpenAIResponsesLLMService.Settings(model="gpt-5.4-mini"),
            ),
        )
    """

    Settings = OpenAILiveLLMSettings
    ResponsesDelegation = ResponsesDelegation
    ClientDelegation = ClientDelegation
    _settings: Settings

    adapter_class = OpenAILiveLLMAdapter

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://api.openai.com/v1/live/sessions",
        settings: Settings | None = None,
        delegation: ResponsesDelegation | ClientDelegation | None = None,
        **kwargs,
    ):
        """Initialize the OpenAI Live LLM service.

        Args:
            api_key: OpenAI project API key.
            base_url: WebSocket base URL of the Live API.
            settings: Runtime-updatable settings. ``model``, ``voice`` and
                ``system_instruction`` are fixed once the session has started.
            delegation: Where the model's delegated work runs. ``None`` selects
                client delegation with no backend configured.
            **kwargs: Additional arguments passed to the parent LLMService.
        """
        default_settings = self.Settings(
            model=DEFAULT_MODEL,
            system_instruction=None,
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            voice=None,
            transcript_turn_gap_secs=DEFAULT_TURN_GAP_SECS,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        if isinstance(delegation, ResponsesDelegation):
            backend_model = delegation.settings.model
            if not is_given(backend_model) or not backend_model:
                raise ValueError("ResponsesDelegation.settings.model is required")

        # The live model is fixed for the session's lifetime, so it is
        # resolved once, here.
        session_model = assert_given(default_settings.model)
        if not session_model:
            raise ValueError("settings.model is required")

        super().__init__(settings=default_settings, **kwargs)

        self.api_key = api_key
        self.base_url = base_url
        self._session_model = session_model
        self._delegation = delegation

        self._websocket = None
        self._receive_task = None
        self._disconnecting = False

        self._context: LLMContext | None = None
        self._needs_session_config = True
        self._session_started = False
        self._session_closed_event = asyncio.Event()
        self._sent_tools_snapshot: str | None = None

        self._resampler = create_stream_resampler()
        self._warned_audio_dropped = False

        # Turns assembled from transcript fragments, one per direction: in
        # full duplex the user and the model can be speaking at once.
        gap = assert_given(default_settings.transcript_turn_gap_secs)
        self._user_turn = _TurnGrouper(gap_secs=gap)
        self._assistant_turn = _TurnGrouper(gap_secs=gap)

        # A trailing developer message asking the model to open the conversation.
        self._opening_instruction: str | None = None

        # Responses delegation: calls awaiting an output, by call id, and the
        # responses they belong to, by response id.
        self._open_function_calls: dict[str, str] = {}
        self._pending_responses: dict[str, _PendingResponse] = {}

        # Client delegation: the conversation not yet sent to the backend, the
        # delegations in flight by id, and whether the backend has run before.
        self._transcript_fragments: list[dict[str, str]] = []
        self._delegated_before = False
        self._delegation_tasks: dict[str, asyncio.Task] = {}

        self._register_event_handler("on_session_started")
        self._register_event_handler("on_delegation_created")

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate usage metrics.

        Returns:
            True if metrics generation is supported.
        """
        return True

    def service_metadata_frame(self) -> LLMServiceMetadataFrame:
        """Recommend external turn strategies, resolved from the API's projected turns.

        ``is_realtime_service`` is left off on purpose: the aggregators'
        realtime mode defers the user-message write until the assistant
        responds, to absorb late transcripts, and here that would place a
        delegation's tool calls ahead of the user message that caused them.
        """
        return LLMServiceMetadataFrame(
            service_name=self.name,
            user_turn_strategies=ExternalUserTurnStrategies(enable_interruptions=False),
        )

    #
    # lifecycle
    #

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service, register the client-delegation backend, and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        if isinstance(self._delegation, ClientDelegation):
            # A child of the pipeline worker: ended and cancelled with it.
            await self.pipeline_worker.add_workers(self._delegation.backend)
        await self._connect()

    async def cleanup(self):
        """Release resources at teardown."""
        await super().cleanup()
        await self._disconnect()

    async def stop(self, frame: EndFrame):
        """Close the session gracefully and disconnect.

        Args:
            frame: The end frame triggering service shutdown.
        """
        await super().stop(frame)
        await self._close_open_turns()
        await self._close_session()
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Disconnect immediately.

        Args:
            frame: The cancel frame triggering service cancellation.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _update_settings(self, delta):
        """Apply a settings delta; the Live API fixes all of them at session start."""
        changed = await super()._update_settings(delta)
        if changed and self._session_started:
            logger.warning(
                f"{self}: settings {sorted(changed)} cannot be changed after the session has "
                "started; they take effect on the next session"
            )
        return changed

    async def reset_conversation(self):
        """Start a new session seeded from the current context.

        The Live API only takes conversation history at session start, so
        replacing the context (for example to restore a saved conversation)
        means dropping the session and opening a new one configured from the
        context as it is now. The old session is abandoned rather than closed
        gracefully: a graceful close waits for its in-flight delegations to
        drain, and their results belong to the conversation being replaced.
        Must not be called from the receive task.
        """
        logger.debug(f"{self}: resetting conversation")
        # Close out turns the old session was in the middle of, so the
        # aggregator records what was said and the response frames stay
        # balanced for the new session's turns.
        await self._close_open_turns()
        await self._disconnect()
        self._needs_session_config = True
        await self._connect()
        if self._context is not None:
            await self._send_session_config()

    #
    # frame processing
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames from the pipeline.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            await self._handle_context(frame.context)
        elif isinstance(frame, InputAudioRawFrame):
            await self._send_user_audio(frame)
        elif isinstance(frame, LLMSetToolsFrame):
            # Continuous session: no fresh context frame per turn, so sync the
            # registered tool handlers to the new tool set here (the base service
            # only does this on LLMContextFrame).
            self._sync_registered_tool_handlers(frame.tools)
            await self._maybe_send_tools_update()

        await self.push_frame(frame, direction)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame, delivering function call outcomes to the API on the way.

        Function call results and cancellations are broadcast by the base
        service; the downstream copy is observed here and answered to the
        backend model immediately. The frames continue to the assistant
        aggregator, which records them in the context.

        Args:
            frame: The frame to push.
            direction: The direction of frame pushing.
        """
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, FunctionCallResultFrame):
                await self._handle_function_call_result(frame)
            elif isinstance(frame, FunctionCallCancelFrame):
                await self._handle_function_call_cancel(frame)
        await super().push_frame(frame, direction)

    async def _handle_context(self, context: LLMContext):
        self._context = context
        if self._needs_session_config:
            await self._send_session_config()
            return

        # A later context carries messages the aggregators recorded from this
        # session's own transcripts, tool results (delivered to the API as
        # they are produced, see push_frame), or messages the app appended.
        # The app's additions are not forwarded to the API: user and
        # assistant messages can't be told apart from the transcript
        # recordings. Newly appended system/developer messages can only come
        # from the app (the aggregators add user, assistant and tool messages,
        # and async-tool envelopes that async_tool_messages.parse_message
        # identifies), so a future diff could forward those as
        # session.context.append(channel="developer") unambiguously.
        await self._maybe_send_tools_update()

    #
    # session configuration
    #

    def _invocation_params(self) -> OpenAILiveLLMInvocationParams:
        assert self._context is not None
        return self.get_llm_adapter().get_llm_invocation_params(
            self._context,
            system_instruction=assert_given(self._settings.system_instruction),
        )

    async def _send_session_config(self):
        """Send ``session.start``, the first message on the socket."""
        params = self._invocation_params()
        history = params["input"]
        # A trailing developer message is the app asking the bot to open the
        # conversation. The startup history is not the place for it: it is
        # delivered as commentary once the session starts, which is how
        # the API asks the model to speak first.
        self._opening_instruction = _trailing_developer_text(history)
        if self._opening_instruction:
            history = history[:-1]

        voice = assert_given(self._settings.voice)
        session = events.SessionConfig(
            model=self._session_model,
            instructions=params["instructions"],
            audio=(
                events.AudioConfig(output=events.AudioOutputConfig(voice=voice)) if voice else None
            ),
            delegation=self._delegation_config(params["tools"], params["tool_choice"]),
            input=history or None,
        )
        self._sent_tools_snapshot = self._tools_snapshot(params)
        self._needs_session_config = False
        logger.debug(
            f"{self}: starting session with {len(history)} prior messages "
            f"and {session.delegation.type if session.delegation else 'client'} delegation"
        )
        await self.send_client_event(events.SessionStartEvent(session=session))

    def _delegation_config(
        self, tools: list[dict[str, Any]], tool_choice: Any | None
    ) -> events.ClientDelegationConfig | events.ResponsesDelegationConfig:
        if isinstance(self._delegation, ResponsesDelegation):
            responses = _responses_delegation_config(self._delegation)
            if tools:
                responses["tools"] = tools
            if tool_choice is not None:
                responses["tool_choice"] = tool_choice
            return events.ResponsesDelegationConfig(responses=responses)
        return events.ClientDelegationConfig()

    async def _maybe_send_tools_update(self):
        """Send a sparse ``session.update`` when the backend model's tools changed."""
        if not (
            self._session_started
            and self._context is not None
            and isinstance(self._delegation, ResponsesDelegation)
        ):
            return
        params = self._invocation_params()
        snapshot = self._tools_snapshot(params)
        if snapshot == self._sent_tools_snapshot:
            return
        self._sent_tools_snapshot = snapshot
        responses: dict[str, Any] = {"tools": params["tools"]}
        if params["tool_choice"] is not None:
            responses["tool_choice"] = params["tool_choice"]
        await self.send_client_event(
            events.SessionUpdateEvent(
                session=events.SessionUpdateConfig(
                    delegation=events.ResponsesDelegationConfig(responses=responses)
                )
            )
        )

    @staticmethod
    def _tools_snapshot(params: OpenAILiveLLMInvocationParams) -> str:
        return json.dumps(
            {"tools": params["tools"], "tool_choice": params["tool_choice"]},
            sort_keys=True,
            default=str,
        )

    #
    # websocket communication
    #

    async def send_client_event(self, event: events.ClientEvent):
        """Send a client event to the Live API.

        Args:
            event: The client event to send.
        """
        await self._ws_send(event.to_payload())

    async def _connect(self):
        try:
            if self._websocket:
                return
            self._websocket = await websocket_connect(
                uri=self.base_url,
                additional_headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Alpha": events.OPENAI_LIVE_ALPHA_HEADER,
                },
            )
            self._receive_task = self.create_task(self._receive_task_handler())
        except Exception as e:
            self._websocket = None
            await self.push_error(
                error_msg=f"Error connecting: {e}", exception=e, force_treat_as_permanent=True
            )

    async def _disconnect(self):
        # `_disconnecting` gates every outgoing event, so it is cleared in a
        # finally: a teardown that fails part-way still leaves the service able
        # to speak on its next session.
        try:
            self._disconnecting = True
            self._session_started = False
            await self.stop_all_metrics()
            if self._websocket:
                await self._websocket.close()
                self._websocket = None
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=1.0)
                self._receive_task = None
            for task in list(self._delegation_tasks.values()):
                await self.cancel_task(task)
            self._delegation_tasks.clear()
            self._transcript_fragments.clear()
            self._delegated_before = False
            self._sent_tools_snapshot = None
            self._open_function_calls.clear()
            self._pending_responses.clear()
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting: {e}", exception=e)
        finally:
            self._disconnecting = False

    async def _close_session(self):
        """Ask the server to shut down gracefully and wait for ``session.closed``."""
        if not self._websocket or not self._session_started:
            return
        self._session_closed_event.clear()
        await self.send_client_event(events.SessionCloseEvent())
        try:
            await asyncio.wait_for(
                self._session_closed_event.wait(), timeout=SESSION_CLOSE_TIMEOUT_SECS
            )
        except TimeoutError:
            logger.warning(f"{self}: timed out waiting for session.closed")

    async def _ws_send(self, message: dict[str, Any]):
        try:
            if not self._disconnecting and self._websocket:
                await self._websocket.send(json.dumps(message))
        except Exception as e:
            if self._disconnecting or not self._websocket:
                return
            await self.push_error(
                error_msg=f"Error sending client event: {e}",
                exception=e,
                force_treat_as_permanent=True,
            )

    #
    # inbound server event handling
    #

    async def _receive_task_handler(self):
        assert self._websocket is not None
        try:
            async for message in self._websocket:
                try:
                    evt = events.parse_server_event(message)
                except ValueError as e:
                    logger.warning(f"{self}: ignoring unparseable server event: {e}")
                    continue
                await self._handle_server_event(evt)
        except ConnectionClosed as e:
            if not self._disconnecting:
                await self.push_error(
                    error_msg=f"Connection closed: {e}",
                    exception=e,
                    force_treat_as_permanent=True,
                )

    async def _handle_server_event(self, evt: events.ServerEvent):
        if isinstance(evt, events.SessionStartedEvent):
            await self._handle_evt_session_started(evt)
        elif isinstance(evt, events.OutputAudioDeltaEvent):
            await self._handle_evt_audio_delta(evt)
        elif isinstance(evt, events.TranscriptDeltaEvent):
            await self._handle_evt_transcript_delta(evt)
        elif isinstance(evt, events.SessionDelegationCreatedEvent):
            await self._handle_evt_delegation_created(evt)
        elif isinstance(evt, events.ResponseEventEnvelope):
            await self._handle_evt_response(evt)
        elif isinstance(evt, events.SessionUsageUpdatedEvent):
            await self._report_usage(evt.usage)
        elif isinstance(evt, events.SessionClosedEvent):
            await self._handle_evt_session_closed(evt)
        elif isinstance(evt, events.ErrorEvent):
            await self._handle_evt_error(evt)
        elif isinstance(evt, events.SessionUpdatedEvent):
            logger.debug(f"{self}: session updated")
        elif isinstance(evt, events.UnknownServerEvent):
            logger.debug(f"{self}: ignoring unknown server event {evt.type}")
        else:
            logger.debug(f"{self}: {evt.type}")

    async def _handle_evt_session_started(self, evt: events.SessionStartedEvent):
        self._session_started = True
        logger.info(
            f"{self}: session {evt.session.id} started. The live model handles being "
            "interrupted by itself; no InterruptionFrame is broadcast, so tools run to "
            "completion regardless of cancel_on_interruption."
        )
        await self._call_event_handler("on_session_started", evt.session)
        if self._opening_instruction:
            # Speakable context is how the API asks the model to open the
            # conversation; the model paraphrases rather than reads it out.
            instruction, self._opening_instruction = self._opening_instruction, None
            await self._send_context_append(None, instruction, spoken=True)

    async def _handle_evt_session_closed(self, evt: events.SessionClosedEvent):
        logger.debug(f"{self}: session closed ({evt.reason})")
        if evt.usage:
            await self._report_usage(evt.usage)
        self._session_started = False
        self._session_closed_event.set()

    async def _handle_evt_error(self, evt: events.ErrorEvent):
        error = evt.error
        details = f"{error.type or 'error'}/{error.code or 'unknown'}: {error.message}"
        if error.param:
            details += f" (param: {error.param})"
        if not self._session_started:
            # A startup error means no session will start.
            await self.push_error(
                error_msg=f"Session startup failed: {details}", force_treat_as_permanent=True
            )
        else:
            await self.push_error(error_msg=details)

    #
    # audio
    #

    async def _send_user_audio(self, frame: InputAudioRawFrame):
        if not self._session_started:
            if not self._warned_audio_dropped:
                self._warned_audio_dropped = True
                logger.debug(f"{self}: dropping input audio until the session has started")
            return
        audio = frame.audio
        if frame.sample_rate != OPENAI_SAMPLE_RATE:
            audio = await self._resampler.resample(audio, frame.sample_rate, OPENAI_SAMPLE_RATE)
        if not audio:
            return
        payload = base64.b64encode(audio).decode("utf-8")
        await self.send_client_event(events.InputAudioAppendEvent(audio=payload))

    async def _handle_evt_audio_delta(self, evt: events.OutputAudioDeltaEvent):
        # The model streams output continuously at real-time pace, silence
        # included, so this is a speech stream rather than TTS output: the
        # output transport derives BotStarted/StoppedSpeakingFrame from the
        # audio itself instead of from TTSStarted/StoppedFrame.
        await self.push_frame(
            SpeechOutputAudioRawFrame(
                audio=base64.b64decode(evt.delta),
                sample_rate=OPENAI_SAMPLE_RATE,
                num_channels=1,
            )
        )

    #
    # transcript turns
    #

    async def _handle_evt_transcript_delta(self, evt: events.TranscriptDeltaEvent):
        if not evt.delta:
            return
        role = evt.role
        # The delegation ledger keeps fragments as they arrive: the backend is
        # better served by the recent role-labelled sequence than by waiting
        # for a turn that may never be declared final.
        self._remember_fragment(role, evt.delta)

        turn = self._user_turn if role == "user" else self._assistant_turn
        if not turn.open:
            turn.open = True
            turn.text = ""
            await self._open_turn(role)
        turn.text += evt.delta
        await self._append_turn(role, evt.delta, turn.text, evt)
        await self._restart_turn_timer(role, turn)

    async def _restart_turn_timer(self, role: str, turn: "_TurnGrouper"):
        if turn.timer is not None:
            await self.cancel_task(turn.timer)
        turn.timer = self.create_task(self._close_turn_after_gap(role, turn), f"turn-gap:{role}")

    async def _close_turn_after_gap(self, role: str, turn: "_TurnGrouper"):
        await asyncio.sleep(turn.gap_secs)
        # Running inside the turn's own timer, so the timer must not cancel itself.
        turn.timer = None
        await self._close_turn(role)

    async def _close_turn(self, role: str):
        """Close an open turn, if any, emitting its end frames."""
        turn = self._user_turn if role == "user" else self._assistant_turn
        if turn.timer is not None:
            await self.cancel_task(turn.timer)
            turn.timer = None
        if not turn.open:
            return
        turn.open = False
        text, turn.text = turn.text, ""
        if role == "user":
            if text.strip():
                await self.push_frame(
                    TranscriptionFrame(text, "", time_now_iso8601()),
                    FrameDirection.UPSTREAM,
                )
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)
        else:
            await self.push_frame(TTSStoppedFrame())
            await self.push_frame(LLMFullResponseEndFrame())

    async def _close_open_turns(self):
        """Close both directions, so response frames stay balanced across sessions."""
        await self._close_turn("user")
        await self._close_turn("assistant")

    async def _open_turn(self, role: str):
        if role == "user":
            await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
        else:
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(TTSStartedFrame())

    async def _append_turn(self, role: str, delta: str, accumulated: str, evt: events.ServerEvent):
        if role == "user":
            await self._push_interim_transcription(accumulated, evt)
        else:
            await self._push_assistant_text(delta)

    def _remember_fragment(self, role: Literal["user", "assistant"], delta: str):
        """Keep a transcript fragment for the next client delegation.

        Fragments land on frame boundaries, so consecutive ones from the same
        speaker are joined back up as they arrive, giving the backend whole
        utterances to read. Deltas carry their own leading spaces, so they
        concatenate into the original text unaltered.
        """
        if not isinstance(self._delegation, ClientDelegation):
            return
        fragments = self._transcript_fragments
        if fragments and fragments[-1]["role"] == role:
            fragments[-1]["content"] += delta
        elif delta.strip():
            fragments.append({"role": role, "content": delta.lstrip()})

    def _take_transcript(self) -> list[LLMStandardMessage]:
        """Take the conversation recorded since the previous delegation."""
        taken, self._transcript_fragments = self._transcript_fragments, []
        return [
            cast(LLMStandardMessage, {"role": f["role"], "content": f["content"].strip()})
            for f in taken
            if f["content"].strip()
        ]

    async def _push_assistant_text(self, text: str):
        if not text:
            return
        # RTVI relies on LLMTextFrames for its "bot-llm-text" event, but the
        # assistant aggregator records the TTSTextFrame, so the LLMTextFrame
        # must not also be appended to the context.
        llm_text_frame = LLMTextFrame(text)
        llm_text_frame.append_to_context = False
        await self.push_frame(llm_text_frame)

        tts_text_frame = TTSTextFrame(text, aggregated_by=AggregationType.SENTENCE)
        tts_text_frame.includes_inter_frame_spaces = True
        await self.push_frame(tts_text_frame)

    async def _push_interim_transcription(self, transcript: str, evt: events.ServerEvent):
        if not transcript.strip():
            return
        await self.push_frame(
            InterimTranscriptionFrame(transcript, "", time_now_iso8601(), result=evt),
            FrameDirection.UPSTREAM,
        )

    #
    # delegation
    #

    async def _handle_evt_delegation_created(self, evt: events.SessionDelegationCreatedEvent):
        delegation = evt.delegation
        logger.debug(f"{self}: delegation {delegation.id} created (target={delegation.target})")
        await self._call_event_handler("on_delegation_created", delegation)
        if delegation.target == "client":
            await self._handle_client_delegation(delegation)

    async def _handle_client_delegation(self, delegation: events.DelegationMetadata):
        if not isinstance(self._delegation, ClientDelegation):
            logger.warning(
                f"{self}: the model delegated work but no backend is configured to handle "
                "client delegations; declining"
            )
            await self._send_context_append(
                delegation.id,
                "No backend is available to handle delegated work in this session.",
                spoken=True,
            )
            return
        task = self.create_task(
            self._run_client_delegation(delegation), f"delegation:{delegation.id}"
        )
        self._delegation_tasks[delegation.id] = task
        task.add_done_callback(lambda _: self._delegation_tasks.pop(delegation.id, None))

    async def _run_client_delegation(self, delegation: events.DelegationMetadata):
        config = self._delegation
        assert isinstance(config, ClientDelegation)
        # The backend is handed the conversation since the last delegation
        # and works out the request from it.
        request = _render_transcript_request(
            self._take_transcript(), first=not self._delegated_before
        )
        self._delegated_before = True

        async def on_update(output: BackendOutput):
            await self._send_context_append(
                delegation.id, output.text, spoken=output.prefers_spoken
            )

        try:
            # Every output, the final answer included, arrives through
            # on_update, so the job's return value is not needed here.
            await _delegate_to_backend(
                self.pipeline_worker,
                config.backend.name,
                request=request,
                on_update=on_update,
                timeout_secs=config.timeout_secs,
            )
        except Exception as e:
            logger.warning(f"{self}: delegation {delegation.id} failed: {e}")
            await self._send_context_append(
                delegation.id,
                f"The delegated task could not be completed: {e}",
                spoken=True,
            )
            await self.push_error(error_msg=f"Delegation {delegation.id} failed: {e}", exception=e)

    async def _send_context_append(self, delegation_id: str | None, text: str, *, spoken: bool):
        """Append text to the live session, for it to speak or to keep to itself.

        The API's two channels are named for what the live model does with the
        text: commentary is paraphrased aloud, thinking joins its private
        reasoning. That vantage is the model's own, which is why what arrives
        from a backend is flagged ``prefers_spoken`` instead — ordinary prose can
        go unspoken without being anything like a thought.
        """
        channel = "commentary" if spoken else "thinking"
        for chunk in _chunk_text(text, MAX_CONTEXT_APPEND_CHARS):
            logger.debug(f"{self}: delegation {delegation_id} {channel} context: {chunk!r}")
            event_class = (
                events.SessionCommentaryAppendEvent if spoken else events.SessionThinkingAppendEvent
            )
            await self.send_client_event(event_class(delegation_id=delegation_id, content=chunk))

    #
    # Responses delegation
    #

    async def _handle_evt_response(self, evt: events.ResponseEventEnvelope):
        """Dispatch a wrapped Responses lifecycle event by its own type."""
        inner_type = evt.inner_type
        if inner_type == "response.created":
            self._track_response(evt)
        elif inner_type == "response.output_item.done":
            await self._handle_response_output_item_done(evt)
        elif inner_type in ("response.completed", "response.incomplete", "response.failed"):
            await self._handle_response_finished(evt)
        else:
            logger.trace(f"{self}: {inner_type or evt.type}")

    def _track_response(self, evt: events.ResponseEventEnvelope) -> str:
        """Remember a delegated response so its function calls can be collected.

        Correlation runs on the envelope's delegation, which every wrapped
        event carries; the individual Responses events do not all name their
        response.
        """
        key = _correlation_key(evt)
        if key and key not in self._pending_responses:
            self._pending_responses[key] = _PendingResponse(delegation_id=evt.delegation_id)
        return key

    async def _handle_response_output_item_done(self, evt: events.ResponseEventEnvelope):
        item = events.ResponseOutputItem.model_validate(evt.event.get("item") or {})
        if item.type != "function_call":
            return
        if item.status != "completed":
            logger.debug(f"{self}: ignoring {item.status} function call item")
            return
        if not item.call_id or not item.name:
            logger.warning(f"{self}: function call item without call_id or name: {item}")
            return
        if item.call_id in self._open_function_calls:
            logger.warning(f"{self}: function call {item.call_id} already in progress, skipping")
            return

        try:
            arguments = json.loads(item.arguments) if item.arguments else {}
        except json.JSONDecodeError as e:
            await self.push_error(
                error_msg=f"Invalid arguments for function call {item.name}: {e}", exception=e
            )
            return

        key = self._track_response(evt)
        pending = self._pending_responses.get(key)
        if pending is not None:
            pending.call_ids.add(item.call_id)
            pending.had_calls = True
        self._open_function_calls[item.call_id] = key
        await self.run_function_calls(
            [
                FunctionCallFromLLM(
                    context=self._context,
                    tool_call_id=item.call_id,
                    function_name=item.name,
                    arguments=arguments,
                )
            ]
        )

    async def _handle_response_finished(self, evt: events.ResponseEventEnvelope):
        """Note that a response emitted all of its output items.

        The lifecycle snapshot's ``output`` is empty even here, so the calls
        collected from the individual item events are what has to be answered.
        """
        if evt.inner_type == "response.completed":
            await self._report_backend_usage(evt.event.get("response") or {})
        elif evt.inner_type in ("response.failed", "response.incomplete"):
            response = evt.event.get("response") or {}
            error = response.get("error") or response.get("incomplete_details") or {}
            message = (
                error.get("message") or error.get("reason") if isinstance(error, dict) else None
            )
            await self.push_error(
                error_msg=f"Delegated response {evt.inner_type.removeprefix('response.')}: "
                f"{message or response.get('status', 'unknown')}"
            )
        key = _correlation_key(evt)
        pending = self._pending_responses.get(key)
        if pending is None:
            return
        pending.finished = True
        await self._maybe_continue_response(key)

    async def _maybe_continue_response(self, key: str):
        """Continue a response once every function call it made has been answered."""
        pending = self._pending_responses.get(key)
        if pending is None or not pending.finished or not pending.had_calls or pending.call_ids:
            return
        del self._pending_responses[key]
        logger.debug(f"{self}: continuing delegated work for {key}")
        await self.send_client_event(events.ResponseCreateEvent())

    async def _handle_function_call_result(self, frame: FunctionCallResultFrame):
        if frame.tool_call_id not in self._open_function_calls:
            return
        is_final = frame.properties.is_final if frame.properties else True
        if not is_final:
            logger.warning(
                f"{self}: the Live API accepts one output per function call; dropping "
                f"intermediate result for {frame.function_name}:{frame.tool_call_id}"
            )
            return
        # Same encoding the assistant aggregator records in the context.
        output = json.dumps(frame.result, ensure_ascii=False) if frame.result else "COMPLETED"
        await self._send_function_call_output(frame.tool_call_id, output)

    async def _handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        if frame.tool_call_id not in self._open_function_calls:
            return
        await self._send_function_call_output(
            frame.tool_call_id,
            json.dumps({"error": "The function call was cancelled before it produced a result."}),
        )

    async def _send_function_call_output(self, call_id: str, output: str):
        """Queue one function result, and continue the response once all are in."""
        key = self._open_function_calls.pop(call_id, "")
        logger.debug(f"{self}: sending function call output for {call_id}")
        await self.send_client_event(
            events.ResponseItemCreateEvent(
                item=events.FunctionCallOutputItem(call_id=call_id, output=output)
            )
        )
        pending = self._pending_responses.get(key)
        if pending is not None:
            pending.call_ids.discard(call_id)
        await self._maybe_continue_response(key)

    #
    # usage metrics
    #

    async def _report_usage(self, usage: events.Usage):
        """Log the session's cumulative live audio duration.

        The live model's usage is reported as seconds rather than tokens.
        Token metrics come from the backend model instead, on the nested
        Responses completion events.
        """
        if usage.seconds is not None:
            logger.debug(f"{self}: live audio usage: {usage.seconds:.1f}s (cumulative)")

    async def _report_backend_usage(self, response: dict[str, Any]):
        """Report the backend model's token usage from a completed delegated response."""
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return
        details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        tokens = LLMTokenUsage(
            prompt_tokens=usage.get("input_tokens") or 0,
            completion_tokens=usage.get("output_tokens") or 0,
            total_tokens=usage.get("total_tokens") or 0,
            cache_read_input_tokens=details.get("cached_tokens") or 0,
            reasoning_tokens=output_details.get("reasoning_tokens") or 0,
        )
        if tokens.total_tokens > 0:
            await self.start_llm_usage_metrics(tokens)


@dataclass
class _PendingResponse:
    """A delegated Responses run whose function calls we are collecting.

    Parameters:
        delegation_id: The delegation the response belongs to.
        call_ids: Calls still awaiting an output.
        had_calls: Whether the response asked for any function call at all.
        finished: Whether the response emitted all of its output items.
    """

    delegation_id: str | None = None
    call_ids: set[str] = field(default_factory=set)
    had_calls: bool = False
    finished: bool = False


def _correlation_key(evt: events.ResponseEventEnvelope) -> str:
    """Identify the delegated work a wrapped Responses event belongs to.

    The envelope's ``delegation_id`` is the correlator the API guarantees; the
    response id is a fallback for an envelope that arrives without one.
    """
    if evt.delegation_id:
        return evt.delegation_id
    response = evt.event.get("response")
    if isinstance(response, dict) and isinstance(response.get("id"), str):
        return response["id"]
    response_id = evt.event.get("response_id")
    return response_id if isinstance(response_id, str) else ""


def _trailing_developer_text(items: list[events.InputItem]) -> str | None:
    """Return the text of a trailing developer message, if the history ends with one."""
    if not items or items[-1].role != "developer":
        return None
    return "\n".join(part.text for part in items[-1].content if part.text).strip() or None


def _responses_delegation_config(delegation: ResponsesDelegation) -> dict[str, Any]:
    """Build ``delegation.responses`` from the nested Responses settings.

    Mirrors how ``OpenAIResponsesLLMService`` builds a request from the same
    settings, minus what the Live API owns for a delegated model.
    """
    settings = delegation.settings

    def is_set(value: Any) -> bool:
        return is_given(value) and value is not None and not isinstance(value, OpenAINotGiven)

    config: dict[str, Any] = {"model": settings.model}
    if is_set(settings.system_instruction):
        config["instructions"] = settings.system_instruction
    if isinstance(settings.max_completion_tokens, int):
        config["max_output_tokens"] = settings.max_completion_tokens
    if isinstance(settings.reasoning, OpenAIResponsesReasoningConfig):
        config["reasoning"] = settings.reasoning.model_dump(exclude_none=True)
    if delegation.service_tier is not None:
        config["service_tier"] = delegation.service_tier

    unsupported = [
        name
        for name in (
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "top_k",
            "max_tokens",
            "user_turn_completion_config",
        )
        if is_set(getattr(settings, name))
    ]
    if is_set(settings.filter_incomplete_user_turns) and settings.filter_incomplete_user_turns:
        unsupported.append("filter_incomplete_user_turns")
    if unsupported:
        logger.warning(
            f"Dropping Responses settings the OpenAI Live API doesn't accept for a delegated "
            f"model: {unsupported}"
        )

    config.update(settings.extra)
    return config


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def _chunk_text(text: str, limit: int) -> list[str]:
    """Split text into chunks of at most ``limit`` characters, preferring sentence boundaries."""
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []

    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for piece in _SENTENCE_BOUNDARY.split(text):
        piece = piece.strip()
        while len(piece) > limit:
            # An overlong sentence: cut at the last space before the limit.
            cut = piece.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            flush()
            chunks.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if not piece:
            continue
        if current and len(current) + 1 + len(piece) > limit:
            flush()
        current = f"{current} {piece}".strip()
    flush()
    return chunks

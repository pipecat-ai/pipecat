#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Rime Prism speech-to-speech conversations through the direct SDK session API."""

import asyncio
import copy
import json
import math
from collections import OrderedDict
from collections.abc import Coroutine
from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np
from loguru import logger

try:
    from rimelabs_sdk import Rime, RimeInputError
    from rimelabs_sdk import realtime as r
except ModuleNotFoundError as error:
    logger.error('Install Rime realtime support with `uv pip install "pipecat-ai[rime-realtime]"`.')
    raise Exception(f"Missing module: {error}") from error

from pipecat.adapters.llm.services.rime_realtime_adapter import RimeRealtimeLLMAdapter
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AudioOutputEndFrame,
    AudioOutputGainFrame,
    AudioOutputStartFrame,
    AudioOutputStopFrame,
    AudioPlaybackReportFrame,
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    InputImageRawFrame,
    InterimTranscriptionFrame,
    InterruptionCause,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMInputTextFrame,
    LLMMessagesAppendFrame,
    LLMMessagesTransformFrame,
    LLMMessagesUpdateFrame,
    LLMOperationOutcome,
    LLMOperationResultFrame,
    LLMRunFrame,
    LLMServiceMetadataFrame,
    LLMSetToolChoiceFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    ServiceSwitcherDeactivatedFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators._llm_operation import (
    _LLMContextBindFrame,
    _LLMContextOperationFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService
from pipecat.services.llm_service import FunctionCallRunnerItem, LLMService
from pipecat.services.rime.realtime._history import History, PartialHistoryError
from pipecat.services.rime.realtime._playback import Output, Playback, Response
from pipecat.services.settings import LLMSettings
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.text.base_text_aggregator import TextType
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given, is_given


@dataclass
class RimeRealtimeSettings(LLMSettings):
    """Acknowledged Prism settings, also used for sparse frame deltas.

    Parameters:
        tools: Advertised tools. An empty list removes all tools.
        voice: Prism voice.
        interrupt_on_speech: Interruption policy for subsequent replies.
        tool_fillers: SDK fillers keyed by the resulting tool names.
        truncation: Future model history capacity policy.
        response_metadata: Labels applied through a separate SDK operation.
    """

    tools: Any = field(default_factory=lambda: NOT_GIVEN)
    voice: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    interrupt_on_speech: bool | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    tool_fillers: dict[str, r.PrismFiller] | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    truncation: r.Truncation | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    response_metadata: dict[str, str] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class RimeRealtimeLLMService(LLMService[RimeRealtimeLLMAdapter]):
    """One Prism conversation with frame-ordered input and accepted mutations.

    Use a separate instance for each concurrent conversation. An injected client
    remains caller-owned. There is no automatic replay or reconnect. Use a fresh
    service with completed context for explicit recovery.

    Events:
        on_session_updated: Accepted SDK session information.
        on_operation_result: One result frame per operation and SDK operation kind.
        on_rime_event: SDK events with provider correlation and diagnostic fields.
    """

    Settings = RimeRealtimeSettings
    _reject_intermediate_results = True
    _settings: RimeRealtimeSettings

    def __init__(
        self,
        *,
        client: Rime | None = None,
        endpoint: str | None = None,
        settings: Settings | None = None,
        turn_detection: r.TurnDetection | None = None,
        timeouts: r.RealtimeTimeouts | None = None,
        request_id: str | None = None,
        speech_hint_gain: float = 0.25,
        speech_hint_timeout: float = 0.6,
        max_pending_operations: int = 256,
        max_input_audio_bytes: int = 1920000,
        max_output_audio_bytes: int = 16000000,
        shutdown_timeout: float = 5.0,
        **kwargs,
    ):
        """Configure connection ownership, limits, and opening turn detection.

        Args:
            client: Optional caller-owned SDK client.
            endpoint: Optional SDK-validated Prism WebSocket endpoint.
            settings: Initial settings and advertised tools.
            turn_detection: Opening-only SDK turn timing settings.
            timeouts: SDK connection, readiness, and acknowledgment limits.
            request_id: Requested provider correlation ID.
            speech_hint_gain: Temporary output multiplier on a speech hint.
            speech_hint_timeout: Hint expiry in seconds.
            max_pending_operations: Bound for each pending operation queue.
            max_input_audio_bytes: Bound for buffered microphone PCM.
            max_output_audio_bytes: Bound for emitted PCM awaiting playback reports.
            shutdown_timeout: Maximum time for graceful output and request drain.
            **kwargs: Arguments for LLMService.
        """
        if not math.isfinite(speech_hint_gain) or not 0 <= speech_hint_gain <= 1:
            raise ValueError("speech_hint_gain must be between zero and one")
        if not math.isfinite(speech_hint_timeout) or not 0 < speech_hint_timeout <= 60:
            raise ValueError("speech_hint_timeout must be above zero and at most 60 seconds")
        if max_pending_operations < 1 or max_input_audio_bytes < 192000:
            raise ValueError("Prism queue limits must be positive and hold one SDK audio chunk")
        if not math.isfinite(shutdown_timeout) or shutdown_timeout <= 0:
            raise ValueError("shutdown_timeout must be finite and positive")
        defaults: dict[str, Any] = {f.name: None for f in fields(LLMSettings) if f.name != "extra"}
        defaults.update(
            model="prism", filter_incomplete_user_turns=False, use_provider_thinking_default=False
        )
        store = self.Settings(
            **defaults,
            tools=None,
            voice=None,
            interrupt_on_speech=True,
            tool_fillers={},
            truncation=None,
            response_metadata=None,
        )
        if settings is not None:
            store.apply_update(copy.deepcopy(settings))
        self._validate_settings(store)
        super().__init__(settings=store, **kwargs)
        self._client = client
        self._owns_client = client is None
        self._endpoint = endpoint
        self._turn_detection = turn_detection
        if turn_detection is not None and not isinstance(turn_detection, r.TurnDetection):
            raise ValueError("Prism requires SDK server turn detection")
        self._timeouts = timeouts or r.RealtimeTimeouts()
        self._request_id = request_id
        self._hint_gain = speech_hint_gain
        self._hint_timeout = speech_hint_timeout
        self._shutdown_timeout = shutdown_timeout
        self._pending_limit = max_pending_operations
        self._audio_limit = max_input_audio_bytes
        if max_output_audio_bytes < 192000:
            raise ValueError("max_output_audio_bytes must hold at least 192000 bytes")
        self._output_audio_limit = max_output_audio_bytes
        self._pending_output_audio_bytes = 0
        self._converted_input_rate: int | None = None
        self._pending_audio_bytes = 0
        self._controls: asyncio.Queue = asyncio.Queue(max_pending_operations)
        self._audio: asyncio.Queue = asyncio.Queue(max_pending_operations)
        self._receipts: asyncio.Queue = asyncio.Queue(max_pending_operations)
        self._ready = asyncio.Event()
        self._closed = asyncio.Event()
        self._closing = False
        self._failure: Exception | None = None
        self._session: r.RealtimeSession | None = None
        self._connection = None
        self._connection_task = None
        self._shutdown_task = None
        self._tasks: list[asyncio.Task] = []
        self._requests: set[asyncio.Task] = set()
        self._context: LLMContext | None = None
        self._history: History | None = None
        self._playback = Playback()
        self._unversioned_policy_ambiguous = False
        self._policies: OrderedDict[int | None, tuple[bool, dict]] = OrderedDict()
        self._calls: dict[str, tuple[r.ToolCall, bool]] = {}
        self._transcripts: dict[str, str] = {}
        self._input_resampler = create_stream_resampler(clear_after_secs=None)
        self._input_lock = asyncio.Lock()
        self._typed: dict[str, LLMInputTextFrame] = {}
        self._typed_bound: dict[str, asyncio.Event] = {}
        self._operation_results: set[tuple[int, str]] = set()
        for event in ("on_session_updated", "on_operation_result", "on_rime_event"):
            self._register_event_handler(event)

    @property
    def _sdk(self) -> r.RealtimeSession:
        if self._session is None:
            raise RuntimeError("Prism session is not open")
        return self._session

    @property
    def _ledger(self) -> History:
        if self._history is None:
            raise RuntimeError("Prism history is not initialized")
        return self._history

    @property
    def _conversation(self) -> LLMContext:
        if self._context is None:
            raise RuntimeError("Prism context is not bound")
        return self._context

    @classmethod
    def _resolve_adapter_class(cls):
        return RimeRealtimeLLMAdapter

    def can_generate_metrics(self) -> bool:
        """Report provider turn-to-first-output timing through normal metrics frames."""
        return True

    @property
    def accepts_intermediate_function_call_results(self) -> bool:
        """Prism accepts final tool results only."""
        return False

    @property
    def session_info(self) -> r.SessionInfo | None:
        """Accepted session information, including the actual provider request ID."""
        return self._sdk.info if self._session else None

    async def get_history(self) -> tuple[r.HistoryItem, ...]:
        """Read completed server history without changing the Pipecat record."""
        if self._connection_task is None:
            raise RuntimeError("Prism conversation is not open")
        await self._ready.wait()
        if not self._session or not self.is_usable or self._closing:
            raise RuntimeError("Prism conversation is not usable")
        return await self._sdk.get_history()

    def service_metadata_frame(self) -> LLMServiceMetadataFrame:
        """Advertise acknowledged mutations, explicit text, and external turns."""
        return LLMServiceMetadataFrame(
            service_name=self.name,
            is_realtime_service=True,
            acknowledges_operations=True,
            accepts_input_text=True,
            user_turn_strategies=ExternalUserTurnStrategies(),
        )

    def owns_audio_output(self, context_id: str) -> bool:
        """Whether a playback report belongs to this conversation."""
        return context_id in self._playback.outputs

    def _service_tools(self):
        return self._settings.tools

    def _validate_settings(self, settings: Settings) -> None:
        supported = {
            "model",
            "system_instruction",
            "tools",
            "voice",
            "interrupt_on_speech",
            "tool_fillers",
            "truncation",
            "response_metadata",
            "filter_incomplete_user_turns",
            "use_provider_thinking_default",
        }
        for name, value in settings.given_fields().items():
            if name not in supported and value is not None:
                raise ValueError(f"Prism does not support setting {name}")
        if settings.model != "prism" or settings.filter_incomplete_user_turns:
            raise ValueError("Prism requires its own model and server turn detection")
        if settings.system_instruction is not None and (
            not isinstance(settings.system_instruction, str)
            or not settings.system_instruction.strip()
        ):
            raise ValueError("Prism instructions cannot be blank or reset")
        if not isinstance(settings.interrupt_on_speech, bool):
            raise ValueError("interrupt_on_speech must be a boolean")

    async def start(self, frame: StartFrame) -> None:
        """Start independent control, audio, and receipt consumers."""
        await super().start(frame)
        for coroutine in (self._control_loop(), self._audio_loop(), self._receipt_loop()):
            self._tasks.append(self.create_task(coroutine))

    def _bind_context(self, context: LLMContext | None = None) -> None:
        if self._context is None:
            self._context = context or LLMContext()
        elif context is not None and context is not self._context:
            raise ValueError("A Prism conversation requires one shared LLMContext")
        if self._connection_task is None:
            self._connection_task = self.create_task(self._open())

    def _sdk_tools(self, settings: Settings) -> list[r.ToolDefinition]:
        tools = self._adapter.tools(settings.tools)
        fillers = settings.tool_fillers
        if not isinstance(fillers, dict) or set(fillers) - {tool.name for tool in tools}:
            raise ValueError("Prism fillers must name advertised tools")
        if any(not isinstance(filler, r.PrismFiller) for filler in fillers.values()):
            raise ValueError("Prism fillers must be SDK PrismFiller values")
        return [replace(tool, prism_filler=fillers.get(tool.name)) for tool in tools]

    def _record_policy(self) -> None:
        info = self._sdk.info
        advertised = {tool.name for tool in self._sdk_tools(self._settings)}
        handlers = {
            name: handler
            for name, handler in self._functions.items()
            if name is None or name in advertised
        }
        policy = (info.interrupt_on_speech, handlers)
        if info.settings_revision is None:
            previous = self._policies.get(None)
            if previous is not None and previous != policy:
                self._unversioned_policy_ambiguous = True
            if self._unversioned_policy_ambiguous:
                self._policies.pop(None, None)
                return
        self._policies[info.settings_revision] = policy
        while len(self._policies) > 128:
            self._policies.popitem(last=False)

    async def _open(self) -> None:
        try:
            if self._client is None:
                self._client = Rime()
            if is_given(self._conversation.tools):
                self._settings.tools = self._conversation.tools
            self._sync_registered_tool_handlers(self._settings.tools)
            # Validate all initial history before opening or submitting mutations.
            await self._adapter.get_llm_invocation_params(self._conversation)
            self._connection = self._client.realtime.connect(
                endpoint=self._endpoint,
                instructions=assert_given(self._settings.system_instruction),
                voice=assert_given(self._settings.voice),
                tools=self._sdk_tools(self._settings),
                interrupt_on_speech=assert_given(self._settings.interrupt_on_speech),
                turn_detection=self._turn_detection,
                timeouts=self._timeouts,
                request_id=self._request_id,
                response_metadata=assert_given(self._settings.response_metadata),
                truncation=assert_given(self._settings.truncation),
            )
            self._session = await self._connection.__aenter__()
            self._history = History(self._session)
            self._record_policy()
            self._tasks.append(self.create_task(self._event_loop()))
            await self._ledger.edit(self._conversation.get_messages(), append=True)
            await self._call_event_handler("on_session_updated", self._sdk.info)
            self._ready.set()
        except Exception as error:
            await self._fail(error)
            self._ready.set()
            await self._close_resources()

    async def _enqueue(self, queue: asyncio.Queue, item: Any) -> None:
        if self._closing or not self.is_usable:
            request = item.request if isinstance(item, _LLMContextOperationFrame) else item
            if isinstance(request, Frame):
                await self._result(request, "input", "rejected", "Prism conversation is not usable")
            return
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            await self._fail(RuntimeError("Prism pending operation limit exceeded"))

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Capture frame intent without waiting for network operations."""
        try:
            await self._process_frame(frame, direction)
        except Exception as error:
            request = frame.request if isinstance(frame, _LLMContextOperationFrame) else frame
            await self._operation_error(request, "input", error)

    async def _process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, (LLMContextFrame, LLMUpdateSettingsFrame)):
            await AIService.process_frame(self, frame, direction)
        else:
            await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
        elif isinstance(frame, (EndFrame, CancelFrame)):
            # Keep output alive until the shutdown task has collected its receipts.
            pass
        elif isinstance(frame, AudioPlaybackReportFrame):
            if frame.context_id in self._playback.outputs:
                if self._closed.is_set():
                    self._playback.accept_report(frame)
                    return
                try:
                    self._receipts.put_nowait(frame)
                except asyncio.QueueFull:
                    await self._fail(RuntimeError("Prism receipt queue limit exceeded"))
            else:
                await self.push_frame(frame, direction)
        elif isinstance(frame, ServiceSwitcherDeactivatedFrame):
            await self._stop_responses("application")
        elif isinstance(frame, InterruptionFrame):
            await self._stop_responses(frame.cause)
            await self.push_frame(frame, direction)
        elif isinstance(frame, _LLMContextBindFrame):
            self._bind_context(frame.context)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            if frame.service is not None and frame.service is not self:
                await self.push_frame(frame, direction)
                return
            proposal = copy.copy(frame)
            proposal.delta = copy.deepcopy(
                frame.delta or self.Settings.from_mapping(frame.settings)
            )
            await self._enqueue(self._controls, proposal)
        elif isinstance(frame, _LLMContextOperationFrame):
            self._bind_context(frame.context)
            await self._enqueue(self._controls, frame)
        elif isinstance(frame, LLMContextFrame):
            self._bind_context(frame.context)
            await self._enqueue(
                self._controls, _LLMContextOperationFrame.capture(frame, self._conversation)
            )
        elif isinstance(frame, InputAudioRawFrame):
            if self._closing or not self.is_usable:
                return
            self._pending_audio_bytes += len(frame.audio)
            if self._pending_audio_bytes > self._audio_limit:
                await self._fail(RuntimeError("Prism microphone buffer limit exceeded"))
            else:
                await self._enqueue(self._audio, frame)
            await self.push_frame(frame, direction)
        elif isinstance(
            frame,
            (
                LLMRunFrame,
                LLMInputTextFrame,
                LLMMessagesAppendFrame,
                LLMMessagesUpdateFrame,
                LLMMessagesTransformFrame,
                LLMSetToolsFrame,
                LLMSetToolChoiceFrame,
            ),
        ):
            self._bind_context()
            await self._enqueue(
                self._controls, _LLMContextOperationFrame.capture(frame, self._conversation)
            )
        elif isinstance(frame, InputImageRawFrame):
            await self._result(frame, "input", "rejected", "Prism does not support images or video")
        else:
            await self.push_frame(frame, direction)

    @staticmethod
    def _operation_kind(request: Frame) -> str:
        if isinstance(request, (LLMUpdateSettingsFrame, LLMSetToolsFrame)):
            return "settings"
        if isinstance(request, LLMSetToolChoiceFrame):
            return "tool_choice"
        if isinstance(request, LLMInputTextFrame):
            return "input"
        if isinstance(request, (LLMRunFrame, LLMContextFrame)):
            return "reply"
        return "history"

    async def _control_loop(self) -> None:
        while True:
            operation = await self._controls.get()
            request = (
                operation.request if isinstance(operation, _LLMContextOperationFrame) else operation
            )
            kind = self._operation_kind(request)
            try:
                await self._ready.wait()
                if not self.is_usable or self._closing:
                    await self._result(
                        request, kind, "rejected", "Prism conversation is not usable"
                    )
                    continue
                if isinstance(operation, _LLMContextOperationFrame) and operation.validation_error:
                    raise ValueError(operation.validation_error)
                if isinstance(request, LLMUpdateSettingsFrame):
                    assert request.delta is not None
                    await self._settings_operation(request, request.delta)
                elif isinstance(request, LLMSetToolsFrame):
                    await self._settings_operation(
                        request,
                        self.Settings(tools=request.tools if is_given(request.tools) else []),
                    )
                elif isinstance(request, LLMSetToolChoiceFrame):
                    if request.tool_choice != "auto":
                        raise ValueError("Prism only supports automatic tool selection")
                    await self._result(request, "tool_choice", "accepted")
                elif isinstance(request, (LLMRunFrame, LLMContextFrame)):
                    assert isinstance(operation, _LLMContextOperationFrame)
                    assert operation.messages is not None
                    if not self._ledger.matches(
                        operation.messages, operation.base, operation.captured_at
                    ):
                        kind = "history"
                        self._ledger.restore(self._conversation)
                        base = self._ledger.edit_base(operation.base, operation.captured_at)
                        await self._ledger.edit(operation.messages, append=False, base=base)
                        operation.commit_messages(operation.messages, append=False, base=base)
                        await self._result(request, "history", "accepted")
                    self._spawn_request(self._reply(request), request, kind="reply")
                elif isinstance(request, LLMInputTextFrame):
                    self._spawn_request(self._typed_input(request), request)
                elif isinstance(
                    request,
                    (LLMMessagesAppendFrame, LLMMessagesUpdateFrame, LLMMessagesTransformFrame),
                ):
                    assert isinstance(operation, _LLMContextOperationFrame)
                    assert operation.messages is not None
                    append = isinstance(request, LLMMessagesAppendFrame)
                    base = (
                        operation.base
                        if append
                        else self._ledger.edit_base(operation.base, operation.captured_at)
                    )
                    await self._ledger.edit(
                        operation.messages,
                        append=append,
                        base=base,
                        retained=operation.retained,
                    )
                    operation.commit_messages(operation.messages, append=append, base=base)
                    await self._result(request, "history", "accepted")
                    if request.run_llm:
                        self._spawn_request(self._reply(request), request, kind="reply")
            except asyncio.CancelledError:
                await self._result(
                    request, kind, "unknown", "Shutdown ended acknowledgment tracking"
                )
                raise
            except Exception as error:
                await self._operation_error(request, kind, error)
            finally:
                self._controls.task_done()

    def _spawn_request(
        self, coroutine: Coroutine[Any, Any, None], request: Frame, *, kind: str = "input"
    ) -> None:
        if len(self._requests) >= self._pending_limit:
            coroutine.close()
            self._tasks.append(
                self.create_task(self._fail(RuntimeError("Prism request limit exceeded")))
            )
            return

        async def run():
            try:
                await coroutine
            except asyncio.CancelledError:
                await self._result(request, "input", "unknown", "Shutdown ended request tracking")
                raise
            except Exception as error:
                await self._operation_error(request, kind, error)

        task = self.create_task(run())
        self._requests.add(task)
        task.add_done_callback(self._requests.discard)

    async def _typed_input(self, frame: LLMInputTextFrame) -> None:
        # Admission does not hold the configuration/history mutation queue.
        async with self._input_lock:
            await self.start_ttfb_metrics()
            ref = await self._sdk.send_text(frame.text)
            self._typed[ref.response_id] = frame
            response = self._playback.responses.get(ref.response_id)
            if response:
                self._commit_typed(response, frame)
                self._typed.pop(ref.response_id, None)
            self._typed_bound.setdefault(ref.response_id, asyncio.Event()).set()
            await self._result(frame, "input", "accepted")

    def _commit_typed(self, response: Response, frame: LLMInputTextFrame) -> None:
        if frame.metadata.get("rime_input_committed"):
            return
        if len(response.input_ids) != 1:
            raise RuntimeError("Prism typed input does not have one unambiguous input identity")
        frame.metadata["rime_input_committed"] = True
        for item_id in response.input_ids:
            self._ledger.transcript(item_id, frame.text, self._context)
        response.audio_response = frame.audio_response
        response.explicit_audio_response = True
        response.input_request_id = frame.id
        response.client_message_id = frame.client_message_id

    async def _reply(self, request: Frame) -> None:
        await self._sdk.request_reply()
        await self._result(request, "reply", "accepted")

    async def _settings_operation(self, request: Frame, delta: LLMSettings) -> None:
        proposal = copy.deepcopy(self._settings)
        proposal.apply_update(self.Settings.from_mapping(delta.given_fields()))
        supplied = delta.given_fields()
        if "tools" in supplied:
            if not is_given(proposal.tools):
                proposal.tools = []
            names = {tool.name for tool in self._adapter.tools(proposal.tools)}
            if "tool_fillers" not in supplied:
                proposal.tool_fillers = {
                    name: filler
                    for name, filler in assert_given(proposal.tool_fillers).items()
                    if name in names
                }
        self._validate_settings(proposal)
        tools = self._sdk_tools(proposal)
        sdk = {}
        for field_name, sdk_name in (
            ("system_instruction", "instructions"),
            ("voice", "voice"),
            ("interrupt_on_speech", "interrupt_on_speech"),
            ("truncation", "truncation"),
        ):
            if field_name in supplied:
                value = getattr(proposal, field_name)
                if value is None:
                    raise ValueError(f"Prism cannot reset {field_name} to None")
                sdk[sdk_name] = value
        if "tools" in supplied or "tool_fillers" in supplied:
            sdk["tools"] = tools
        if sdk:
            await self._sdk.update(**sdk)
            metadata = self._settings.response_metadata
            self._settings = proposal
            self._settings.response_metadata = metadata
            self._conversation.set_tools(proposal.tools or [])
            self._base_system_instruction = assert_given(proposal.system_instruction)
            self._sync_registered_tool_handlers(proposal.tools)
            self._record_policy()
            await self._call_event_handler("on_session_updated", self._sdk.info)
            await self._result(request, "settings", "accepted")
        if "response_metadata" in supplied:
            try:
                await self._sdk.set_response_metadata(supplied["response_metadata"] or {})
                self._settings.response_metadata = copy.deepcopy(supplied["response_metadata"])
                self._record_policy()
                await self._result(request, "response_metadata", "accepted")
                await self._call_event_handler("on_session_updated", self._sdk.info)
            except asyncio.CancelledError:
                await self._result(
                    request,
                    "response_metadata",
                    "unknown",
                    "Shutdown ended acknowledgment tracking",
                )
                raise
            except Exception as error:
                await self._operation_error(request, "response_metadata", error)
        if not sdk and "response_metadata" not in supplied:
            await self._result(request, "settings", "accepted")

    async def _audio_loop(self) -> None:
        while True:
            frame = await self._audio.get()
            try:
                await self._ready.wait()
                if not self.is_usable or self._closing:
                    continue
                audio, sample_rate, channels = frame.audio, frame.sample_rate, frame.num_channels
                if (
                    not 1 <= channels <= 32
                    or not 8000 <= sample_rate <= 192000
                    or len(audio) % (channels * 2)
                ):
                    raise ValueError(
                        "Prism input requires complete PCM16 frames, 1 to 32 channels, and 8 to 192 kHz"
                    )
                if channels > 2 or (
                    sample_rate not in (8000, 16000, 24000, 48000) and channels > 1
                ):
                    samples = np.frombuffer(audio, dtype="<i2").reshape(-1, channels)
                    audio = samples.astype(np.int32).mean(axis=1).astype("<i2").tobytes()
                    channels = 1
                if (
                    self._converted_input_rate is not None
                    and sample_rate != self._converted_input_rate
                ):
                    tail = await self._input_resampler.flush()
                    if tail:
                        await self._sdk.send_audio(r.AudioChunk(data=tail))
                    self._input_resampler = create_stream_resampler(clear_after_secs=None)
                    self._converted_input_rate = None
                if sample_rate not in (8000, 16000, 24000, 48000):
                    self._converted_input_rate = sample_rate
                    audio = await self._input_resampler.resample(audio, sample_rate, 16000)
                    sample_rate = 16000
                size = 192000 - 192000 % (channels * 2)
                for offset in range(0, len(audio), size):
                    await self._sdk.send_audio(
                        r.AudioChunk(
                            data=audio[offset : offset + size],
                            format=r.PCMFormat(sample_rate=sample_rate, channels=channels),
                        )
                    )
            except Exception as error:
                await self._fail(error)
            finally:
                self._pending_audio_bytes -= len(frame.audio)
                self._audio.task_done()

    async def _receipt_loop(self) -> None:
        while True:
            frame = await self._receipts.get()
            try:
                accepted = self._playback.accept_report(frame)
                if accepted:
                    response, output = accepted
                    self._pending_output_audio_bytes -= output.audio_bytes
                    if (
                        frame.cause == "user"
                        and frame.outcome != "completed"
                        and frame.played_ms is not None
                    ):
                        await self._sdk.report_playback(
                            r.PlaybackInterrupted(
                                output=output.ref, audio_end_ms=int(frame.played_ms)
                            )
                        )
                    await self._finish_playback(response)
            except Exception as error:
                await self._fail(error)
            finally:
                self._receipts.task_done()

    async def _finish_playback(self, response: Response) -> None:
        report = self._playback.finished_report(response)
        if report:
            await self._sdk.report_playback(report)
            response.finished = True

    async def _event_loop(self) -> None:
        try:
            async for event in self._sdk.events:
                await self._event(event)
            if not self._closing:
                raise ConnectionError("Prism event stream ended")
        except Exception as error:
            await self._fail(error)

    async def _event(self, event: r.SessionEvent) -> None:
        payload = event.payload
        await self._call_event_handler("on_rime_event", event)
        if isinstance(payload, r.SpeechHint):
            if self._speech_policy():
                await self.push_frame(AudioOutputGainFrame(self._hint_gain, self._hint_timeout))
        elif isinstance(payload, r.SpeechStarted):
            self._ledger.reserve_input(payload.item_id, self._context)
            await self.push_frame(AudioOutputGainFrame())
            await self.broadcast_frame(
                ProposedUserStartedSpeakingFrame, enable_interruptions=self._speech_policy()
            )
        elif isinstance(payload, r.SpeechStopped):
            await self.start_ttfb_metrics()
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)
        elif isinstance(payload, r.InputCommitted):
            self._ledger.reserve_input(payload.item_id, self._context)
        elif isinstance(payload, (r.TranscriptDelta, r.TranscriptFinal)):
            final = isinstance(payload, r.TranscriptFinal)
            text = (
                payload.text
                if final
                else self._transcripts.get(payload.item_id, "") + payload.delta
            )
            if len(text) > 65536 or len(self._transcripts) >= 128:
                raise RuntimeError("Prism transcript retention limit exceeded")
            if final:
                self._transcripts.pop(payload.item_id, None)
                self._ledger.transcript(payload.item_id, text, self._context)
            else:
                self._transcripts[payload.item_id] = text
            cls = TranscriptionFrame if final else InterimTranscriptionFrame
            frame = cls(text=text, user_id="", timestamp=time_now_iso8601())
            frame.metadata["provider_managed_context"] = True
            await self.broadcast_frame_instance(frame)
        elif isinstance(payload, r.TranscriptFailed):
            self._transcripts.pop(payload.item_id, None)
            await self.push_error(str(payload.error), exception=r.RimeRealtimeError(payload.error))
        elif isinstance(payload, r.ResponseStarted):
            policy = self._policies.get(payload.settings_revision)
            if policy is None:
                raise RuntimeError("Prism response settings revision is unknown or expired")
            response = self._playback.start(payload, *policy)
            if payload.parent:
                parent = self._playback.responses.get(payload.parent.response_id)
                if parent is None:
                    raise RuntimeError("Prism continuation parent is unknown or expired")
                response.audio_response = parent.audio_response
                response.explicit_audio_response = parent.explicit_audio_response
                response.input_request_id = parent.input_request_id
                response.client_message_id = parent.client_message_id
            self._ledger.response_epoch += 1
            self._ledger.active_responses.add(payload.response.response_id)
            self._ledger.active_inputs.update(response.input_ids)
            for item_id in response.input_ids:
                self._ledger.reserve_input(item_id, self._context)
            typed = self._typed.pop(payload.response.response_id, None)
            # The event can precede delivery of send_text's returned reference.
            # The request task binds the input to that returned reference.
            if typed:
                self._commit_typed(response, typed)
            elif payload.cause == "text":
                bound = self._typed_bound.setdefault(payload.response.response_id, asyncio.Event())
                async with asyncio.timeout(self._timeouts.request_s):
                    await bound.wait()
            self._typed_bound.pop(payload.response.response_id, None)
            started = LLMFullResponseStartFrame()
            started.metadata["rime"] = {
                "provider_request_id": self._sdk.info.request_id,
                "response_id": payload.response.response_id,
                "input_item_ids": response.input_ids,
                "settings_revision": payload.settings_revision,
                "response_metadata": dict(payload.metadata),
                "input_request_id": response.input_request_id,
                "client_message_id": response.client_message_id,
            }
            await self.push_frame(started)
            if payload.parent:
                parent = self._playback.responses.get(payload.parent.response_id)
                if parent is None:
                    raise RuntimeError("Prism continuation parent is unknown or expired")
                if parent.stopped:
                    await self._stop_response(response, parent.cause or "application")
        elif isinstance(payload, r.ResponseAbandoned):
            await self._stop_response(
                self._playback.responses[payload.response.response_id], "application"
            )
        elif isinstance(payload, r.MessageStarted):
            output = self._playback.message(payload.output)
            response = self._playback.responses[payload.output.response.response_id]
            await self.push_frame(AudioOutputStartFrame(output.context_id))
            await self.push_frame(TTSStartedFrame(context_id=output.context_id))
            if response.stopped:
                await self._end_output(output)
        elif isinstance(payload, (r.TextDelta, r.TextDone, r.AudioDelta, r.AudioDone)):
            response = self._playback.responses[payload.output.response.response_id]
            output = response.outputs[payload.output.item_id]
            if (
                output.discarded
                or response.stopped
                or (output.closed and isinstance(payload, r.AudioDelta))
            ):
                return
            if isinstance(payload, r.TextDelta):
                await self.stop_ttfb_metrics()
                output.text += payload.delta
                if len(output.text) > 65536:
                    raise RuntimeError("Prism output text retention limit exceeded")
                text = LLMTextFrame(payload.delta)
                text.append_to_context = False
                await self.push_frame(text)
            elif isinstance(payload, r.TextDone):
                output.text = payload.text
                text = TTSTextFrame(payload.text, text_type=TextType.SENTENCE)
                text.append_to_context = False
                await self.push_frame(text)
            elif isinstance(payload, r.AudioDelta):
                if response.audio_response and (
                    response.explicit_audio_response or not self._skip_tts
                ):
                    audio = payload.audio
                    if (
                        self._pending_output_audio_bytes + len(audio.data)
                        > self._output_audio_limit
                    ):
                        raise RuntimeError("Prism output PCM retention limit exceeded")
                    output.audio_bytes += len(audio.data)
                    self._pending_output_audio_bytes += len(audio.data)
                    frame = TTSAudioRawFrame(
                        audio.data,
                        audio.format.sample_rate,
                        audio.format.channels,
                        context_id=output.context_id,
                    )
                    await self.stop_ttfb_metrics()
                    await self.process_ttfa_metrics(frame)
                    await self.push_frame(frame)
            else:
                await self._end_output(output)
        elif isinstance(payload, r.ToolCall):
            if len(self._calls) >= 1024:
                raise RuntimeError("Prism tool retention limit exceeded")
            if payload.call.call_id in self._calls:
                return
            response = self._playback.responses[payload.call.response_id]
            if response.stopped:
                raise RuntimeError("Prism emitted a tool call for a stopped reply")
            self._calls[payload.call.call_id] = payload, False
            response.calls.add(payload.call.call_id)
            self._ledger.pending_calls.add(payload.call.call_id)
            self._ledger.add_live(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": payload.call.call_id,
                            "type": "function",
                            "function": {
                                "name": payload.name,
                                "arguments": json.dumps(payload.arguments),
                            },
                        }
                    ],
                },
                [payload.item_id],
                self._context,
            )
            await self.run_function_calls(
                [
                    FunctionCallFromLLM(
                        function_name=payload.name,
                        tool_call_id=payload.call.call_id,
                        arguments=payload.arguments,
                        context=self._context,
                    )
                ]
            )
        elif isinstance(payload, r.ResponseEnded):
            response = self._playback.responses[payload.response.response_id]
            response.ended = True
            self._ledger.active_responses.discard(payload.response.response_id)
            self._ledger.active_inputs.difference_update(response.input_ids)
            for output in response.outputs.values():
                await self._end_output(output)
                if output.text:
                    self._ledger.add_live(
                        {"role": "assistant", "content": output.text},
                        [output.ref.item_id],
                        self._context,
                    )
            await self.push_frame(LLMFullResponseEndFrame())
            self._spawn_request(self._finish_playback(response), LLMRunFrame(), kind="playback")
        elif isinstance(payload, r.FaultEvent) and payload.error.request_id is None:
            error = r.RimeRealtimeError(payload.error)
            if payload.error.scope == "session":
                await self._fail(error)
            else:
                await self.push_error(str(error), exception=error)

    def _speech_policy(self) -> bool:
        for response in reversed(self._playback.responses.values()):
            if not response.finished and not response.stopped:
                return response.interrupt_on_speech
        return assert_given(self._settings.interrupt_on_speech)

    async def _end_output(self, output: Output) -> None:
        if not output.closed:
            output.closed = True
            await self.push_frame(TTSStoppedFrame(context_id=output.context_id))
            await self.push_frame(AudioOutputEndFrame(output.context_id))

    async def _stop_response(self, response: Response, cause: InterruptionCause) -> None:
        if response.stopped:
            return
        response.stopped = True
        response.cause = cause
        for output in response.outputs.values():
            output.discarded = True
            await self.push_frame(AudioOutputStopFrame(output.context_id, cause))
            await self._end_output(output)
        if not response.ended:
            self._spawn_request(self._sdk.cancel(response.ref), LLMRunFrame(), kind="stop")

    async def _stop_responses(self, cause: InterruptionCause) -> None:
        await self.push_frame(AudioOutputGainFrame())
        for response in self._playback.responses.values():
            if not response.finished:
                await self._stop_response(response, cause)

    async def _run_function_call(self, runner_item: FunctionCallRunnerItem) -> None:
        call, _ = self._calls[runner_item.tool_call_id]
        response = self._playback.responses[call.call.response_id]
        runner_item.registry_item = response.handlers.get(
            runner_item.function_name,
            response.handlers.get(None)
            or self._build_missing_function_call_registry_item(runner_item.function_name),
        )
        runner_item.pin_handler = True
        runner_item.run_llm = False
        await super()._run_function_call(runner_item)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Keep live context writes and tool-result delivery under the conversation owner."""
        if isinstance(
            frame,
            (
                FunctionCallResultFrame,
                FunctionCallInProgressFrame,
                FunctionCallsStartedFrame,
                FunctionCallCancelFrame,
            ),
        ):
            frame.metadata["provider_managed_context"] = True
        if isinstance(frame, FunctionCallCancelFrame) and frame.tool_call_id in self._calls:
            call, submitted = self._calls[frame.tool_call_id]
            if not submitted:
                self._calls[frame.tool_call_id] = call, True
                self._spawn_request(
                    self._tool_result(call, {"error": "cancelled"}), frame, kind="tool_result"
                )
        if isinstance(frame, FunctionCallResultFrame) and frame.tool_call_id in self._calls:
            call, submitted = self._calls[frame.tool_call_id]
            if frame.properties and not frame.properties.is_final:
                await self.push_error("Prism does not support intermediate tool results")
                return
            if not submitted:
                self._calls[frame.tool_call_id] = call, True
                self._spawn_request(
                    self._tool_result(call, frame.result), frame, kind="tool_result"
                )
            frame.run_llm = False
        await super().push_frame(frame, direction)

    async def _tool_result(self, call: r.ToolCall, result: Any) -> None:
        output = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        await self._sdk.submit_tool_result(call.call, output)
        self._ledger.pending_calls.discard(call.call.call_id)
        self._ledger.add_live(
            {"role": "tool", "tool_call_id": call.call.call_id, "content": output},
            [],
            self._context,
        )
        response = self._playback.responses[call.call.response_id]
        response.calls.discard(call.call.call_id)
        if not response.calls and not response.stopped:
            try:
                await self._sdk.continue_reply(response.ref)
            except (RimeInputError, r.RimeRealtimeError) as error:
                if isinstance(error, r.RimeRealtimeError) and (
                    error.fault.scope != "event"
                    or error.fault.code != "tool_continuation_unavailable"
                ):
                    raise
                await self._sdk.request_reply(tool_call=call.call)

    async def _result(
        self, request: Frame, kind: str, outcome: LLMOperationOutcome, detail: str | None = None
    ) -> None:
        key = request.id, kind
        if key in self._operation_results:
            return
        if len(self._operation_results) >= 65536:
            await self._fail(RuntimeError("Prism operation result retention limit exceeded"))
            return
        self._operation_results.add(key)
        frame = LLMOperationResultFrame(request.id, kind, outcome, detail, self.name)
        await self.broadcast_frame_instance(frame)
        await self._call_event_handler("on_operation_result", frame)

    async def _operation_error(self, request: Frame, kind: str, error: Exception) -> None:
        outcome: LLMOperationOutcome
        if isinstance(error, PartialHistoryError):
            outcome = "partial"
        elif isinstance(error, (ValueError, TypeError, RimeInputError, r.RealtimeAdmissionTimeout)):
            outcome = "rejected"
        elif isinstance(error, r.RimeRealtimeError) and error.fault.scope == "event":
            fault = error.fault
            known_rejection = kind in {
                "history",
                "settings",
                "response_metadata",
                "tool_choice",
            } or (
                fault.response_id is None
                and fault.item_id is None
                and fault.code not in {"typed_turn_superseded", "response_superseded"}
            )
            outcome = "rejected" if known_rejection else "unknown"
        else:
            outcome = "unknown"
        await self._result(request, kind, outcome, str(error))
        if outcome in ("unknown", "partial"):
            await self._fail(error)
        else:
            await self.push_error(str(error), exception=error)

    async def _fail(self, error: Exception) -> None:
        if self._failure is not None:
            return
        self._failure = error
        if self._session is not None:
            await self._stop_responses("error")
        await self.push_frame(AudioOutputGainFrame())
        await self.push_error(str(error), exception=error, force_treat_as_permanent=True)

    async def stop(self, frame: EndFrame) -> None:
        """Drain accepted work with one bounded shutdown deadline."""
        self._begin_shutdown(frame)

    async def cancel(self, frame: CancelFrame) -> None:
        """Start cancellation once; repeated cancellation keeps the original deadline."""
        self._begin_shutdown(frame)

    def _begin_shutdown(self, frame: EndFrame | CancelFrame) -> None:
        if self._shutdown_task is None:
            self._shutdown_task = self.create_task(self._shutdown(frame))

    async def _shutdown(self, frame: EndFrame | CancelFrame) -> None:
        try:
            async with asyncio.timeout(self._shutdown_timeout):
                if isinstance(frame, EndFrame):
                    await self._controls.join()
                    await self._audio.join()
                    if self._session and self.is_usable:
                        tail = await self._input_resampler.flush()
                        if tail:
                            await self._sdk.send_audio(r.AudioChunk(data=tail))
                    if self._requests:
                        await asyncio.wait(self._requests)
                    while any(
                        not response.finished for response in self._playback.responses.values()
                    ):
                        await asyncio.sleep(0.01)
                self._closing = True
                if self._session:
                    await self._stop_responses("shutdown")
                    await self._controls.join()
                    if self._requests:
                        await asyncio.wait(self._requests)
                    while any(
                        not response.finished for response in self._playback.responses.values()
                    ):
                        await asyncio.sleep(0.01)
                    await self._receipts.join()
        except TimeoutError:
            logger.warning(f"{self}: Prism shutdown drain deadline expired")
        finally:
            self._closing = True
            unresolved = [
                response for response in self._playback.responses.values() if not response.finished
            ]
            if unresolved and self._session:
                await self._stop_responses("shutdown")
                await self.push_error(
                    "Prism shutdown could not confirm all playback reports; timing remains unknown"
                )
                for response in unresolved:
                    for output in response.outputs.values():
                        if output.report is None:
                            self._playback.accept_report(
                                AudioPlaybackReportFrame(
                                    output.context_id, "failed", None, "shutdown"
                                )
                            )
            await self._close_resources()
            if isinstance(frame, EndFrame):
                await super().stop(frame)
                await self.push_frame(frame)
            else:
                await super().cancel(frame)
                await self.push_frame(frame)
            self._closed.set()

    async def _close_resources(self) -> None:
        current = asyncio.current_task()
        for task in [*self._requests, *self._tasks, self._connection_task]:
            if task is not None and task is not current and not task.done():
                await self.cancel_task(task)
        while not self._controls.empty():
            operation = self._controls.get_nowait()
            request = (
                operation.request if isinstance(operation, _LLMContextOperationFrame) else operation
            )
            kind = self._operation_kind(request)
            await self._result(request, kind, "rejected", "Conversation closed before submission")
            self._controls.task_done()
        while not self._audio.empty():
            audio = self._audio.get_nowait()
            self._pending_audio_bytes -= len(audio.audio)
            self._audio.task_done()
        while not self._receipts.empty():
            report = self._receipts.get_nowait()
            try:
                self._playback.accept_report(report)
            except ValueError as error:
                await self.push_error(str(error), exception=error)
            finally:
                self._receipts.task_done()
        if self._connection is not None:
            connection, self._connection = self._connection, None
            await connection.__aexit__(None, None, None)
        if self._owns_client and self._client is not None:
            await self._client.close()

    async def cleanup(self):
        """Close all owned resources, including failed or unused startup."""
        if self._shutdown_task is not None and not self._shutdown_task.done():
            await self._shutdown_task
        else:
            self._closing = True
            await self._close_resources()
        await super().cleanup()

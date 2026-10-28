#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Rime Prism conversations through Pipecat's standard realtime frame interface."""

import asyncio
import copy
import json
import math
import time
from collections import OrderedDict
from collections.abc import Coroutine
from dataclasses import dataclass, field, fields, replace
from typing import Any
from uuid import uuid4

import numpy as np
from loguru import logger

try:
    from rimelabs_sdk import Rime, RimeInputError
    from rimelabs_sdk import realtime as r
except ModuleNotFoundError as error:
    logger.error('Install Rime realtime support with `uv pip install "pipecat-ai[rime-realtime]"`.')
    raise Exception(f"Missing module: {error}") from error

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    InputImageRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMServiceMetadataFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService
from pipecat.services.llm_service import FunctionCallRunnerItem, LLMService
from pipecat.services.rime.realtime._adapter import RimeRealtimeLLMAdapter
from pipecat.services.rime.realtime._aggregators import (
    RimeContextAggregatorPair,
    RimeReplyFrame,
    _RimeContextFrame,
)
from pipecat.services.rime.realtime._history import History
from pipecat.services.settings import LLMSettings
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.text.base_text_aggregator import TextType
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given, is_given


@dataclass
class RimeRealtimeSettings(LLMSettings):
    """Prism configuration and sparse settings deltas.

    Parameters:
        tools: Advertised tools. An empty list removes all tools.
        voice: Prism voice.
        interrupt_on_speech: Server interruption setting. Local turn strategies
            control Pipecat interruptions independently.
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


@dataclass
class _Output:
    ref: r.OutputRef
    context_id: str = field(default_factory=lambda: uuid4().hex)
    text: str = ""
    audio_ms: float = 0
    started_at: float | None = None
    closed: bool = False
    recorded: bool = False


@dataclass
class _Response:
    ref: r.ResponseRef
    instruction_item_id: str | None = None
    outputs: dict[str, _Output] = field(default_factory=dict)
    calls: set[str] = field(default_factory=set)
    input_ids: tuple[str, ...] = ()
    ended: bool = False
    stopped: bool = False
    audio_response: bool = True
    done: asyncio.Event = field(default_factory=asyncio.Event)
    continued: bool = False


@dataclass
class _Operation:
    messages: list | None = None
    append_messages: list | None = None
    tools: Any = NOT_GIVEN
    tool_choice: Any = NOT_GIVEN
    delta: LLMSettings | None = None
    run_llm: bool = False
    audio_response: bool = True
    instruction: str | None = None
    response_metadata: dict[str, str] | None = None


class RimeRealtimeLLMService(LLMService[RimeRealtimeLLMAdapter]):
    """One Prism conversation using Rime context aggregators and standard transports.

    Text input is imported into history before requesting a reply. Aggregators
    own the local conversation record; rejected provider edits do not roll it
    back. Playback positions are estimates and completion receipts are not sent.
    Speech hints do not change output volume.

    Use one instance per conversation. An injected client remains caller-owned.
    There is no automatic reconnect or replay. Start a fresh instance with known
    valid, completed history after an unknown or partially applied operation.

    Events:
        on_session_updated: Acknowledged SDK session information.
        on_rime_event: SDK events, including provider diagnostics.
    """

    Settings = RimeRealtimeSettings
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
        max_pending_operations: int = 256,
        max_input_audio_bytes: int = 1920000,
        shutdown_timeout: float = 5.0,
        **kwargs,
    ):
        """Configure the SDK connection and bounded background work.

        Args:
            client: Optional caller-owned SDK client.
            endpoint: Optional SDK-validated Prism WebSocket endpoint.
            settings: Initial settings and advertised tools.
            turn_detection: Opening-only SDK server turn timing settings.
            timeouts: SDK connection, readiness, and acknowledgment limits.
            request_id: Requested provider correlation ID.
            max_pending_operations: Bound for pending operations and requests.
            max_input_audio_bytes: Bound for buffered microphone PCM.
            shutdown_timeout: Maximum time to drain requests during shutdown.
            **kwargs: Arguments for LLMService.
        """
        if max_pending_operations < 1 or max_input_audio_bytes < 192000:
            raise ValueError("Prism queue limits must be positive and hold one SDK audio chunk")
        if not math.isfinite(shutdown_timeout) or shutdown_timeout <= 0:
            raise ValueError("shutdown_timeout must be finite and positive")
        if turn_detection is not None and not isinstance(turn_detection, r.TurnDetection):
            raise ValueError("Prism requires SDK server turn detection")
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
        self._timeouts = timeouts or r.RealtimeTimeouts()
        self._request_id = request_id
        self._shutdown_timeout = shutdown_timeout
        self._pending_limit = max_pending_operations
        self._audio_limit = max_input_audio_bytes
        self._pending_audio_bytes = 0
        self._converted_input_rate: int | None = None
        self._input_resampler = create_stream_resampler(clear_after_secs=None)
        self._controls: asyncio.Queue[_Operation] = asyncio.Queue(max_pending_operations)
        self._audio: asyncio.Queue[InputAudioRawFrame] = asyncio.Queue(max_pending_operations)
        self._ready = asyncio.Event()
        self._closing = False
        self._failure: Exception | None = None
        self._session: r.RealtimeSession | None = None
        self._connection = None
        self._connection_task: asyncio.Task | None = None
        self._tasks: list[asyncio.Task] = []
        self._requests: set[asyncio.Task] = set()
        self._context: LLMContext | None = None
        self._history: History | None = None
        self._responses: OrderedDict[str, _Response] = OrderedDict()
        self._calls: dict[str, tuple[r.ToolCall, bool]] = {}
        self._transcripts: dict[str, str] = {}
        self._reply_lock = asyncio.Lock()
        self._reply_audio_response = True
        self._reply_started = asyncio.Event()
        self._instruction_items: OrderedDict[str, tuple[str, bool]] = OrderedDict()
        self._instruction_lock = asyncio.Lock()
        self._latest_audio_output: _Output | None = None
        for event in ("on_session_updated", "on_rime_event"):
            self._register_event_handler(event)

    @classmethod
    def _resolve_adapter_class(cls):
        return RimeRealtimeLLMAdapter

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
    def session_info(self) -> r.SessionInfo | None:
        """Acknowledged session information, including the provider request ID."""
        return self._sdk.info if self._session else None

    @property
    def accepts_intermediate_function_call_results(self) -> bool:
        """Prism accepts final tool results only."""
        return False

    def can_generate_metrics(self) -> bool:
        """Emit standard turn-to-first-output metrics."""
        return True

    def service_metadata_frame(self) -> LLMServiceMetadataFrame:
        """Use the existing realtime aggregation and external turn strategies."""
        return LLMServiceMetadataFrame(
            service_name=self.name,
            is_realtime_service=True,
            user_turn_strategies=ExternalUserTurnStrategies(),
        )

    def create_context_aggregator(
        self,
        context: LLMContext,
        *,
        user_params: LLMUserAggregatorParams | None = None,
        assistant_params: LLMAssistantAggregatorParams | None = None,
    ) -> LLMContextAggregatorPair:
        """Create the aggregator pair required for this Prism conversation.

        Speech transcripts update local context without requesting a second
        reply. Application message frames preserve append or replace intent.

        Args:
            context: Shared conversation record.
            user_params: Standard user aggregator configuration.
            assistant_params: Standard assistant aggregator configuration.

        Returns:
            User and assistant aggregators for the pipeline.
        """
        return RimeContextAggregatorPair(
            context, user_params=user_params, assistant_params=assistant_params
        )

    async def get_history(self) -> tuple[r.HistoryItem, ...]:
        """Read completed server history without changing local context."""
        if self._connection_task is None:
            raise RuntimeError("Prism conversation is not open")
        await self._ready.wait()
        if not self.is_usable or self._closing:
            raise RuntimeError("Prism conversation is not usable")
        return await self._sdk.get_history()

    async def delete_instruction_item(self, item_id: str) -> None:
        """Remove an observed saved instruction after its reply ends."""
        if self._connection_task is None:
            raise RuntimeError("Prism conversation is not open")
        await self._ready.wait()
        if not self.is_usable or self._closing:
            raise RuntimeError("Prism conversation is not usable")
        async with self._instruction_lock:
            state = self._instruction_items.get(item_id)
            if state is None:
                raise RimeInputError("Unknown saved Prism instruction item")
            if not state[1]:
                raise RimeInputError("Cannot delete a saved instruction before its reply ends")
            await self._sdk.delete_item(item_id)
            if self._instruction_items.get(item_id) == state:
                del self._instruction_items[item_id]

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

    def _sdk_tools(self, settings: Settings) -> list[r.ToolDefinition]:
        tools = self._adapter.tools(settings.tools)
        fillers = settings.tool_fillers
        if not isinstance(fillers, dict) or set(fillers) - {tool.name for tool in tools}:
            raise ValueError("Prism fillers must name advertised tools")
        if any(not isinstance(filler, r.PrismFiller) for filler in fillers.values()):
            raise ValueError("Prism fillers must be SDK PrismFiller values")
        return [replace(tool, prism_filler=fillers.get(tool.name)) for tool in tools]

    async def start(self, frame: StartFrame) -> None:
        """Start independent control and audio consumers."""
        await super().start(frame)
        for coroutine in (self._control_loop(), self._audio_loop()):
            self._tasks.append(self.create_task(coroutine))

    def _bind_context(self, context: LLMContext) -> None:
        if self._context is not None and context is not self._context:
            raise ValueError("A Prism conversation requires one shared LLMContext")
        self._context = context
        if self._connection_task is None:
            self._connection_task = self.create_task(self._open())

    async def _open(self) -> None:
        try:
            if self._client is None:
                self._client = Rime()
            self._sync_registered_tool_handlers(self._settings.tools)
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
            self._tasks.append(self.create_task(self._event_loop()))
            await self._call_event_handler("on_session_updated", self._sdk.info)
        except Exception as error:
            await self._fail(error)
        finally:
            self._ready.set()

    async def _enqueue(self, operation: _Operation) -> None:
        if self._closing or not self.is_usable:
            return
        try:
            self._controls.put_nowait(operation)
        except asyncio.QueueFull:
            await self._fail(RuntimeError("Prism pending operation limit exceeded"))

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Queue network work without holding up audio and control frames."""
        try:
            if isinstance(frame, (LLMContextFrame, LLMUpdateSettingsFrame)):
                await AIService.process_frame(self, frame, direction)
            else:
                await super().process_frame(frame, direction)
            if isinstance(frame, LLMContextFrame):
                initial = self._context is None
                self._bind_context(frame.context)
                messages = None
                append_messages = None
                if initial or not isinstance(frame, _RimeContextFrame):
                    messages = copy.deepcopy(frame.context.get_messages())
                elif frame.history_update is not None:
                    if frame.history_update.append:
                        append_messages = frame.history_update.messages
                    else:
                        messages = frame.history_update.messages
                await self._enqueue(
                    _Operation(
                        messages=messages,
                        append_messages=append_messages,
                        tools=frame.context.tools,
                        tool_choice=frame.context.tool_choice,
                        run_llm=True,
                        audio_response=not self._skip_tts,
                        instruction=(
                            frame.instruction if isinstance(frame, _RimeContextFrame) else None
                        ),
                        response_metadata=(
                            dict(frame.response_metadata)
                            if isinstance(frame, _RimeContextFrame)
                            and frame.response_metadata is not None
                            else None
                        ),
                    )
                )
            elif isinstance(frame, LLMRunFrame):
                self._bind_context(self._context if self._context is not None else LLMContext())
                await self._enqueue(
                    _Operation(
                        run_llm=True,
                        audio_response=not self._skip_tts,
                        instruction=frame.instruction
                        if isinstance(frame, RimeReplyFrame)
                        else None,
                        response_metadata=(
                            dict(frame.response_metadata)
                            if isinstance(frame, RimeReplyFrame)
                            and frame.response_metadata is not None
                            else None
                        ),
                    )
                )
            elif isinstance(frame, LLMUpdateSettingsFrame):
                if frame.service is not None and frame.service is not self:
                    await self.push_frame(frame, direction)
                else:
                    await self._enqueue(
                        _Operation(
                            delta=copy.deepcopy(
                                frame.delta or self.Settings.from_mapping(frame.settings)
                            ),
                        )
                    )
            elif isinstance(frame, LLMSetToolsFrame):
                await self._enqueue(_Operation(delta=self.Settings(tools=frame.tools)))
                await self.push_frame(frame, direction)
            elif isinstance(frame, InputAudioRawFrame):
                if not self._closing and self.is_usable:
                    self._pending_audio_bytes += len(frame.audio)
                    if self._pending_audio_bytes > self._audio_limit or self._audio.full():
                        self._pending_audio_bytes -= len(frame.audio)
                        await self._fail(RuntimeError("Prism microphone buffer limit exceeded"))
                    else:
                        self._audio.put_nowait(frame)
                await self.push_frame(frame, direction)
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._latest_audio_output = None
                await self.push_frame(frame, direction)
            elif isinstance(frame, InterruptionFrame):
                self._spawn_request(self._interrupt())
                await self.push_frame(frame, direction)
            elif isinstance(frame, InputImageRawFrame):
                await self.push_error("Prism does not support images or video")
            else:
                await self.push_frame(frame, direction)
        except Exception as error:
            await self._operation_error(error)

    async def _control_loop(self) -> None:
        while True:
            operation = await self._controls.get()
            try:
                await self._ready.wait()
                if self._closing or not self.is_usable:
                    continue
                if is_given(operation.tool_choice) and operation.tool_choice != "auto":
                    raise ValueError("Prism only supports automatic tool selection")
                if operation.messages is not None:
                    await self._ledger.sync(operation.messages)
                if operation.append_messages is not None:
                    await self._ledger.edit(operation.append_messages, append=True)
                if is_given(operation.tools) and operation.tools != self._settings.tools:
                    await self._settings_operation(self.Settings(tools=operation.tools))
                if operation.delta is not None:
                    await self._settings_operation(operation.delta)
                if operation.run_llm:
                    self._spawn_request(
                        self._reply(
                            operation.audio_response,
                            instruction=operation.instruction,
                            metadata=operation.response_metadata,
                        )
                    )
            except Exception as error:
                await self._operation_error(error)
            finally:
                self._controls.task_done()

    def _spawn_request(self, coroutine: Coroutine[Any, Any, None]) -> None:
        if self._closing or not self.is_usable:
            coroutine.close()
            return
        if len(self._requests) >= self._pending_limit:
            coroutine.close()
            self._tasks.append(
                self.create_task(self._fail(RuntimeError("Prism request limit exceeded")))
            )
            return

        async def run():
            try:
                await coroutine
            except Exception as error:
                await self._operation_error(error)

        task = self.create_task(run())
        self._requests.add(task)
        task.add_done_callback(self._requests.discard)

    async def _reply(
        self,
        audio_response: bool,
        *,
        tool_call: r.ToolCallRef | None = None,
        instruction: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        async with self._reply_lock:
            self._reply_audio_response = audio_response
            self._reply_started.clear()
            await self.start_ttfb_metrics()
            await self._sdk.request_reply(
                instruction=instruction, tool_call=tool_call, metadata=metadata
            )
            await self._reply_started.wait()

    async def _settings_operation(self, delta: LLMSettings) -> None:
        proposal = copy.deepcopy(self._settings)
        proposal.apply_update(self.Settings.from_mapping(delta.given_fields()))
        supplied = delta.given_fields()
        if "tools" in supplied and "tool_fillers" not in supplied:
            names = {tool.name for tool in self._adapter.tools(proposal.tools)}
            proposal.tool_fillers = {
                name: filler
                for name, filler in assert_given(proposal.tool_fillers).items()
                if name in names
            }
        self._validate_settings(proposal)
        tools = self._sdk_tools(proposal)
        update = {}
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
                update[sdk_name] = value
        if "tools" in supplied or "tool_fillers" in supplied:
            update["tools"] = tools
        if update:
            await self._sdk.update(**update)
            metadata = self._settings.response_metadata
            self._settings = proposal
            self._settings.response_metadata = metadata
            self._base_system_instruction = assert_given(proposal.system_instruction)
            self._sync_registered_tool_handlers(proposal.tools)
            if self._context is not None and "tools" in supplied:
                self._context.set_tools(proposal.tools or [])
            await self._call_event_handler("on_session_updated", self._sdk.info)
        if "response_metadata" in supplied:
            await self._sdk.set_response_metadata(supplied["response_metadata"] or {})
            self._settings.response_metadata = copy.deepcopy(supplied["response_metadata"])
            await self._call_event_handler("on_session_updated", self._sdk.info)

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
                            format=r.PCMFormat(
                                sample_rate=sample_rate, channels=1 if channels == 1 else 2
                            ),
                        )
                    )
            except Exception as error:
                await self._fail(error)
            finally:
                self._pending_audio_bytes -= len(frame.audio)
                self._audio.task_done()

    async def _event_loop(self) -> None:
        try:
            async for event in self._sdk.events:
                if self._closing or self._failure:
                    break
                await self._event(event)
            if not self._closing and self._failure is None:
                raise ConnectionError("Prism event stream ended")
        except Exception as error:
            if not self._closing:
                await self._fail(error)

    async def _event(self, event: r.SessionEvent) -> None:
        payload = event.payload
        await self._call_event_handler("on_rime_event", event)
        if isinstance(payload, r.SpeechStarted):
            self._ledger.reserve_input(payload.item_id)
            await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
        elif isinstance(payload, r.SpeechStopped):
            await self.start_ttfb_metrics()
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)
        elif isinstance(payload, r.InputCommitted):
            self._ledger.reserve_input(payload.item_id)
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
                self._ledger.transcript(payload.item_id, text)
            else:
                self._transcripts[payload.item_id] = text
            cls = TranscriptionFrame if final else InterimTranscriptionFrame
            await self.broadcast_frame_instance(
                cls(text=text, user_id="", timestamp=time_now_iso8601())
            )
        elif isinstance(payload, r.TranscriptFailed):
            self._transcripts.pop(payload.item_id, None)
            await self.push_error(str(payload.error), exception=r.RimeRealtimeError(payload.error))
        elif isinstance(payload, r.ResponseStarted):
            if len(self._responses) >= 128:
                key, oldest = next(iter(self._responses.items()))
                if not oldest.ended or oldest.calls:
                    raise RuntimeError("Prism response retention limit has active replies or tools")
                self._responses.pop(key)
                for call_id in tuple(self._calls):
                    if self._calls[call_id][0].call.response_id == key:
                        self._calls.pop(call_id)
            response = _Response(
                payload.response,
                instruction_item_id=payload.instruction_item_id,
                input_ids=payload.input_item_ids,
            )
            if payload.cause == "proactive":
                response.audio_response = self._reply_audio_response
                self._reply_started.set()
            if payload.parent:
                parent = self._responses.get(payload.parent.response_id)
                response.stopped = parent is not None and parent.stopped
                if parent is not None:
                    response.audio_response = parent.audio_response
            self._responses[payload.response.response_id] = response
            if payload.instruction_item_id is not None:
                self._instruction_items.pop(payload.instruction_item_id, None)
                if len(self._instruction_items) >= 128:
                    self._instruction_items.popitem(last=False)
                self._instruction_items[payload.instruction_item_id] = (
                    payload.response.response_id,
                    False,
                )
            self._ledger.response_epoch += 1
            self._ledger.active_responses.add(payload.response.response_id)
            self._ledger.active_inputs.update(response.input_ids)
            for item_id in response.input_ids:
                self._ledger.reserve_input(item_id)
            started = LLMFullResponseStartFrame()
            started.metadata["rime"] = {
                "provider_request_id": self._sdk.info.request_id,
                "response_id": payload.response.response_id,
                "input_item_ids": response.input_ids,
                "settings_revision": payload.settings_revision,
                "response_metadata": dict(payload.metadata),
                "instruction_item_id": payload.instruction_item_id,
            }
            await self.push_frame(started)
            if response.stopped:
                self._spawn_request(self._sdk.cancel(response.ref))
        elif isinstance(payload, r.ResponseAbandoned):
            response = self._responses[payload.response.response_id]
            response.stopped = True
            await self.broadcast_interruption()
        elif isinstance(payload, r.MessageStarted):
            response = self._responses[payload.output.response.response_id]
            if len(response.outputs) >= 128:
                raise RuntimeError("Prism output retention limit exceeded")
            output = _Output(payload.output)
            response.outputs[payload.output.item_id] = output
            if not response.stopped:
                await self.push_frame(TTSStartedFrame(context_id=output.context_id))
        elif isinstance(payload, (r.TextDelta, r.TextDone, r.AudioDelta, r.AudioDone)):
            response = self._responses[payload.output.response.response_id]
            output = response.outputs[payload.output.item_id]
            if response.stopped or output.closed:
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
                if output.text and not output.recorded:
                    self._ledger.add_live(
                        {"role": "assistant", "content": output.text}, [output.ref.item_id]
                    )
                    output.recorded = True
                await self.push_frame(TTSTextFrame(payload.text, text_type=TextType.SENTENCE))
            elif isinstance(payload, r.AudioDelta):
                if response.audio_response:
                    audio = payload.audio
                    self._latest_audio_output = output
                    output.audio_ms += (
                        len(audio.data)
                        * 1000
                        / (audio.format.sample_rate * audio.format.channels * 2)
                    )
                    if output.started_at is None:
                        output.started_at = time.monotonic()
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
            if payload.call.call_id in self._calls:
                return
            if len(self._calls) >= 1024:
                raise RuntimeError("Prism tool retention limit exceeded")
            response = self._responses[payload.call.response_id]
            if response.stopped:
                return
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
                                "arguments": json.dumps(payload.arguments, ensure_ascii=False),
                            },
                        }
                    ],
                },
                [payload.item_id],
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
            response = self._responses[payload.response.response_id]
            response.ended = True
            if response.instruction_item_id is not None:
                state = self._instruction_items.get(response.instruction_item_id)
                if state is not None and state[0] == payload.response.response_id:
                    self._instruction_items[response.instruction_item_id] = (state[0], True)
            response.done.set()
            self._ledger.active_responses.discard(payload.response.response_id)
            self._ledger.active_inputs.difference_update(response.input_ids)
            for output in response.outputs.values():
                await self._end_output(output)
                if output.text and not output.recorded:
                    self._ledger.add_live(
                        {"role": "assistant", "content": output.text}, [output.ref.item_id]
                    )
            await self.push_frame(LLMFullResponseEndFrame())
        elif isinstance(payload, r.FaultEvent) and payload.error.request_id is None:
            error = r.RimeRealtimeError(payload.error)
            if payload.error.scope == "session":
                await self._fail(error)
            else:
                await self.push_error(str(error), exception=error)

    async def _end_output(self, output: _Output) -> None:
        if not output.closed:
            output.closed = True
            await self.push_frame(TTSStoppedFrame(context_id=output.context_id))

    async def _interrupt(self) -> None:
        if self._session is None:
            return
        output, self._latest_audio_output = self._latest_audio_output, None
        for response in tuple(self._responses.values()):
            if response.stopped or (
                response.ended
                and not response.calls
                and (output is None or output.ref.response.response_id != response.ref.response_id)
            ):
                continue
            response.stopped = True
            if not response.ended:
                await self._sdk.cancel(response.ref)
            for current in response.outputs.values():
                await self._end_output(current)
        # Standard transports do not return output identities or device clocks.
        # Report one elapsed-time estimate for the latest emitted audio only.
        if output is not None and output.started_at is not None:
            elapsed = (time.monotonic() - output.started_at) * 1000
            await self._sdk.report_playback(
                r.PlaybackInterrupted(
                    output=output.ref,
                    audio_end_ms=int(min(elapsed, output.audio_ms)),
                )
            )

    async def _run_function_call(self, runner_item: FunctionCallRunnerItem) -> None:
        runner_item.run_llm = False
        await super()._run_function_call(runner_item)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Submit final tool results once while normal aggregators record them."""
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, (FunctionCallResultFrame, FunctionCallCancelFrame))
            and frame.tool_call_id in self._calls
        ):
            if (
                isinstance(frame, FunctionCallResultFrame)
                and frame.properties
                and not frame.properties.is_final
            ):
                await self.push_error("Prism does not support intermediate tool results")
                return
            call, submitted = self._calls[frame.tool_call_id]
            if not submitted:
                self._calls[frame.tool_call_id] = call, True
                result = (
                    (json.dumps(frame.result, ensure_ascii=False) if frame.result else "COMPLETED")
                    if isinstance(frame, FunctionCallResultFrame)
                    else "CANCELLED"
                )
                self._spawn_request(self._tool_result(call, result))
            frame.run_llm = False
        await super().push_frame(frame, direction)

    async def _tool_result(self, call: r.ToolCall, output: str) -> None:
        await self._sdk.submit_tool_result(call.call, output)
        self._ledger.pending_calls.discard(call.call.call_id)
        self._ledger.add_live(
            {"role": "tool", "tool_call_id": call.call.call_id, "content": output}, []
        )
        response = self._responses[call.call.response_id]
        response.calls.discard(call.call.call_id)
        await response.done.wait()
        if not response.calls and not response.stopped and not response.continued:
            response.continued = True
            try:
                await self._sdk.continue_reply(response.ref)
            except (RimeInputError, r.RimeRealtimeError) as error:
                if isinstance(error, r.RimeRealtimeError) and (
                    error.fault.scope != "event"
                    or error.fault.code != "tool_continuation_unavailable"
                ):
                    raise
                await self._reply(response.audio_response, tool_call=call.call)

    async def _operation_error(self, error: Exception) -> None:
        if isinstance(error, (ValueError, TypeError, RimeInputError, r.RealtimeAdmissionTimeout)):
            await self.push_error(str(error), exception=error)
        elif isinstance(error, r.RimeRealtimeError) and error.fault.scope == "event":
            await self.push_error(str(error), exception=error)
        else:
            await self._fail(error)

    async def _fail(self, error: Exception) -> None:
        if self._failure is not None:
            return
        self._failure = error
        await self.broadcast_interruption()
        await self.push_error(str(error), exception=error, force_treat_as_permanent=True)

    async def stop(self, frame: EndFrame) -> None:
        """Drain queued requests, then close the SDK with a bounded deadline."""
        try:
            async with asyncio.timeout(self._shutdown_timeout):
                if self._connection_task is not None:
                    await self._controls.join()
                    await self._audio.join()
                    if self._session and self.is_usable:
                        tail = await self._input_resampler.flush()
                        if tail:
                            await self._sdk.send_audio(r.AudioChunk(data=tail))
                    while self._requests or any(
                        not response.ended for response in self._responses.values()
                    ):
                        await asyncio.sleep(0.005)
        except TimeoutError:
            logger.warning(f"{self}: Prism shutdown drain deadline expired")
        finally:
            self._closing = True
            await self._close_resources()
            await super().stop(frame)

    async def cancel(self, frame: CancelFrame) -> None:
        """Cancel owned work without waiting for playback completion."""
        self._closing = True
        await self._close_resources()
        await super().cancel(frame)

    async def _close_resources(self) -> None:
        current = asyncio.current_task()
        for task in [*self._requests, *self._tasks, self._connection_task]:
            if task is not None and task is not current and not task.done():
                await self.cancel_task(task)
        if self._connection is not None:
            connection, self._connection = self._connection, None
            await connection.__aexit__(None, None, None)
        if self._owns_client and self._client is not None:
            client, self._client = self._client, None
            await client.close()

    async def cleanup(self):
        """Close resources after startup failure or worker cleanup."""
        self._closing = True
        await self._close_resources()
        await super().cleanup()

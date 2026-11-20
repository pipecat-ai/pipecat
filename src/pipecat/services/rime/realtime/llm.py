#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Rime Prism conversations through Pipecat's standard realtime frame interface."""

import asyncio
import copy
import json
import time
from collections import OrderedDict, deque
from collections.abc import Coroutine
from dataclasses import dataclass, field, fields, replace
from typing import Any, Literal

import numpy as np
from loguru import logger

try:
    from rimelabs_sdk import Rime, RimeInputError
    from rimelabs_sdk import realtime as r
except ModuleNotFoundError as error:
    logger.error('In order to use RimeRealtimeLLMService, you need to `uv add "pipecat-ai[rime]"`.')
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
    LLMMessagesAppendFrame,
    LLMServiceMetadataFrame,
    LLMSetToolChoiceFrame,
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
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.rime.realtime._adapter import RimeRealtimeLLMAdapter
from pipecat.services.rime.realtime._conversation import (
    Conversation,
    RimeStartupParams,
    Tool,
    connect,
)
from pipecat.services.rime.realtime._conversation import (
    Output as _Output,
)
from pipecat.services.rime.realtime._conversation import (
    Response as _Response,
)
from pipecat.services.rime.realtime._history import History, PrismHistoryUpdateError
from pipecat.services.rime.realtime._playback import DiscardPlayback
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
        speed: Speech speed from 0.85 to 1.3. ``None`` uses the server default.
        interrupt_min_words: Words, from 1 to 10, the user says over a reply
            before Prism interrupts it. ``None`` uses the server default.
        interrupt_phrases: Phrases that interrupt a reply regardless of
            ``interrupt_min_words``. An empty list clears them.
        acknowledgment_phrases: Phrases, such as "uh huh", that do not
            interrupt a reply. An empty list clears them.
    """

    tools: Any = field(default_factory=lambda: NOT_GIVEN)
    voice: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    interrupt_on_speech: bool | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    tool_fillers: dict[str, r.PrismFiller] | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    truncation: r.Truncation | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    response_metadata: dict[str, str] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    interrupt_min_words: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    interrupt_phrases: list[str] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    acknowledgment_phrases: list[str] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


_INTERRUPTION_RULES = ("interrupt_min_words", "interrupt_phrases", "acknowledgment_phrases")
# Settings sent to Prism under the same value, by their SDK names.
_SESSION_FIELDS = (
    ("voice", "voice"),
    ("interrupt_on_speech", "interrupt_on_speech"),
    ("truncation", "truncation"),
    ("speed", "speed"),
)
# Settings an update can't set to None, which the SDK reads as keeping the value.
_REQUIRED_FIELDS = (
    "system_instruction",
    *(name for name, _ in _SESSION_FIELDS),
    *_INTERRUPTION_RULES,
)

# The SDK refuses more than 128 pending operations of its own.
_MAX_PENDING_OPERATIONS = 128
# 20 s of 48 kHz mono PCM16 waiting to be sent.
_MAX_INPUT_AUDIO_BYTES = 1_920_000
_SHUTDOWN_TIMEOUT_S = 5.0


@dataclass
class _Operation:
    append: list | None = None
    tools: Any = NOT_GIVEN
    tool_choice: Any = NOT_GIVEN
    delta: LLMSettings | None = None
    run_llm: bool = False
    audio_response: bool = True
    tool: Tool | None = None
    tool_output: str = ""
    tool_final: bool = False
    # The async-tool message that settles the call after progress was written.
    tool_message: str = ""
    # The write is content a reply can answer: appended messages or a tool update.
    answerable: bool = False
    # The reply answers earlier content and is skipped if none since the last
    # reply was written.
    answers_content: bool = False


class RimeRealtimeLLMService(LLMService[RimeRealtimeLLMAdapter]):
    """One Prism conversation using standard context aggregators and transports.

    Use it with :class:`~pipecat.processors.aggregators.llm_response_universal.LLMContextAggregatorPair`.
    The first context frame opens the conversation with the context's messages
    and starts a reply. Messages from an
    :class:`~pipecat.frames.frames.LLMMessagesAppendFrame` are added to Prism
    history as they arrive; with ``run_llm=True``, Prism replies when the
    assistant aggregator asks for a response, unless a Prism response that
    started after the write already took them in. The service only appends
    to Prism history: replacing or transforming context messages
    mid-conversation does not change it, and an :class:`~pipecat.frames.frames.LLMRunFrame` or
    a replacement with ``run_llm=True`` after the opening starts no reply. Use
    ``LLMMessagesAppendFrame(run_llm=True)`` to ask for one. Rejected provider
    writes do not roll back the local context.

    Tool results reach Prism as they are produced. An intermediate result is a
    history message beside the call, which stays open until its final result
    is submitted. Prism answers results when the assistant aggregator asks for
    a response, continuing the tool round when it is still current. Prism
    mentions an async call's result that arrives after its round closed only
    when asked, so the service asks for a reply that reports it.

    Playback positions are estimates and completion receipts are not sent.
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
        startup: RimeStartupParams | None = None,
        playback: Literal["estimated", "discard"] = "estimated",
        **kwargs,
    ):
        """Configure the SDK connection and bounded background work.

        Args:
            client: Optional caller-owned SDK client.
            endpoint: Optional SDK-validated Prism WebSocket endpoint.
            settings: Initial settings and advertised tools.
            turn_detection: SDK server turn settings for the opening, such as
                eagerness and the end-of-turn silence floor. Its interruption
                rules may instead be given in ``settings``, which can also
                change them later.
            timeouts: SDK connection, readiness, and acknowledgment limits.
            request_id: Requested provider correlation ID.
            startup: Bounded connection retry policy.
            playback: Estimated normal output or discard all audio with zero-played receipts.
            **kwargs: Arguments for LLMService.
        """
        if turn_detection is not None and not isinstance(turn_detection, r.TurnDetection):
            raise ValueError("Prism requires SDK server turn detection")
        given = {
            "extra",
            "model",
            "filter_incomplete_user_turns",
            "use_provider_thinking_default",
            "extensions",
        }
        unset: dict[str, Any] = {f.name: None for f in fields(LLMSettings) if f.name not in given}
        default_settings = self.Settings(
            **unset,
            model="prism",
            filter_incomplete_user_turns=False,
            use_provider_thinking_default=False,
            tools=None,
            voice=None,
            interrupt_on_speech=True,
            tool_fillers={},
            truncation=None,
            response_metadata=None,
            speed=None,
            interrupt_min_words=None,
            interrupt_phrases=None,
            acknowledgment_phrases=None,
        )
        if settings is not None:
            default_settings.apply_update(copy.deepcopy(settings))
        if turn_detection is not None:
            for name in _INTERRUPTION_RULES:
                value = getattr(turn_detection, name)
                if value is None:
                    continue
                if getattr(default_settings, name) is not None:
                    raise ValueError(f"Set {name} in turn_detection or settings, not both")
                setattr(default_settings, name, list(value) if isinstance(value, tuple) else value)
        self._validate_settings(default_settings)
        super().__init__(settings=default_settings, **kwargs)
        if playback not in ("estimated", "discard"):
            raise ValueError("playback must be estimated or discard")
        self._startup = startup or RimeStartupParams()
        self._playback_mode = playback
        self._playback: DiscardPlayback | None = None
        self._conversation = Conversation()
        self._tool_choice = "auto"
        self._reply_owed = False
        self._owed_audio_response = True
        self._writes_unwritten = 0
        self._content_written = False
        # Calls whose final results were written since the last reply.
        self._owed_calls: list[Tool] = []
        self._early_appends: list[list] = []
        self._interruption_epoch = 0
        self._reply_epoch = 0
        self._response_seen = asyncio.Event()
        self._request_pending = False
        self._admitted_response: _Response | None = None
        self._client = client
        self._owns_client = client is None
        self._endpoint = endpoint
        self._turn_detection = turn_detection
        self._timeouts = timeouts or r.RealtimeTimeouts()
        self._request_id = request_id
        self._shutdown_timeout = _SHUTDOWN_TIMEOUT_S
        self._pending_audio_bytes = 0
        self._converted_input_rate: int | None = None
        self._input_resampler = create_stream_resampler(clear_after_secs=None)
        self._controls: asyncio.Queue[_Operation] = asyncio.Queue(_MAX_PENDING_OPERATIONS)
        # Bounded by _MAX_INPUT_AUDIO_BYTES.
        self._audio: asyncio.Queue[InputAudioRawFrame] = asyncio.Queue()
        self._ready = asyncio.Event()
        self._closing = False
        self._failure: Exception | None = None
        self._session: r.RealtimeSession | None = None
        self._connection = None
        self._connection_task: asyncio.Task | None = None
        self._tasks: list[asyncio.Task] = []
        self._requests: set[asyncio.Task] = set()
        self._context: LLMContext | None = None
        # The instructions and tools Prism last accepted.
        self._prism_instructions: str | None = None
        self._prism_tools: list[r.ToolDefinition] = []
        self._history: History | None = None
        self._transcripts: dict[str, str] = {}
        self._reply_lock = asyncio.Lock()
        self._reply_audio_response = True
        # The requested reply reports a late tool result and survives interruptions.
        self._reply_reports = False
        # Audio outputs in the order the transport plays them; the first is playing.
        self._playing: deque[_Output] = deque()
        for event in ("on_session_updated", "on_rime_event"):
            self._register_event_handler(event)

    @property
    def _responses(self) -> OrderedDict[str, _Response]:
        return self._conversation.responses

    @property
    def _calls(self) -> dict[str, Tool]:
        return self._conversation.calls

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
        """Intermediate results reach Prism as history messages beside the open call.

        Prism keeps one result per call, so only the final result answers it.
        """
        return True

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
            "speed",
            *_INTERRUPTION_RULES,
            "filter_incomplete_user_turns",
            "use_provider_thinking_default",
            "extensions",
        }
        for name, value in settings.given_fields().items():
            if name not in supported and value is not None:
                raise ValueError(f"Prism does not support setting {name}")
        if settings.model != "prism" or settings.filter_incomplete_user_turns:
            raise ValueError("Prism requires its own model and server turn detection")
        if settings.system_instruction is not None and not isinstance(
            settings.system_instruction, str
        ):
            raise ValueError("Prism instructions must be a string")
        if not isinstance(settings.interrupt_on_speech, bool):
            raise ValueError("interrupt_on_speech must be a boolean")
        speed = settings.speed
        if speed is not None and (
            isinstance(speed, bool)
            or not isinstance(speed, (int, float))
            or not 0.85 <= speed <= 1.3
        ):
            raise ValueError("Prism speed must be a number from 0.85 to 1.3")
        self._interruption_rules(settings, _INTERRUPTION_RULES)

    @staticmethod
    def _interruption_rules(settings: Settings, names) -> r.TurnDetectionUpdate:
        rules = {name: getattr(settings, name) for name in names}
        try:
            return r.TurnDetectionUpdate(
                **{name: value for name, value in rules.items() if value is not None}
            )
        except RimeInputError as error:
            raise ValueError(str(error)) from None

    def _sdk_tools(self, settings: Settings) -> list[r.ToolDefinition]:
        tools = self._adapter.tools(settings.tools)
        fillers = settings.tool_fillers
        if not isinstance(fillers, dict):
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

    def _opening_turn_detection(self) -> r.TurnDetection | None:
        rules = self._interruption_rules(self._settings, _INTERRUPTION_RULES)
        given = {
            name: getattr(rules, name)
            for name in _INTERRUPTION_RULES
            if getattr(rules, name) is not None
        }
        if not given:
            return self._turn_detection
        return replace(self._turn_detection or r.TurnDetection(), **given)

    async def _open(self) -> None:
        try:
            if self._client is None:
                self._client = Rime()
            self._sync_registered_tool_handlers(self._settings.tools)
            instructions, tools = self._prism_view()
            self._connection = connect(
                self._client,
                self._startup,
                endpoint=self._endpoint,
                instructions=instructions,
                voice=assert_given(self._settings.voice),
                tools=tools,
                interrupt_on_speech=assert_given(self._settings.interrupt_on_speech),
                turn_detection=self._opening_turn_detection(),
                speed=assert_given(self._settings.speed),
                timeouts=self._timeouts,
                request_id=self._request_id,
                response_metadata=assert_given(self._settings.response_metadata),
                truncation=assert_given(self._settings.truncation),
            )
            self._session = await self._connection.__aenter__()
            self._prism_instructions, self._prism_tools = instructions, tools
            self._history = History(self._session)
            if self._playback_mode == "discard":
                self._playback = DiscardPlayback(self._session)
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
            await super().process_frame(frame, direction)
            if isinstance(frame, LLMContextFrame):
                await self._handle_context(frame, direction)
            elif isinstance(frame, LLMMessagesAppendFrame):
                if self._context is None:
                    # The assistant aggregator adds these to the context, which
                    # may happen after the opening context frame is read.
                    self._early_appends.append(frame.messages)
                else:
                    await self._enqueue(
                        _Operation(append=copy.deepcopy(frame.messages), answerable=True)
                    )
                    self._writes_unwritten += 1
                    self._reply_owed = True
                    # Output settings apply to the reply as of the append.
                    self._owed_audio_response = not self._skip_tts
                await self.push_frame(frame, direction)
            elif isinstance(frame, LLMSetToolChoiceFrame):
                await self._enqueue(_Operation(tool_choice=frame.tool_choice))
                await self.push_frame(frame, direction)
            elif isinstance(frame, LLMUpdateSettingsFrame):
                # LLMService passes it to _update_settings or pushes it on.
                pass
            elif isinstance(frame, LLMSetToolsFrame):
                await self._enqueue(_Operation(delta=self.Settings(tools=frame.tools)))
                await self.push_frame(frame, direction)
            elif isinstance(frame, InputAudioRawFrame):
                if not self._closing and self.is_usable:
                    self._pending_audio_bytes += len(frame.audio)
                    if self._pending_audio_bytes > _MAX_INPUT_AUDIO_BYTES:
                        self._pending_audio_bytes -= len(frame.audio)
                        await self._fail(RuntimeError("Prism microphone buffer limit exceeded"))
                    else:
                        self._audio.put_nowait(frame)
                await self.push_frame(frame, direction)
            elif isinstance(frame, BotStoppedSpeakingFrame):
                # Stops name no output but arrive in playback order, one per
                # output. A stop while the oldest output still streams is a
                # pause in it.
                if self._playing and self._playing[0].closed:
                    self._playing.popleft()
                    if self._playing:
                        self._playing[0].started_at = time.monotonic()
                await self.push_frame(frame, direction)
            elif isinstance(frame, InterruptionFrame):
                self._interruption_epoch += 1
                # The assistant aggregator drops a held reply request on interruption.
                self._reply_owed = False
                self._content_written = False
                self._settle_owed_calls()
                playing = self._playing[0] if self._playing else None
                self._playing.clear()
                self._spawn_request(self._interrupt(playing))
                await self.push_frame(frame, direction)
            elif isinstance(frame, InputImageRawFrame):
                await self.push_error("Prism does not support images or video")
            else:
                await self.push_frame(frame, direction)
        except Exception as error:
            await self._operation_error(error)

    async def _handle_context(self, frame: LLMContextFrame, direction: FrameDirection) -> None:
        initial = self._context is None
        self._bind_context(frame.context)
        if initial:
            # The opening context becomes the conversation's history, followed
            # by earlier appends the assistant aggregator has not added yet.
            messages = frame.context.get_messages()
            present = {id(message) for message in messages}
            missing = [
                message
                for batch in self._early_appends
                for message in batch
                if id(message) not in present
            ]
            self._early_appends.clear()
            append = copy.deepcopy([*messages, *missing])
            run_llm = True
            audio_response = not self._skip_tts
        else:
            # Appended messages and tool results are already in Prism. A context
            # frame travelling upstream comes from the assistant aggregator,
            # asking for a reply to them.
            append = None
            run_llm = direction == FrameDirection.UPSTREAM and self._reply_owed
            audio_response = self._owed_audio_response
        if run_llm and not initial:
            self._reply_owed = False
        await self._enqueue(
            _Operation(
                append=append,
                tools=frame.context.tools,
                tool_choice=frame.context.tool_choice,
                run_llm=run_llm,
                audio_response=audio_response,
                answers_content=run_llm and not initial,
            )
        )

    async def _control_loop(self) -> None:
        while True:
            operation = await self._controls.get()
            unwritten = operation.answerable
            try:
                await self._ready.wait()
                if self._closing or not self.is_usable:
                    continue
                if is_given(operation.tool_choice):
                    if operation.tool_choice not in ("auto", "none", None):
                        raise ValueError("Prism supports automatic or disabled tool selection")
                    self._tool_choice = operation.tool_choice or "auto"
                if operation.append or operation.tool is not None:
                    try:
                        if operation.tool is not None:
                            await self._write_tool_update(
                                operation.tool,
                                operation.tool_output,
                                final=operation.tool_final,
                                message=operation.tool_message,
                            )
                        elif operation.append:
                            await self._ledger.append(operation.append)
                    finally:
                        if unwritten:
                            unwritten = False
                            self._writes_unwritten -= 1
                    if operation.answerable:
                        self._content_written = True
                if is_given(operation.tools) and operation.tools != self._settings.tools:
                    await self._settings_operation(self.Settings(tools=operation.tools))
                if operation.delta is not None:
                    await self._settings_operation(operation.delta)
                if operation.run_llm and (not operation.answers_content or self._content_written):
                    parent = tool_call = None
                    if operation.answers_content:
                        self._content_written = False
                        target = self._tool_reply_target()
                        if target is None:
                            continue
                        parent, tool_call = target
                    self._spawn_request(
                        self._reply(
                            operation.audio_response,
                            parent=parent,
                            tool_call=tool_call,
                            disable_tools=self._tool_choice == "none",
                        )
                    )
            except Exception as error:
                await self._operation_error(error)
            finally:
                if unwritten:
                    self._writes_unwritten -= 1
                self._controls.task_done()

    def _tool_reply_target(self) -> tuple[r.ResponseRef | None, r.ToolCallRef | None] | None:
        """Choose how a reply answers tool results written since the last one.

        Returns the tool round to continue and the call to report, or None when
        a requested continuation already covers them.
        """
        calls = list(self._owed_calls)
        if not calls:
            return None, None
        latest = calls[-1].event.call
        response = self._responses.get(latest.response_id)
        if response is not None:
            # The reply holds the reference while it reports the call.
            response.operations += 1
        self._settle_owed_calls()
        if response is None or response.stopped:
            return None, latest
        if response.continuing and all(
            tool.event.call.response_id == latest.response_id for tool in calls
        ):
            # Prism's continuation waits for every result in the round.
            response.operations -= 1
            self._conversation.release(response)
            return None
        open_calls = any(
            not tool.submitted
            for tool in self._calls.values()
            if tool.event.call.response_id == latest.response_id
        )
        if response.continued or open_calls:
            return None, latest
        response.continued = True
        response.continuing = True
        return response.ref, latest

    def _settle_owed_calls(self) -> None:
        """Release the references held for results that need no reply of their own."""
        calls, self._owed_calls = self._owed_calls, []
        for tool in calls:
            response = self._responses.get(tool.event.call.response_id)
            if response is not None:
                response.operations -= 1
                self._conversation.release(response)

    def _spawn_request(self, coroutine: Coroutine[Any, Any, None]) -> None:
        if self._closing or not self.is_usable:
            coroutine.close()
            return
        if len(self._requests) >= _MAX_PENDING_OPERATIONS:
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
        disable_tools: bool = False,
        parent: r.ResponseRef | None = None,
    ) -> None:
        try:
            await self._locked_reply(
                audio_response,
                tool_call=tool_call,
                instruction=instruction,
                disable_tools=disable_tools,
                parent=parent,
            )
        finally:
            held = parent or tool_call
            response = self._responses.get(held.response_id) if held else None
            if response is not None:
                response.continuing = False
                response.operations -= 1
                self._conversation.release(response)

    async def _locked_reply(
        self,
        audio_response: bool,
        *,
        tool_call: r.ToolCallRef | None,
        instruction: str | None,
        disable_tools: bool,
        parent: r.ResponseRef | None,
        report: bool = False,
    ) -> None:
        epoch = self._interruption_epoch
        async with self._reply_lock:
            if epoch != self._interruption_epoch and not report:
                return
            if self._playback:
                await self._playback.settle()
            if epoch != self._interruption_epoch and not report:
                return
            self._reply_epoch = epoch
            self._reply_reports = report
            self._reply_audio_response = audio_response
            await self.start_ttfb_metrics()
            if not disable_tools:
                await self._request_reply(tool_call, instruction, parent)
                return
            async with self._conversation.settings_lock:
                self._conversation.suppressing_tools = True
                try:
                    info = await self._sdk.update(tools=[])
                    self._conversation.suppress(info)
                    task = self.create_task(self._request_reply(tool_call, instruction, parent))
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        try:
                            ref = await asyncio.shield(task)
                            await self._sdk.cancel(ref)
                        finally:
                            raise
                finally:
                    if not self._closing and self._failure is None:
                        try:
                            tools = self._sdk_tools(self._settings)
                            await self._sdk.update(tools=tools)
                            self._prism_tools = tools
                        except Exception as error:
                            await self._fail(error)
                    self._conversation.suppressing_tools = False

    async def _request_reply(
        self,
        tool_call: r.ToolCallRef | None,
        instruction: str | None,
        parent: r.ResponseRef | None,
    ) -> r.ResponseRef:
        self._request_pending = True
        try:
            return await self._create_reply(tool_call, instruction, parent)
        finally:
            self._request_pending = False
            if self._admitted_response is not None:
                self._admitted_response.operations -= 1
                self._conversation.release(self._admitted_response)
                self._admitted_response = None

    async def _create_reply(
        self,
        tool_call: r.ToolCallRef | None,
        instruction: str | None,
        parent: r.ResponseRef | None,
    ) -> r.ResponseRef:
        if parent is not None:
            try:
                ref = await self._sdk.continue_reply(parent)
                return await self._created_reply(ref)
            except (RimeInputError, r.RimeRealtimeError) as error:
                if isinstance(error, r.RimeRealtimeError) and (
                    error.fault.scope != "event"
                    or error.fault.code != "tool_continuation_unavailable"
                ):
                    raise
        if tool_call is not None and instruction is None:
            instruction = "Continue according to the agent's instructions using this tool result. Reply or call a tool as appropriate."
        ref = await self._sdk.request_reply(instruction=instruction, tool_call=tool_call)
        return await self._created_reply(ref)

    async def _created_reply(self, ref: r.ResponseRef) -> r.ResponseRef:
        while ref.response_id not in self._responses:
            self._response_seen.clear()
            await self._response_seen.wait()
        if self._responses[ref.response_id].stopped:
            await self._sdk.cancel(ref)
        return ref

    async def _update_settings(self, delta: LLMSettings) -> dict[str, Any]:
        """Queue a settings delta behind earlier operations.

        The delta is applied by :meth:`_apply_settings` once earlier operations
        have been written.

        Args:
            delta: An LLM settings delta.

        Returns:
            An empty dict, since nothing has changed yet.
        """
        await self._enqueue(_Operation(delta=copy.deepcopy(delta)))
        return {}

    async def _settings_operation(self, delta: LLMSettings) -> None:
        async with self._conversation.settings_lock:
            await self._apply_settings(delta)

    def _copy_settings(self) -> Settings:
        """Copy the settings for a delta to change without touching them.

        ``apply_update`` replaces fields and merges ``extra`` in place, so a
        shallow copy with its own ``extra`` is enough. A deep copy would also
        copy tools, and with them the objects their bound methods belong to.
        """
        settings = copy.copy(self._settings)
        settings.extra = dict(self._settings.extra)
        return settings

    def _prism_view(self) -> tuple[str | None, list[r.ToolDefinition]]:
        """The instructions and tools Prism should hold.

        The instructions are the full system instruction ``LLMService`` composes,
        and the tools include the built-in ones, such as cancel tools.
        """
        return assert_given(self._settings.system_instruction), self._sdk_tools(self._settings)

    async def _apply_settings(self, delta: LLMSettings) -> None:
        """Apply a settings delta, keeping local settings matched to Prism's.

        ``LLMService`` applies the delta and recomposes the system instruction
        and tools, and Prism receives whatever then differs from what it holds.
        Prism can refuse an update, so a refusal restores the previous settings.
        """
        supplied = delta.given_fields()
        proposal = self._copy_settings()
        proposal.apply_update(self.Settings.from_mapping(supplied))
        self._validate_settings(proposal)
        self._sdk_tools(proposal)  # Raises for invalid tools or fillers.
        for name in _REQUIRED_FIELDS:
            if name in supplied and getattr(proposal, name) is None:
                raise ValueError(f"Prism cannot reset {name} to None")

        # Response metadata is sent apart from the session update, below.
        session_delta = self.Settings.from_mapping(
            {name: value for name, value in supplied.items() if name != "response_metadata"}
        )
        previous_settings = self._copy_settings()
        previous_base = self._base_system_instruction
        previous_unregistered = set(self._explicitly_unregistered_function_names)
        changed: dict[str, Any] = {}
        try:
            changed = await super()._update_settings(session_delta)
            # The base service syncs handlers to the context's tools on every
            # context frame, so the context changes with the handlers.
            if self._context is not None and "tools" in changed:
                self._context.set_tools(self._settings.tools or [])
            self._sync_registered_tool_handlers(self._settings.tools)
            instructions, tools = self._prism_view()
            update: dict[str, Any] = {
                sdk_name: getattr(self._settings, name)
                for name, sdk_name in _SESSION_FIELDS
                if name in changed
            }
            if instructions != self._prism_instructions:
                update["instructions"] = instructions or ""
            if tools != self._prism_tools:
                update["tools"] = tools
            rules = [name for name in _INTERRUPTION_RULES if name in changed]
            if rules:
                update["turn_detection"] = self._interruption_rules(self._settings, rules)
            if update:
                await self._sdk.update(**update)
        except Exception:
            self._settings = previous_settings
            self._base_system_instruction = previous_base
            self._explicitly_unregistered_function_names = previous_unregistered
            # The user aggregator may have set the refused tools on the context.
            if self._context is not None and "tools" in changed:
                self._context.set_tools(self._settings.tools or [])
            self._sync_registered_tool_handlers(self._settings.tools)
            raise
        if update:
            self._prism_instructions, self._prism_tools = instructions, tools
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
                if self._history is not None and self._history.lock.locked():
                    uncertain = PrismHistoryUpdateError("uncertain")
                    uncertain.__cause__ = error
                    error = uncertain
                await self._fail(error)

    async def _event(self, event: r.SessionEvent) -> None:
        payload = event.payload
        await self._call_event_handler("on_rime_event", event)
        if isinstance(payload, r.SpeechStarted):
            await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
        elif isinstance(payload, r.SpeechStopped):
            await self.start_ttfb_metrics()
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)
        elif isinstance(payload, (r.TranscriptDelta, r.TranscriptFinal)):
            final = isinstance(payload, r.TranscriptFinal)
            if final:
                text = payload.text
            elif payload.transcript is not None:
                # A full transcript replaces the partial text, revisions included.
                text = payload.transcript
            else:
                text = self._transcripts.get(payload.item_id, "") + payload.delta
            if len(text) > 65536 or len(self._transcripts) >= 128:
                raise RuntimeError("Prism transcript retention limit exceeded")
            if final:
                self._transcripts.pop(payload.item_id, None)
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
            response = self._conversation.start(self._sdk, payload)
            self._supersede_open_calls(payload.response.response_id)
            # A response that starts while content is being written may not
            # take it in, so the reply stays owed.
            if not self._writes_unwritten:
                self._reply_owed = False
                self._content_written = False
                self._settle_owed_calls()
            if self._request_pending and payload.cause in ("proactive", "tools"):
                if self._admitted_response is not None:
                    raise RuntimeError("Prism created multiple responses for one request")
                response.operations += 1
                self._admitted_response = response
            if self._playback:
                self._playback.start(payload.response)
            if payload.cause == "proactive":
                response.audio_response = self._reply_audio_response
                response.stopped = (
                    not self._reply_reports and self._reply_epoch != self._interruption_epoch
                )
            if payload.parent:
                parent = self._responses.get(payload.parent.response_id)
                response.stopped = parent is not None and parent.stopped
                if parent is not None:
                    response.audio_response = parent.audio_response
            if self._playback_mode == "discard":
                response.audio_response = False
            self._response_seen.set()
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
                await self.push_frame(TTSTextFrame(payload.text, text_type=TextType.SENTENCE))
            elif isinstance(payload, r.AudioDelta):
                if response.audio_response:
                    audio = payload.audio
                    if not self._playing or self._playing[-1] is not output:
                        self._playing.append(output)
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
            if response.tools_disabled:
                raise RuntimeError("Prism emitted a tool call during a tool-disabled reply")
            self._calls[payload.call.call_id] = Tool(payload)
            response.calls.add(payload.call.call_id)
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
            response.done.set()
            for output in response.outputs.values():
                await self._end_output(output)
            if payload.usage is not None:
                await self.start_llm_usage_metrics(
                    LLMTokenUsage(
                        prompt_tokens=payload.usage.input_tokens,
                        completion_tokens=payload.usage.output_tokens,
                        total_tokens=payload.usage.total_tokens,
                    )
                )
            if self._playback:
                self._playback.end(payload.response)
                self._spawn_request(self._playback.settle())
            self._conversation.release(response)
            await self.push_frame(LLMFullResponseEndFrame())
        elif isinstance(payload, r.FaultEvent) and payload.error.request_id is None:
            error = r.RimeRealtimeError(payload.error)
            if payload.error.scope == "session":
                await self._fail(error)
            elif payload.error.code == "tool_result_timeout" and any(
                tool is not None and not tool.final
                for tool in map(self._calls.get, payload.error.call_ids)
            ):
                # The application tool is still running. Prism records its
                # result when it arrives, and a reply can still report it.
                for tool in map(self._calls.get, payload.error.call_ids):
                    if tool is not None:
                        tool.superseded = True
                logger.debug(f"{self}: {error}")
            else:
                await self.push_error(str(error), exception=error)

    async def _end_output(self, output: _Output) -> None:
        if not output.closed:
            output.closed = True
            await self.push_frame(TTSStoppedFrame(context_id=output.context_id))

    async def _interrupt(self, output: _Output | None) -> None:
        if self._session is None:
            return
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
        # Report one elapsed-time estimate for the output playing now only.
        if output is not None and output.started_at is not None:
            elapsed = (time.monotonic() - output.started_at) * 1000
            await self._sdk.report_playback(
                r.PlaybackInterrupted(
                    output=output.ref,
                    audio_end_ms=int(min(elapsed, output.audio_ms)),
                )
            )

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Write each tool result to Prism as it is produced.

        The frames travel on to the assistant aggregator, which records them in
        the context and decides when Prism should reply.
        """
        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, (FunctionCallResultFrame, FunctionCallCancelFrame))
            and frame.tool_call_id in self._calls
        ):
            tool = self._calls[frame.tool_call_id]
            if not tool.final:
                final = (
                    not isinstance(frame, FunctionCallResultFrame)
                    or not frame.properties
                    or frame.properties.is_final
                )
                tool.final = final
                tool.report = final and self._reports_late_result(tool, frame)
                await self._enqueue(
                    _Operation(
                        tool=tool,
                        tool_output=self._tool_output(tool, frame, final=final),
                        tool_final=final,
                        tool_message=self._settling_message(tool, frame) if tool.progress else "",
                        answerable=not tool.report,
                    )
                )
                tool.progress = tool.progress or not final
                if not tool.report:
                    self._writes_unwritten += 1
                    self._reply_owed = True
                    response = self._responses.get(tool.event.call.response_id)
                    self._owed_audio_response = (
                        response.audio_response if response is not None else not self._skip_tts
                    )
        await super().push_frame(frame, direction)

    def _tool_output(
        self, tool: Tool, frame: FunctionCallResultFrame | FunctionCallCancelFrame, *, final: bool
    ) -> str:
        if isinstance(frame, FunctionCallCancelFrame):
            return "CANCELLED"
        result = (
            json.dumps(frame.result, ensure_ascii=False)
            if frame.result is not None
            else "COMPLETED"
        )
        if final:
            return result
        message = async_tool_messages.build_intermediate_result_message(
            tool.event.call.call_id, result
        )
        return str(message.get("content", ""))

    def _settling_message(
        self, tool: Tool, frame: FunctionCallResultFrame | FunctionCallCancelFrame
    ) -> str:
        call_id = tool.event.call.call_id
        if isinstance(frame, FunctionCallCancelFrame):
            message = async_tool_messages.build_cancelled_message(call_id)
        else:
            output = self._tool_output(tool, frame, final=True)
            message = async_tool_messages.build_final_result_message(call_id, output)
        return str(message.get("content", ""))

    def _reports_late_result(
        self, tool: Tool, frame: FunctionCallResultFrame | FunctionCallCancelFrame
    ) -> bool:
        """Whether an async call's final result needs a reply of its own.

        A round closes when a later reply starts or Prism's result deadline
        passes. Prism keeps a result that arrives after that in its history but
        mentions it only when asked, and the assistant aggregator drops the
        reply it holds for the result when the user interrupts.
        """
        if not isinstance(frame, FunctionCallResultFrame) or frame.result is None:
            return False
        run_llm = frame.run_llm
        if frame.properties and frame.properties.run_llm is not None:
            run_llm = frame.properties.run_llm
        entry = self._functions.get(tool.event.name) or self._functions.get(None)
        if run_llm is False or entry is None or entry.cancel_on_interruption:
            return False
        response = self._responses.get(tool.event.call.response_id)
        return tool.superseded or response is None or response.stopped

    def _supersede_open_calls(self, response_id: str) -> None:
        for tool in self._calls.values():
            if not tool.final and tool.event.call.response_id != response_id:
                tool.superseded = True

    async def _write_tool_update(
        self, tool: Tool, output: str, *, final: bool, message: str = ""
    ) -> None:
        call = tool.event.call
        if not final:
            # Prism keeps one result per call, so progress goes in beside the open call.
            await self._ledger.append([{"role": "developer", "content": output}])
            return
        try:
            async with self._ledger.lock:
                await self._sdk.submit_tool_result(call, output)
        except Exception as error:
            # An unsettled call leaves the conversation in an unknown state.
            await self._fail(error)
            raise
        tool.submitted = True
        response = self._responses.get(call.response_id)
        if response is not None:
            response.calls.discard(call.call_id)
            # A reply that reports the call needs the response reference.
            response.operations += 1
        if message:
            # Prism files the result beside its call, ahead of progress written
            # since, so the result is stated again after that progress.
            try:
                await self._ledger.append([{"role": "developer", "content": message}])
            except Exception as error:
                await self._operation_error(error)
        if tool.report:
            self._spawn_request(self._report_late_result(tool))
        else:
            self._owed_calls.append(tool)

    async def _report_late_result(self, tool: Tool) -> None:
        call = tool.event.call
        response = self._responses.get(call.response_id)
        audio_response = response.audio_response if response is not None else not self._skip_tts
        try:
            for attempt in range(2):
                try:
                    await self._locked_reply(
                        audio_response,
                        tool_call=call,
                        instruction=None,
                        disable_tools=self._tool_choice == "none",
                        parent=None,
                        report=True,
                    )
                    return
                except (r.RimeRealtimeError, r.RealtimeAdmissionTimeout) as error:
                    # The caller's speech, or a reply that stays busy, can push
                    # the report aside. It waits for Prism to be idle again.
                    if attempt or (
                        isinstance(error, r.RimeRealtimeError) and error.fault.scope == "session"
                    ):
                        raise
                    logger.debug(f"{self}: requesting the tool result report again: {error}")
        finally:
            if response is not None:
                response.operations -= 1
                self._conversation.release(response)

    async def _operation_error(self, error: Exception) -> None:
        if isinstance(error, PrismHistoryUpdateError) and error.recoverable:
            await self.push_error(str(error), exception=error)
        elif isinstance(error, (ValueError, TypeError, RimeInputError, r.RealtimeAdmissionTimeout)):
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
        if self._playback:
            self._playback.close()
        self._conversation.close()
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

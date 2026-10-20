#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Text generation with Dynamo's experimental realtime streaming-prefill API."""

import asyncio
from collections import deque
from copy import copy, deepcopy
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from loguru import logger

from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.frames.frames import (
    CancelFrame,
    EagerEndOfTurnCancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMUpdateSettingsFrame,
    StartFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.services.websocket_service import WS_CLOSE_TIMEOUT
from pipecat.utils.types import assert_given, is_given


class DynamoRealtimeLLMService(LLMService[OpenAILLMAdapter]):
    """Stream ASR text into Dynamo before requesting an LLM response.

    Uses Dynamo's experimental ``input_text.append/clear/commit`` extension to
    ``/v1/realtime``. These input events are not part of the OpenAI standard.
    Only text messages are supported; tools, multimodal input, and sampling
    overrides are not.

    Place :class:`DynamoStreamingPrefillProcessor` before the user context aggregator
    and this service after it. Final context remains authoritative: revised
    transcripts replace speculative input instead of being appended to it.
    Each turn uses one connection for both prefill and generation.

    Event handlers available (events may contain conversation text):

    - ``on_client_event(service, event)``: A realtime event was sent.
    - ``on_server_event(service, event)``: A realtime event was received.
    """

    Settings = LLMSettings
    adapter_class = OpenAILLMAdapter

    def __init__(
        self,
        *,
        base_url: str,
        settings: Settings,
        api_key: str | None = None,
        incremental_prefill: bool = True,
        timeout: float = 30.0,
        ws_close_timeout: float = WS_CLOSE_TIMEOUT,
        **kwargs,
    ):
        """Initialize the Dynamo client.

        Args:
            base_url: Dynamo HTTP base URL ending in ``/v1``, or the full
                ``ws(s)://.../v1/realtime`` URL. Reverse-proxy paths are kept.
            settings: Model, optional system instruction, and output token limit.
            api_key: Optional bearer token for an authenticated frontend.
            incremental_prefill: Whether to send interim transcripts for prefill.
                False sends only finalized context, using the same realtime API.
            timeout: Timeout in seconds for connection and each socket operation.
            ws_close_timeout: Maximum time in seconds for a closing handshake.
            **kwargs: Additional arguments passed to :class:`LLMService`.
        """
        defaults = self.Settings(
            model=None,
            system_instruction=None,
            max_tokens=None,
            temperature=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
        )
        defaults.apply_update(settings)
        self._validate_settings(defaults)
        super().__init__(settings=defaults, **kwargs)
        url = urlsplit(base_url)
        if url.scheme not in ("http", "https", "ws", "wss") or not url.netloc:
            raise ValueError("base_url must be an HTTP or WebSocket URL")
        path = url.path.rstrip("/")
        if not path.endswith("/realtime"):
            path += "/realtime" if path.endswith("/v1") else "/v1/realtime"
        scheme = {"http": "ws", "https": "wss"}.get(url.scheme, url.scheme)
        self._url = urlunsplit((scheme, url.netloc, path, url.query, ""))
        if timeout <= 0 or ws_close_timeout <= 0:
            raise ValueError("Socket timeouts must be positive")
        self._timeout = timeout
        self._ws_close_timeout = ws_close_timeout
        self._api_key = api_key
        self._incremental_prefill = incremental_prefill
        self._http: aiohttp.ClientSession | None = None
        self._websocket: aiohttp.ClientWebSocketResponse | None = None
        self._history: list[dict[str, str]] | None = None
        self._text = ""
        self._pending_text = ""
        self._prefill_task: asyncio.Task | None = None
        self._generation_task: asyncio.Task | None = None
        self._generation_queue: deque[tuple[LLMContextFrame, FrameDirection]] = deque()
        self._prefill_ready = asyncio.Event()
        self._connection_lock = asyncio.Lock()
        self._prefill_failed = False
        self._generating = False
        self._closing = False
        self._register_event_handler("on_client_event")
        self._register_event_handler("on_server_event")

    @staticmethod
    def _validate_settings(settings: LLMSettings) -> None:
        if not isinstance(settings.model, str) or not settings.model:
            raise ValueError("Dynamo requires a model in settings")
        supported = {"model", "system_instruction", "max_tokens"}
        for name, value in settings.given_fields().items():
            if name == "use_provider_thinking_default":
                # Dynamo leaves thinking configuration to the backend.
                continue
            if name == "filter_incomplete_user_turns" and value is False:
                continue
            if name not in supported and value is not None:
                raise ValueError(f"Dynamo realtime does not support the {name} setting")

    def can_generate_metrics(self) -> bool:
        """Return whether this service supports latency and token usage metrics."""
        return True

    async def _update_settings(self, delta: LLMSettings) -> dict[str, Any]:
        candidate = deepcopy(self._settings)
        candidate.apply_update(delta)
        self._validate_settings(candidate)
        await self._cancel_generation()
        async with self._connection_lock:
            await self._stop_prefill()
            await self._close()
            return await super()._update_settings(delta)

    async def _messages(self, context: LLMContext) -> list[dict[str, str]]:
        if is_given(context.tools) and context.tools:
            raise ValueError("Dynamo realtime supports text only, not tools")
        params = await self.get_llm_adapter().get_llm_invocation_params(
            context,
            system_instruction=assert_given(self._settings.system_instruction),
            convert_developer_to_user=False,
        )
        messages = []
        for item in params["messages"]:
            role, content = item.get("role"), item.get("content")
            if role not in ("system", "developer", "user", "assistant") or not isinstance(
                content, str
            ):
                raise ValueError("Dynamo realtime requires text-only messages")
            messages.append({"role": "system" if role == "developer" else role, "content": content})
        return messages

    def prefill(self, context: LLMContext, text: str) -> None:
        """Queue an append-only transcript delta without waiting on the network.

        Args:
            context: Conversation history, excluding the current user turn.
            text: New transcript text, not a cumulative ASR hypothesis.
        """
        if (
            not text
            or not self._incremental_prefill
            or self._generating
            or (self._generation_task is not None and not self._generation_task.done())
            or self._prefill_failed
            or self._closing
            or self._connection_lock.locked()
        ):
            return
        self._pending_text += text
        if self._prefill_task is None:
            snapshot = LLMContext(messages=deepcopy(context.get_messages()), tools=context.tools)
            self._prefill_task = self.create_task(self._prefill_loop(snapshot))
        self._prefill_ready.set()

    async def _prefill_loop(self, context: LLMContext) -> None:
        try:
            history = await self._messages(context)
            while True:
                await self._prefill_ready.wait()
                self._prefill_ready.clear()
                # Cancellation during I/O invalidates the socket; cancellation
                # while idle preserves completed prefill for finalization.
                try:
                    await self._open(history)
                    target = self._pending_text
                    delta = target[len(self._text) :]
                    if delta:
                        await self._send({"type": "input_text.append", "text": delta})
                        self._text = target
                except BaseException:
                    await self._close()
                    raise
        except Exception as exc:
            self._prefill_failed = True
            await self.push_error(f"Dynamo prefill failed: {exc}", exception=exc)

    async def _stop_prefill(self) -> None:
        task, self._prefill_task = self._prefill_task, None
        if task is not None:
            await self.cancel_task(task)
        self._pending_text = ""
        self._prefill_ready.clear()

    async def _send(self, event: dict[str, Any]) -> None:
        if self._websocket is None:
            raise RuntimeError("Dynamo realtime connection is closed")
        async with asyncio.timeout(self._timeout):
            await self._websocket.send_json(event)
        await self._call_event_handler("on_client_event", event)

    async def _receive(self) -> dict[str, Any]:
        if self._websocket is None:
            raise RuntimeError("Dynamo realtime connection is closed")
        async with asyncio.timeout(self._timeout):
            event = await self._websocket.receive_json()
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise ValueError("Invalid Dynamo realtime event")
        await self._call_event_handler("on_server_event", event)
        if event["type"] == "error":
            raise RuntimeError(f"Dynamo realtime error: {event.get('error')}")
        return event

    async def _open(self, history: list[dict[str, str]]) -> None:
        if self._websocket is not None:
            return
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        http = self._http = aiohttp.ClientSession(headers=headers)
        try:
            # Bound the entire handshake, not just each acknowledgement.
            async with asyncio.timeout(self._timeout):
                self._websocket = await http.ws_connect(self._url)
                if (await self._receive())["type"] != "session.created":
                    raise RuntimeError("Expected Dynamo session.created")
                session: dict[str, Any] = {
                    "type": "realtime",
                    "model": self._settings.model,
                    "output_modalities": ["text"],
                }
                if self._settings.max_tokens is not None:
                    session["max_output_tokens"] = self._settings.max_tokens
                await self._send({"type": "session.update", "session": session})
                if (await self._receive())["type"] != "session.updated":
                    raise RuntimeError("Expected Dynamo session.updated")
                for message in history:
                    await self._send_item(message)
                self._history = deepcopy(history)
        except BaseException:
            await self._close()
            raise

    async def _send_item(self, message: dict[str, str]) -> None:
        await self._send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": message["role"],
                    "content": [
                        {
                            "type": "output_text"
                            if message["role"] == "assistant"
                            else "input_text",
                            "text": message["content"],
                        }
                    ],
                },
            }
        )

    async def _close(self) -> None:
        websocket, self._websocket = self._websocket, None
        http, self._http = self._http, None
        self._history = None
        self._text = ""
        try:
            if websocket is not None:
                async with asyncio.timeout(self._ws_close_timeout):
                    await websocket.close()
        except TimeoutError:
            logger.debug(f"{self}: Dynamo websocket close timed out; dropping connection")
        finally:
            if http is not None:
                await http.close()

    async def _process_context(self, context: LLMContext) -> None:
        async with self._connection_lock:
            await self._generate(context)

    async def _generate(self, context: LLMContext) -> None:
        self._generating = True
        try:
            # Do not flush queued speculation after the turn has ended. The
            # finalized request can reuse work already done without another prefill.
            await self._stop_prefill()
            messages = await self._messages(context)
            if not messages or messages[-1]["role"] != "user" or not messages[-1]["content"]:
                raise ValueError("Dynamo realtime requires a final user text message")
            history, text = messages[:-1], messages[-1]["content"]
            if self._history is not None and self._history != history:
                await self._close()
            await self._open(history)
            if self._text and self._text == text:
                await self._send({"type": "input_text.commit"})
            else:
                if self._text:
                    await self._send({"type": "input_text.clear"})
                await self._send_item(messages[-1])
            await self._send({"type": "response.create"})
            while True:
                event = await self._receive()
                if event["type"] == "response.output_text.delta" and event.get("delta"):
                    await self.stop_ttfb_metrics()
                    await self._push_llm_text(event["delta"])
                elif event["type"] == "response.done":
                    response = event["response"]
                    if response["status"] not in ("completed", "incomplete"):
                        raise RuntimeError(
                            f"Dynamo response failed: {response.get('status_details')}"
                        )
                    if usage := response.get("usage"):
                        await self.start_llm_usage_metrics(
                            LLMTokenUsage(
                                prompt_tokens=usage["input_tokens"],
                                completion_tokens=usage["output_tokens"],
                                total_tokens=usage["total_tokens"],
                            )
                        )
                    break
        finally:
            try:
                await self._close()
            finally:
                self._generating = False
                self._prefill_failed = False

    async def _generation_loop(self) -> None:
        while self._generation_queue:
            frame, direction = self._generation_queue.popleft()
            # Initialize the speculation gate only when this response starts.
            await super().process_frame(frame, direction)
            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()
            try:
                await self._process_context(frame.context)
            except TimeoutError as exc:
                await self._call_event_handler("on_completion_timeout")
                await self.push_error("Dynamo completion timeout", exception=exc)
            except Exception as exc:
                await self.push_error(f"Dynamo completion failed: {exc}", exception=exc)
            finally:
                await self.stop_ttfb_metrics()
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())

    async def _cancel_generation(self) -> None:
        self._generation_queue.clear()
        if self._generation_task is not None:
            await self.cancel_task(self._generation_task)
            self._generation_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Queue finalized context for generation and forward other frames."""
        if isinstance(frame, LLMContextFrame):
            if not self._closing:
                snapshot = copy(frame)
                snapshot.context = LLMContext(
                    messages=deepcopy(frame.context.get_messages()), tools=frame.context.tools
                )
                self._generation_queue.append((snapshot, direction))
                if self._generation_task is None or self._generation_task.done():
                    self._generation_task = self.create_task(self._generation_loop())
            return

        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMUpdateSettingsFrame):
            await self.push_frame(frame, direction)

    async def _handle_interruptions(self, frame: InterruptionFrame):
        await self._cancel_generation()
        async with self._connection_lock:
            await super()._handle_interruptions(frame)
            await self._stop_prefill()
            await self._close()
            self._prefill_failed = False

    async def _handle_eager_end_of_turn_cancel(self, frame: EagerEndOfTurnCancelFrame) -> None:
        if self._speculation_gate.is_speculating or any(
            frame.speculation for frame, _ in self._generation_queue
        ):
            await self._cancel_generation()
        await super()._handle_eager_end_of_turn_cancel(frame)

    async def start(self, frame: StartFrame):
        """Enable speculative input when the pipeline starts."""
        await super().start(frame)
        self._closing = False

    async def stop(self, frame: EndFrame):
        """Finish queued responses and close the realtime connection on shutdown."""
        self._closing = True
        try:
            task = self._generation_task
            if task is not None:
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    if not task.cancelled():
                        raise
        finally:
            await self._cancel_generation()
            await self._stop_prefill()
            await self._close()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel pending responses and prefill, then close the realtime connection."""
        self._closing = True
        await self._cancel_generation()
        await self._stop_prefill()
        await self._close()
        await super().cancel(frame)

    async def cleanup(self):
        """Release connections even when teardown skips normal lifecycle frames."""
        self._closing = True
        try:
            await super().cleanup()
        finally:
            await self._cancel_generation()
            await self._stop_prefill()
            await self._close()


class DynamoStreamingPrefillProcessor(FrameProcessor):
    """Forward append-only ASR deltas to Dynamo before user context aggregation.

    Use with :class:`~pipecat.services.openai.stt.OpenAIRealtimeSTTService`,
    which emits transcript deltas. Cumulative or revisable interim hypotheses
    from other STT providers must first be converted to append-only deltas.
    Final context always determines the actual prompt used for generation.
    """

    def __init__(self, *, llm: DynamoRealtimeLLMService, context: LLMContext, **kwargs):
        """Initialize the streaming prefill processor.

        Args:
            llm: Dynamo LLM service later in the same pipeline.
            context: Shared context managed by the user and assistant aggregators.
            **kwargs: Additional arguments passed to :class:`FrameProcessor`.
        """
        super().__init__(**kwargs)
        self._llm = llm
        self._context = context
        self._muted = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Queue ASR partials without delaying transcription or turn detection."""
        await super().process_frame(frame, direction)
        if isinstance(frame, (UserMuteStartedFrame, UserMuteStoppedFrame)):
            self._muted = isinstance(frame, UserMuteStartedFrame)
        elif (
            not self._muted
            and direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, InterimTranscriptionFrame)
        ):
            self._llm.prefill(self._context, frame.text)
        await self.push_frame(frame, direction)

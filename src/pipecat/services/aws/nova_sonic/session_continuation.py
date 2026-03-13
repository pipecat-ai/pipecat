#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Session continuation helper for AWS Nova Sonic.

Nova Sonic sessions have an AWS-imposed time limit (~8 minutes). This module
provides transparent session continuation that rotates sessions in the background
before the limit is reached, preserving conversation context with no
user-perceptible interruption.

Implementation follows the AWS reference architecture:
https://github.com/aws-samples/amazon-nova-samples/tree/main/speech-to-speech/amazon-nova-2-sonic/repeatable-patterns/session-continuation/console-python
"""

import asyncio
import time
from collections import deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from loguru import logger
from pydantic import BaseModel, Field

_MAX_HISTORY_MESSAGE_BYTES = 50 * 1024  # 50 KB per message
_MAX_HISTORY_TOTAL_BYTES = 200 * 1024  # 200 KB total history


@runtime_checkable
class NovaSonicSessionSender(Protocol):
    """Protocol for sending events to a Nova Sonic session stream.

    The LLM service implements this to expose the Nova Sonic wire protocol to
    the session continuation helper without coupling the helper to service
    internals (audio config, voice, model, etc.). Each method targets an
    explicit ``stream`` / ``prompt_name`` so the same implementation can write
    to either the current session or the next (pre-created) session.
    """

    async def open_stream(self, client: Any) -> Any:
        """Open a bidirectional stream on the given client."""
        ...

    async def send_event(self, event_json: str, stream: Any) -> None:
        """Send a raw event JSON string to the given stream."""
        ...

    def build_session_start_json(self) -> str:
        """Build the ``sessionStart`` event JSON string."""
        ...

    async def send_prompt_start(self, tools: list, prompt_name: str, stream: Any) -> None:
        """Send a ``promptStart`` event to the given stream."""
        ...

    async def send_text(
        self, text: str, role: str, prompt_name: str, stream: Any, interactive: bool
    ) -> None:
        """Send a text content block (contentStart/textInput/contentEnd) to the given stream."""
        ...

    async def send_audio_input_start(
        self, prompt_name: str, content_name: str, stream: Any
    ) -> None:
        """Send an audio input ``contentStart`` to the given stream."""
        ...

    async def send_audio(
        self, audio: bytes, prompt_name: str, content_name: str, stream: Any
    ) -> None:
        """Send an ``audioInput`` event to the given stream."""
        ...

    def create_client(self) -> Any:
        """Create a new Bedrock runtime client."""
        ...

    @property
    def audio_config(self) -> Any:
        """Return the audio configuration (``AudioConfig`` instance)."""
        ...

    def get_setup_params(self) -> "tuple[str | None, list]":
        """Return ``(system_instruction, tools)`` for the next session setup."""
        ...


class SessionContinuationParams(BaseModel):
    """Configuration for automatic session continuation.

    Nova Sonic sessions have an AWS-imposed time limit (~8 minutes). When enabled,
    session continuation proactively creates a new session in the background before
    the limit is reached, buffers user audio during the transition, and seamlessly
    hands off — preserving conversation context with no user-perceptible gap.

    Parameters:
        enabled: Whether automatic session continuation is enabled.
        transition_threshold_seconds: How many seconds into a session to begin
            monitoring for a transition opportunity. The transition will occur
            when the assistant next starts speaking after this threshold.
        audio_buffer_duration_seconds: Duration of the rolling audio buffer
            (in seconds) that captures user audio during the transition window.
            This audio is replayed into the new session so no user input is lost.
        audio_start_timeout_seconds: Maximum time to wait for the assistant to
            start speaking after the threshold is reached. If no assistant audio
            arrives within this window, the transition is forced. Set to 0 to
            disable the timeout (wait indefinitely).
    """

    enabled: bool = Field(default=True)
    transition_threshold_seconds: float = Field(default=360.0)
    audio_buffer_duration_seconds: float = Field(default=3.0)
    audio_start_timeout_seconds: float = Field(default=80.0)


@dataclass
class _NextSession:
    """Holds pre-created resources for the next session during a transition."""

    client: Any  # BedrockRuntimeClient
    stream: Any  # DuplexEventStream
    prompt_name: str
    input_audio_content_name: str


class SessionContinuationHelper:
    """Manages proactive session rotation for Nova Sonic.

    This helper encapsulates all session continuation state and logic, providing
    a clean API for the LLM service to integrate with. It delegates stream I/O
    back to the LLM service via callbacks.

    The LLM service hooks into this helper at key points:
    - ``on_audio_input(audio)``: called for each user audio frame
    - ``on_assistant_audio_started()``: called on AUDIO contentStart from assistant
    - ``on_assistant_text_output(role, text, stage)``: called on textOutput events
    - ``on_content_end(role, content_type, stop_reason, text_content, text_stage)``:
      called on contentEnd events
    - ``on_completion_end()``: called on completionEnd events
    - ``on_user_content_started()``: called on USER contentStart events
    """

    def __init__(
        self,
        params: SessionContinuationParams,
        *,
        sender: NovaSonicSessionSender,
        create_task: Callable[[Coroutine], asyncio.Task],
        cancel_task: Callable[[asyncio.Task, float], Coroutine[Any, Any, None]],
    ):
        """Initialize the session continuation helper.

        Args:
            params: Configuration for session continuation behavior.
            sender: Object implementing the ``NovaSonicSessionSender`` protocol.
                The LLM service provides this to expose Nova Sonic wire I/O
                without coupling the helper to service internals. Audio
                configuration is read from ``sender.audio_config``.
            create_task: Callable to spawn a task managed by the service's task
                manager (typically ``self.create_task`` from the LLM service).
            cancel_task: Callable to cancel a task (typically
                ``self.cancel_task`` from the LLM service).
        """
        self._params = params
        self._sender = sender
        self._create_task = create_task
        self._cancel_task = cancel_task

        # Audio buffer — sized from the sender's audio config
        ac = sender.audio_config
        self._audio_buffer: deque[bytes] = deque()
        self._audio_buffer_max_bytes: int = int(
            params.audio_buffer_duration_seconds
            * ac.input_sample_rate
            * (ac.input_sample_size / 8)
            * ac.input_channel_count
        )

        # Transition state
        self._next_session: _NextSession | None = None
        self._is_buffering = False
        self._waiting_for_audio_start = False
        self._waiting_for_completion = False
        self._handoff_in_progress = False
        self._audio_start_wait_time: float | None = None
        self._next_session_created_time: float | None = None
        self._monitor_task: asyncio.Task | None = None
        self._connected_time: float | None = None
        self._wants_connection = False

        # Session-level text counters — always incremented, never gated.
        # Matches reference: counts live on SessionInfo from session start.
        self._speculative_text_count = 0
        self._final_text_count = 0
        self._barge_in_detected = False

        # Conversation history — tracked in real-time from FINAL text events.
        # TODO: Integrate with pipecat's LLMContext once the pipeline supports
        # synchronous context reads or a flush mechanism.
        self._conversation_history: list[dict[str, str]] = []

    # --- Public API for the LLM service ---

    @property
    def is_buffering(self) -> bool:
        """Whether user audio is currently being buffered for the transition."""
        return self._is_buffering

    @property
    def next_session(self) -> _NextSession | None:
        """The pre-created next session, if any."""
        return self._next_session

    @property
    def handoff_in_progress(self) -> bool:
        """Whether a handoff is currently in progress."""
        return self._handoff_in_progress

    def set_connected(self, connected_time: float):
        """Called when the current session finishes connecting.

        Resets session-level counters. In the reference these live on
        SessionInfo and are zero-initialized per session.
        """
        self._connected_time = connected_time
        self._wants_connection = True
        self._speculative_text_count = 0
        self._final_text_count = 0
        self._barge_in_detected = False

    def set_disconnected(self):
        """Called when the current session disconnects."""
        self._wants_connection = False
        self._connected_time = None

    def seed_history(self, role: str, text: str):
        """Seed conversation history with initial context messages."""
        if text:
            self._conversation_history.append({"role": role, "text": text})

    def start_monitor(self):
        """Start the session duration monitor."""
        if not self._params.enabled or self._monitor_task:
            return
        self._monitor_task = self._create_task(self._monitor_loop())

    async def stop_monitor(self):
        """Stop the session duration monitor."""
        if self._monitor_task:
            await self._cancel_task(self._monitor_task, 1.0)
            self._monitor_task = None

    def on_audio_input(self, audio: bytes):
        """Called for each user audio frame. Buffers audio during transition."""
        if self._is_buffering:
            self._audio_buffer.append(audio)
            total = sum(len(c) for c in self._audio_buffer)
            while total > self._audio_buffer_max_bytes and self._audio_buffer:
                removed = self._audio_buffer.popleft()
                total -= len(removed)

    async def on_assistant_audio_started(self):
        """Called when assistant AUDIO contentStart is detected.

        Starts buffering and creates the next session if we're past the threshold.
        Returns True if session continuation was triggered.
        """
        if not self._waiting_for_audio_start or self._handoff_in_progress:
            return False

        self._waiting_for_audio_start = False
        self._audio_start_wait_time = None
        self._is_buffering = True
        self._waiting_for_completion = True

        logger.info(
            "Session continuation: assistant audio started, "
            "buffering user audio and creating next session"
        )

        if not self._next_session:
            try:
                await self._prepare_next_session()
            except Exception as e:
                logger.error(f"Session continuation: failed to prepare next session: {e}")
                self._is_buffering = False
                self._waiting_for_completion = False
                return False

        return True

    def on_text_output(self, role: str, stage: str | None):
        """Called on textOutput events. Always tracks speculative/final counts.

        Matches reference: counts are session-level, always incremented for
        ASSISTANT text regardless of transition state. The completion check
        (in on_content_end_assistant_final_text) gates on _waiting_for_completion.
        """
        if role != "ASSISTANT":
            return

        if stage == "SPECULATIVE":
            self._speculative_text_count += 1
            logger.debug(f"Session continuation: SPECULATIVE text #{self._speculative_text_count}")
        elif stage == "FINAL":
            self._final_text_count += 1
            logger.debug(
                f"Session continuation: FINAL text #{self._final_text_count} "
                f"(speculative={self._speculative_text_count})"
            )

    def on_content_end_assistant_final_text(self, text: str | None):
        """Called on contentEnd for ASSISTANT FINAL TEXT (non-interrupted).

        Adds text to history and checks for completion signal.
        Returns True if handoff should be triggered.
        """
        if text:
            self._conversation_history.append({"role": "ASSISTANT", "text": text})

        # Check completion signal after adding to history
        if (
            self._waiting_for_completion
            and self._speculative_text_count > 0
            and self._final_text_count > 0
            and self._final_text_count >= self._speculative_text_count
            and not self._handoff_in_progress
        ):
            logger.info(
                f"Session continuation: completion signal — text pairs matched "
                f"(final={self._final_text_count} >= speculative={self._speculative_text_count})"
            )
            self._waiting_for_completion = False
            return True
        return False

    def on_content_end_text_interrupted(self) -> bool:
        """Called on contentEnd for TEXT with stopReason=INTERRUPTED.

        Marks barge-in detected. If we're waiting for completion, triggers
        handoff immediately (matches reference lines 650-654).
        Returns True if handoff should be triggered.
        """
        self._barge_in_detected = True
        if self._waiting_for_completion and not self._handoff_in_progress:
            logger.info("Session continuation: completion signal — TEXT INTERRUPTED (barge-in)")
            self._waiting_for_completion = False
            return True
        return False

    def on_content_end_user_final_text(self, text: str | None):
        """Called on contentEnd for USER FINAL TEXT. Adds to history.

        Also handles barge-in count reconciliation: when the user speaks after
        a barge-in, remaining FINAL texts for the interrupted response will
        never arrive. Force final = speculative so the counts match.
        Matches reference lines 602-609.
        """
        if text:
            self._conversation_history.append({"role": "USER", "text": text})

        if self._barge_in_detected and self._speculative_text_count > self._final_text_count:
            logger.info(
                f"Session continuation: user spoke after barge-in — "
                f"setting final={self._speculative_text_count} (was {self._final_text_count})"
            )
            self._final_text_count = self._speculative_text_count

    def on_user_content_started(self) -> bool:
        """Called on USER contentStart during transition.

        Marks barge-in during transition (matches reference lines 527-534).
        Returns True if handoff should be triggered (forced transition, no
        assistant response yet — matches reference lines 579-583).
        """
        if self._waiting_for_completion and self._next_session:
            self._barge_in_detected = True

        if (
            self._waiting_for_completion
            and not self._handoff_in_progress
            and self._next_session
            and self._final_text_count == 0
        ):
            logger.info(
                "Session continuation: user spoke during forced transition "
                "(no assistant response yet) — completing handoff immediately"
            )
            self._waiting_for_completion = False
            return True
        return False

    def on_completion_end(self) -> bool:
        """Called on completionEnd. Fallback completion signal.

        Returns True if handoff should be triggered.
        """
        if self._waiting_for_completion and not self._handoff_in_progress:
            logger.info("Session continuation: completion signal — completionEnd (fallback)")
            self._waiting_for_completion = False
            return True
        return False

    async def execute_handoff(self) -> _NextSession | None:
        """Execute the session handoff.

        Sends conversation history + audioInputStart + buffered audio to the next
        session. Returns (old_client, old_stream, old_receive_task, old_prompt_name)
        for the caller to swap and clean up, or None if handoff couldn't proceed.
        """
        if self._handoff_in_progress:
            return None
        self._handoff_in_progress = True

        try:
            ns = self._next_session
            if not ns:
                logger.warning("Session continuation: no next session available for handoff")
                return None

            logger.info("Session continuation: executing handoff")

            # Build trimmed history: walk backwards to prioritize recent
            # messages, truncate individual messages, and cap total size.
            prepared: list[dict[str, str]] = []
            total_bytes = 0
            for msg in reversed(self._conversation_history):
                text = msg["text"]
                encoded = text.encode("utf-8")
                if len(encoded) > _MAX_HISTORY_MESSAGE_BYTES:
                    encoded = encoded[:_MAX_HISTORY_MESSAGE_BYTES]
                    text = encoded.decode("utf-8", errors="ignore")
                    encoded = text.encode("utf-8")
                msg_bytes = len(encoded)
                if total_bytes + msg_bytes > _MAX_HISTORY_TOTAL_BYTES:
                    logger.debug(
                        f"Session continuation: dropping older history to fit "
                        f"{_MAX_HISTORY_TOTAL_BYTES} byte limit "
                        f"(total_bytes={total_bytes}, msg_bytes={msg_bytes})"
                    )
                    break
                total_bytes += msg_bytes
                prepared.append({"role": msg["role"], "text": text})
            prepared.reverse()

            # Ensure history starts with a USER message
            while prepared and prepared[0]["role"] != "USER":
                dropped = prepared.pop(0)
                logger.debug(
                    f"Session continuation: dropping leading {dropped['role']} message from history"
                )

            # Send conversation history
            if prepared:
                logger.info(
                    f"Session continuation: sending {len(prepared)} history "
                    f"messages ({total_bytes} bytes) to new session"
                )
                for msg in prepared:
                    logger.debug(
                        f"Session continuation: history [{msg['role']}]: "
                        f"{msg['text'][:80]}{'...' if len(msg['text']) > 80 else ''}"
                    )
                    await self._sender.send_text(
                        msg["text"], msg["role"], ns.prompt_name, ns.stream, False
                    )

            # Send audioInputStart
            await self._sender.send_audio_input_start(
                ns.prompt_name, ns.input_audio_content_name, ns.stream
            )

            # Send buffered audio
            buffer_chunks = list(self._audio_buffer)
            ac = self._sender.audio_config
            bytes_per_second = (
                ac.input_sample_rate * (ac.input_sample_size / 8) * ac.input_channel_count
            )
            buffer_duration = sum(len(c) for c in buffer_chunks) / bytes_per_second
            logger.info(
                f"Session continuation: sending {len(buffer_chunks)} buffered audio chunks "
                f"(~{buffer_duration:.1f}s) to new session"
            )
            for chunk in buffer_chunks:
                await self._sender.send_audio(
                    chunk, ns.prompt_name, ns.input_audio_content_name, ns.stream
                )

            # Return the next session info for the caller to promote
            logger.info("Session continuation: promoting new session")
            result = ns
            self._next_session = None
            self._is_buffering = False
            self._audio_buffer.clear()

            return result

        except Exception as e:
            logger.error(f"Session continuation: handoff error: {e}")
            await self.cleanup_next_session()
            return None
        finally:
            self._handoff_in_progress = False
            self._waiting_for_audio_start = False
            self._waiting_for_completion = False
            self._audio_start_wait_time = None
            self._next_session_created_time = None
            # Note: speculative/final counts and barge_in_detected are NOT
            # reset here — they are session-level and get reset in
            # set_connected() when the new session starts.

    async def close_old_session(
        self, client, stream, receive_task, prompt_name, input_audio_content_name=None
    ):
        """Close the old session's resources in the background.

        Audio input to the old stream is already stopped (handoff_in_progress
        gate in _handle_input_audio_frame). Sends contentEnd (audio) →
        promptEnd → sessionEnd → closes stream → cancels receive task.
        The receive task is cancelled last as a safety net to avoid leaks;
        by that point the stream is closed so the CRT future should already
        be resolved.
        """
        try:
            if stream and prompt_name:
                try:
                    import json

                    if input_audio_content_name:
                        audio_content_end_json = json.dumps(
                            {
                                "event": {
                                    "contentEnd": {
                                        "promptName": prompt_name,
                                        "contentName": input_audio_content_name,
                                    }
                                }
                            }
                        )
                        await self._sender.send_event(audio_content_end_json, stream)

                    prompt_end_json = json.dumps(
                        {"event": {"promptEnd": {"promptName": prompt_name}}}
                    )
                    session_end_json = json.dumps({"event": {"sessionEnd": {}}})
                    await self._sender.send_event(prompt_end_json, stream)
                    await self._sender.send_event(session_end_json, stream)
                except Exception:
                    pass

            if stream:
                try:
                    await asyncio.wait_for(stream.input_stream.close(), timeout=2.0)
                except (TimeoutError, Exception):
                    pass

            # Wait for the receive task to exit naturally (the stream is
            # closed, so it will hit an error or the stale-stream check).
            # Do NOT cancel — that cancels the underlying CRT future and
            # races with native set_result() callbacks, producing an
            # InvalidStateError traceback we can't catch from Python.
            if receive_task:
                try:
                    await asyncio.wait_for(asyncio.shield(receive_task), timeout=5.0)
                except (TimeoutError, Exception):
                    pass

            logger.debug("Session continuation: old session closed")
        except Exception as e:
            logger.warning(f"Session continuation: error closing old session: {e}")

    async def cleanup_next_session(self):
        """Clean up the pre-created next session if it exists."""
        ns = self._next_session
        if not ns:
            return

        if ns.stream:
            try:
                await ns.stream.close()
            except Exception:
                pass

        self._next_session = None
        self._is_buffering = False
        self._audio_buffer.clear()
        self._next_session_created_time = None

    # --- Internal methods ---

    async def _monitor_loop(self):
        """Periodically check session age and manage next session lifecycle."""
        try:
            while self._wants_connection:
                await asyncio.sleep(1)

                if not self._connected_time or self._handoff_in_progress:
                    continue

                session_age = time.time() - self._connected_time
                threshold = self._params.transition_threshold_seconds

                # Threshold reached — start waiting for assistant audio
                if (
                    session_age >= threshold
                    and not self._waiting_for_audio_start
                    and not self._next_session
                    and not self._waiting_for_completion
                ):
                    logger.info(
                        f"Session continuation: session age {session_age:.0f}s >= "
                        f"threshold {threshold:.0f}s, waiting for assistant audio"
                    )
                    self._waiting_for_audio_start = True
                    self._audio_start_wait_time = time.time()

                # Audio start timeout — force transition
                audio_start_timeout = self._params.audio_start_timeout_seconds
                if (
                    self._waiting_for_audio_start
                    and self._audio_start_wait_time
                    and audio_start_timeout > 0
                    and (time.time() - self._audio_start_wait_time) > audio_start_timeout
                ):
                    logger.info(
                        f"Session continuation: TIMEOUT — no assistant audio after "
                        f"{audio_start_timeout:.0f}s, forcing transition"
                    )
                    self._waiting_for_audio_start = False
                    self._audio_start_wait_time = None
                    self._is_buffering = True
                    self._waiting_for_completion = False
                    try:
                        if not self._next_session:
                            await self._prepare_next_session()
                        self._create_task(self.execute_handoff())
                    except Exception as e:
                        logger.error(f"Session continuation: forced transition failed: {e}")
                        self._is_buffering = False

                # Dead session detection — recreate if idle too long
                next_session_timeout = 30.0
                if (
                    self._next_session
                    and self._next_session_created_time
                    and not self._handoff_in_progress
                    and (time.time() - self._next_session_created_time) > next_session_timeout
                ):
                    logger.warning(
                        f"Session continuation: next session idle for "
                        f">{next_session_timeout:.0f}s, recreating"
                    )
                    await self.cleanup_next_session()
                    try:
                        await self._prepare_next_session()
                    except Exception as e:
                        logger.error(f"Session continuation: failed to recreate next session: {e}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Session continuation monitor error: {e}")

    async def _prepare_next_session(self):
        """Create a new session and send setup (sessionStart + promptStart + system instruction).

        Conversation history and audio are deferred to handoff time.
        """
        import uuid

        prompt_name = str(uuid.uuid4())
        input_audio_content_name = str(uuid.uuid4())

        client = self._sender.create_client()
        stream = await self._sender.open_stream(client)

        self._next_session = _NextSession(
            client=client,
            stream=stream,
            prompt_name=prompt_name,
            input_audio_content_name=input_audio_content_name,
        )
        self._next_session_created_time = time.time()

        ns = self._next_session

        # Send sessionStart
        await self._sender.send_event(self._sender.build_session_start_json(), ns.stream)

        # Get setup params: (system_instruction, tools)
        system_instruction, tools = self._sender.get_setup_params()

        # Send promptStart with tools
        await self._sender.send_prompt_start(tools, ns.prompt_name, ns.stream)

        # Send system instruction
        if system_instruction:
            await self._sender.send_text(
                system_instruction, "SYSTEM", ns.prompt_name, ns.stream, False
            )

        logger.debug(f"Session continuation: next session prepared (prompt={prompt_name[:8]}...)")

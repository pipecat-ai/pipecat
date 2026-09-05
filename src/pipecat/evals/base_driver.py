#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The base driver: what the user says next, and how the outcome is judged.

A :class:`BaseEvalDriver` runs the conversation with the bot over the session's
runtime (the client's pipeline, the event stream, the trace) and assembles the
run's result. :class:`~pipecat.evals.script_driver.EvalScriptDriver` plays a scenario's
``turns:`` and matches each turn's expectations;
:class:`~pipecat.evals.simulation_driver.EvalSimulationDriver` lets the persona LLM
in the pipeline hold the conversation and judges the whole of it.
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

from pipecat.evals.client import EvalClient
from pipecat.evals.events import EvalEventStream
from pipecat.evals.judge import EvalJudge
from pipecat.evals.results import EvalAssertionFailure, EvalProgress, EvalTrace

R = TypeVar("R")


class BaseEvalDriver(ABC, Generic[R]):
    """Base class for the drivers: drives the conversation and scores it.

    The runtime is shared, the client sends and the stream receives, and a
    driver decides what to send next and what counts as success. Subclasses
    implement :meth:`run` and :meth:`result`. The user-turn primitives here,
    :meth:`_say` and :meth:`_press`, keep the judge's transcript and the
    stream's turn bookkeeping consistent whichever driver sends.
    """

    def __init__(
        self,
        *,
        client: EvalClient,
        stream: EvalEventStream,
        judge: EvalJudge | None,
        trace: EvalTrace,
        progress: Callable[[EvalProgress], Awaitable[None]],
    ):
        """Initialize the driver.

        Args:
            client: The connection to the bot, for the user's sends.
            stream: The bot's output as events.
            judge: The judge for ``eval:`` assertions, or ``None``; the user's
                turns are added to its conversation so replies are judged in
                context.
            trace: The run's trace.
            progress: Awaited with an :class:`~pipecat.evals.results.EvalProgress`
                record as the conversation advances.
        """
        self._client = client
        self._stream = stream
        self._judge = judge
        self._trace = trace
        self._progress = progress

    @abstractmethod
    async def run(self) -> list[EvalAssertionFailure]:
        """Drive the conversation to its end and return the failures."""

    @abstractmethod
    def result(
        self,
        *,
        failures: list[EvalAssertionFailure],
        duration_ms: int,
        events_seen: list[dict],
        debug_log: list[str],
        skipped: str | None = None,
    ) -> R:
        """Assemble the run's result from what the driver scored and the session saw.

        Args:
            failures: The run's failures: the driver's own, plus the session's
                (a failed connect or handshake, a harness error).
            duration_ms: Wall-clock time the run took.
            events_seen: Every event observed, for diagnostics.
            debug_log: The run's trace.
            skipped: Why the run was not driven at all, or ``None``.
        """

    def record_failure(self, failure: EvalAssertionFailure) -> None:
        """Note a run-level failure the session raised while the driver was running.

        Args:
            failure: The failure, scored against the trace's current turn.
        """

    async def _say(self, text: str, *, audio_file: str | None = None) -> None:
        """Send one user utterance to the bot.

        The utterance goes out as the recording in ``audio_file`` when given,
        spoken by the user TTS when the client has one, else as text. Bot output
        still queued from an earlier turn is dropped first, so nothing the bot
        said before this input can be matched as its reply. The drop has to
        precede the send: once the input reaches the bot, its reaction to this
        very input would be dropped along with the stale output.

        Args:
            text: What the user says; also recorded in the judge's conversation
                so a later reply is judged in context (e.g. a terse "That's
                four" answering this question).
            audio_file: Optional recording to play in place of synthesizing
                ``text``.
        """
        self._stream.drop_pending_bot_output("before send")
        how = audio_file or ("audio" if self._client.has_user_tts else "text")
        self._trace.log(f"send: {text!r} ({how})")
        if audio_file is not None:
            await self._client.play(audio_file)
        elif self._client.has_user_tts:
            await self._client.say(text)
        else:
            await self._client.send_text(text)
        if self._judge is not None:
            self._judge.add_user_message(text)
        # Only what the bot says in reply to this input is matched from here on.
        self._stream.input_sent()

    async def _press(self, keys: str) -> None:
        """Send DTMF keypresses as the user's turn (see :meth:`_say` for the drop).

        Args:
            keys: The keys to press, in order; recorded for the judge so the
                bot's reply is judged knowing what was pressed.
        """
        self._stream.drop_pending_bot_output("before send")
        self._trace.log(f"send: dtmf {keys!r}")
        await self._client.send_dtmf(keys)
        if self._judge is not None:
            self._judge.add_user_message(f"(DTMF keypad input: {keys})")
        self._stream.input_sent()

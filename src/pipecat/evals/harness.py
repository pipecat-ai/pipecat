#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Eval session: drives a bot over RTVI and asserts on the events it emits.

An :class:`EvalSession` connects to a running bot's eval transport (a
``SingleClientWebsocketServerTransport`` speaking RTVI via
:class:`~pipecat.evals.serializer.RTVIEvalSerializer`), walks through a parsed
:class:`~pipecat.evals.scenario.EvalScenario`, and verifies that the expected
semantic events arrive in order, with the right payloads, within their latency
budgets. It returns an :class:`EvalResult`.

The session is a thin RTVI client. It builds outgoing messages with the RTVI
models (:mod:`pipecat.processors.frameworks.rtvi.models`) and translates the
RTVI server messages it receives back into a small set of friendly event names
the scenario files assert on:

==========================      ==============================================
scenario ``event:``             RTVI server message(s)
==========================      ==============================================
``user_started_speaking``       ``user-started-speaking``
``user_stopped_speaking``       ``user-stopped-speaking``
``vad_user_started_speaking``   ``vad-user-started-speaking`` (raw VAD, ungated by turn detection)
``vad_user_stopped_speaking``   ``vad-user-stopped-speaking`` (raw VAD, ungated by turn detection)
``user_transcription``          ``user-transcription`` (final only)
``llm_started``                 ``bot-llm-started``
``llm_response``                the LLM text: ``bot-llm-text`` joined at ``bot-llm-stopped``
``tts_response``                the TTS's spoken text: one segment per ``bot-tts-text``
                                (audio modality only)
``response``                    local-STT transcription of the bot's actual audio
                                (audio modality only); ``llm_response`` in text modality
``function_call``               ``llm-function-call-in-progress``
``function_call_stopped``       ``llm-function-call-stopped``; its ``args`` carry
                                ``tool_call_id`` and ``cancelled``, so a scenario
                                can tell work that was stopped from work that
                                finished on its own
==========================      ==============================================

Matching semantics: expected events must appear in the specified order, but
unmatched events may appear between them (so a scenario doesn't have to
enumerate every event the bot emits). The ``within_ms`` budget for each
expectation is measured from the most recent ``send-text`` / ``raw-audio`` / ``dtmf`` send
(default 60s when omitted).

A turn with a failed assertion ends the scenario, since the conversation is in
an unknown state from there on. A scenario that scores each turn independently
sets ``stop_on_failure: false`` to have every turn driven and reported.

An ``llm_response`` with a content check (``text_contains`` / ``eval:``)
aggregates: the harness accumulates the text of successive response segments
within the turn and re-checks on each one, so an interim filler ("Let me check
on that.") or the on-connect greeting is rolled past rather than mistaken for
the turn's answer. Responses that began before the turn's input are skipped, so
an interrupted prior turn doesn't bleed in. The judge returns yes / no /
continue; ``text_contains`` treats a missing substring as continue. The
``within_ms`` budget bounds the wait. A ``user_transcription`` with
``text_contains`` aggregates the same way: an STT may finalize one utterance in
several pieces, and the check runs on the pieces accumulated so far. Substring
checks ignore differences in whitespace, so pieces that carry their own spacing
still match a phrase.

Example::

    scenario = EvalScenario.load("scenarios/greeting.yaml")
    result = await EvalSession.from_scenario(scenario, "ws://localhost:7860").run()
    if result.passed:
        print("PASS")
    else:
        for f in result.failures:
            print(f"  {f}")

    # Per-turn outcomes, for a scenario scored a turn at a time.
    scored = [t for t in result.turns if t.status != "not_run"]
    print(f"{sum(1 for t in scored if t.status == 'passed')}/{len(scored)} turns")
"""

import asyncio
import base64
import mimetypes
import time
import traceback
import warnings
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from loguru import logger

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.evals.audio import load_user_audio
from pipecat.evals.client_transport import EvalHarnessTransport, HarnessRecorder
from pipecat.evals.events import EvalEventStream
from pipecat.evals.judge import EvalJudge
from pipecat.evals.results import (
    EvalAssertionFailure,
    EvalResult,
    EvalTrace,
    EvalTurnProgress,
    EvalTurnResult,
)
from pipecat.evals.scenario import (
    FUNCTION_CALL_EVENTS,
    EvalExpectation,
    EvalScenario,
    EvalSendAfter,
    EvalTurn,
    describe_config,
)
from pipecat.evals.serializer import (
    EVAL_CANCEL_MESSAGE_TYPE,
    EVAL_CONFIGURE_MESSAGE_TYPE,
    EVAL_CONTEXT_MESSAGE_TYPE,
    EVAL_IMAGE_MESSAGE_TYPE,
    HARNESS_STT_SAMPLE_RATE,
    RTVIHarnessSerializer,
)
from pipecat.evals.services import stt_service_from_config, tts_service_from_config
from pipecat.evals.tts import CachingTTSService, tts_sample_rate
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterruptionFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.stt_service import STTService
from pipecat.transports.websocket.client import WebsocketClientParams
from pipecat.utils.base_object import BaseObject
from pipecat.workers.runner import WorkerRunner

# Generous default so an expectation without an explicit ``within_ms`` waits
# long enough for slow LLM/TTS responses (and function-call round-trips) rather
# than failing on latency. Set ``within_ms`` explicitly to assert on timing.
DEFAULT_EVENT_TIMEOUT_MS = 60000
SEND_AFTER_MAX_WAIT_S = 30.0
SEND_AFTER_POLL_S = 0.01
BOT_READY_TIMEOUT_S = 10.0


class _BotFrameSink(FrameProcessor):
    """Pipeline tap that turns the bot's incoming frames into matcher events.

    Sits between the bot-audio side and the user-audio side of the eval pipeline;
    for every frame it calls the stream's
    :meth:`~pipecat.evals.events.EvalEventStream.frames_to_events` and appends
    the results with :meth:`~pipecat.evals.events.EvalEventStream.append`, then
    passes the frame on. Outgoing frames (the RTVI client messages the session
    injects) flow through untouched — they don't map to events.

    It also stops the harness's *computed* interruptions here: the user aggregator
    runs a VAD on the bot's incoming audio, so when the bot speaks it broadcasts an
    ``InterruptionFrame`` downstream. Letting that reach the user TTS / output would
    flush the user audio we're paced-sending (dropping a barge-in turn, and the
    user's side of the recording), so the sink swallows it.
    """

    def __init__(self, stream: EvalEventStream):
        super().__init__()
        self._stream = stream

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        for event in self._stream.frames_to_events(frame):
            await self._stream.append(event)
        # The bot-audio VAD's interruption must not propagate into the user-audio
        # path (user TTS + output); see the class docstring.
        if isinstance(frame, InterruptionFrame) and direction == FrameDirection.DOWNSTREAM:
            return
        await self.push_frame(frame, direction)


class EvalSession(BaseObject):
    """Runs one :class:`EvalScenario` against a bot over a single WebSocket session.

    Connects as an RTVI client, drives each turn (sending ``send-text``,
    ``raw-audio``, or ``dtmf``), collects the RTVI events the bot emits, and
    asserts on them. Build one with :meth:`from_scenario` (which constructs the
    judge, user TTS, and STT the scenario needs), then await :meth:`run`.

    Event handlers available:

    - on_progress: Called with an :class:`EvalTurnProgress` as each turn and each
      expectation resolves. Records are emitted in order, and :meth:`run` waits for
      every handler before it returns.

    Example::

        @session.event_handler("on_progress")
        async def on_progress(session, progress):
            print(progress.event_name, progress.status)
    """

    def __init__(
        self,
        scenario: EvalScenario,
        bot_url: str,
        *,
        connect_timeout_s: float = 5.0,
        default_timeout_ms: int = DEFAULT_EVENT_TIMEOUT_MS,
        on_progress: Callable[[EvalTurnProgress], None] | None = None,
        record_path: str | None = None,
        stop_bot: bool = False,
        trigger_disconnect: bool = False,
        judge: EvalJudge | None = None,
        user_tts: CachingTTSService | None = None,
        bot_stt: STTService | None = None,
    ):
        """Initialize the eval session.

        The ``judge``, ``user_tts``, and ``bot_stt`` are injected pre-built:
        :meth:`from_scenario` constructs the defaults from the scenario's config
        and passes them in. Construct and pass your own to override them (e.g. a
        custom judge LLM, TTS, or STT service).

        Args:
            scenario: The parsed scenario to run.
            bot_url: WebSocket URL of the bot's eval transport.
            connect_timeout_s: How long to wait for the bot to accept the WS
                connection before giving up.
            default_timeout_ms: Per-expectation latency budget for expectations
                without their own ``within_ms`` (the turn's expectations share one
                deadline anchored at the send). Defaults to 60s.
            on_progress: Optional callback invoked with a :class:`EvalTurnProgress`
                as each turn and expectation resolves (used for verbose output).

                .. deprecated:: 1.9.0
                    Use the ``on_progress`` event handler instead.
                    Will be removed in 2.0.0.

            record_path: When set (and the scenario is audio mode), asks the eval
                transport to record the conversation audio to this path (bot-side).
            stop_bot: When True, ask the bot to cancel its pipeline (and exit) on
                teardown via ``eval-cancel``. The suite enables it to clean up
                each spawned bot.
            trigger_disconnect: When True (or when the scenario sets
                ``trigger_disconnect``), ask the eval transport to fire the bot's
                ``on_client_disconnected`` handler when this connection ends.
                Bots often cancel their pipeline there, so it is off by default
                to avoid that between scenarios.
            judge: The :class:`~pipecat.evals.judge.EvalJudge` for ``eval:``
                assertions, or ``None`` if the scenario has none.
            user_tts: The :class:`~pipecat.evals.tts.CachingTTSService` that
                synthesizes user audio (added to the eval pipeline in audio mode),
                or ``None`` for text-mode scenarios.
            bot_stt: The ``STTService`` that transcribes the bot's audio into the
                ``response`` event (added to the eval pipeline in audio mode), or
                ``None`` when unused.
        """
        super().__init__()

        self._scenario = scenario
        self._bot_url = bot_url
        self._connect_timeout_s = connect_timeout_s
        self._default_timeout_ms = default_timeout_ms
        self._record_path = record_path
        self._stop_bot = stop_bot
        # Either the run-wide CLI flag or the scenario's own field opts in.
        self._trigger_disconnect = trigger_disconnect or scenario.trigger_disconnect

        # The eval pipeline's worker that talks to the bot (built in run()).
        self._worker: PipelineWorker | None = None
        # Records the conversation audio (bot + user) when record_path is set and
        # the scenario is audio mode; fed raw audio by the transport, written in run().
        self._recorder: HarnessRecorder | None = None
        # Set by the transport's on_bot_ready handler once the bot completes the
        # RTVI handshake; _handshake() waits on it.
        self._bot_ready_event = asyncio.Event()
        # function_call events popped while matching another expectation, held so
        # the turn's calls can be matched by name in any order (reset per turn).
        self._pending_function_calls: list[dict] = []
        # Timestamped trace of the harness's own decisions, for diagnosing flakes.
        self._trace = EvalTrace()
        self._next_id = 0
        self._judge: EvalJudge | None = judge

        # One persistent TTS pipeline reused across the scenario's audio turns,
        # started in run(); None for text-mode scenarios.
        self._user_tts: CachingTTSService | None = user_tts

        # The bot's output as events: fed by the pipeline's sink, read by the matcher.
        self._stream = EvalEventStream(bot_audio=scenario.bot_audio, trace=self._trace)

        # Text content of the most recently matched event (the bot's response, or
        # a user transcript), surfaced to verbose progress. Empty for events with
        # no text (llm_started, function_call, speaking events).
        self._last_match_text: str = ""

        # response (audio modality): an STT in the eval pipeline transcribes the
        # bot's actual audio. Only built when a scenario asserts `response`.
        self._wants_response: bool = any(
            exp.event == "response" for turn in scenario.turns for exp in turn.expect
        )
        self._bot_stt: STTService | None = bot_stt

        self._register_event_handler("on_progress")
        if on_progress is not None:
            self._add_legacy_progress_callback(on_progress)

    @classmethod
    def from_scenario(
        cls,
        scenario: EvalScenario,
        bot_url: str,
        *,
        connect_timeout_s: float = 5.0,
        default_timeout_ms: int = DEFAULT_EVENT_TIMEOUT_MS,
        on_progress: Callable[[EvalTurnProgress], None] | None = None,
        record_path: str | None = None,
        cache_dir: str | None = None,
        use_cache: bool = True,
        stop_bot: bool = False,
        trigger_disconnect: bool = False,
        judge: EvalJudge | None = None,
        user_tts: CachingTTSService | None = None,
        bot_stt: STTService | None = None,
    ) -> "EvalSession":
        """Build a ready-to-run session from a scenario, constructing what it needs.

        Builds the judge, user TTS, and STT the scenario calls for and injects them
        into a new session. Pass ``judge`` /
        ``user_tts`` / ``bot_stt`` to override any of them with your own pre-built
        instance. Then await :meth:`run`::

            session = EvalSession.from_scenario(scenario, "ws://localhost:7860")
            result = await session.run()

        Args:
            scenario: The parsed scenario to run.
            bot_url: WebSocket URL of the bot's eval transport.
            connect_timeout_s: How long to wait for the bot to accept the WS
                connection before giving up.
            default_timeout_ms: Per-expectation latency budget for expectations
                without their own ``within_ms``. Defaults to 60s.
            on_progress: Optional per-turn/expectation progress callback (verbose).

                .. deprecated:: 1.9.0
                    Use the ``on_progress`` event handler instead.
                    Will be removed in 2.0.0.

            record_path: Optional path to record the conversation audio (audio mode).
            cache_dir: Optional directory for cached synthesized user audio
                (default ``<user-cache-dir>/pipecat/tts``).
            use_cache: When False, ignore cached user audio and force fresh synthesis
                (no cache reads or writes). Defaults to True.
            stop_bot: When True, ask the bot to cancel its pipeline (and exit) on
                teardown. Leave False to keep it running for more scenarios.
            trigger_disconnect: When True, fire the bot's ``on_client_disconnected``
                handler when the connection ends (the scenario's own
                ``trigger_disconnect`` field also opts in). Off by default.
            judge: Override the judge (default: built from ``scenario.judge`` when the
                scenario has ``eval:`` assertions).
            user_tts: Override the user-audio TTS (default: built from
                ``scenario.user_speech`` in audio mode).
            bot_stt: Override the bot-audio STT (default: built from
                ``scenario.transcriber`` when the scenario asserts ``response``).

        Returns:
            A configured session, ready for :meth:`run`.
        """
        turns = scenario.turns
        if judge is None and any(exp.eval is not None for turn in turns for exp in turn.expect):
            with logger.contextualize(eval_pipeline="judge"):
                judge = EvalJudge.from_config(scenario.judge)

        if user_tts is None and scenario.user_speech is not None:
            with logger.contextualize(eval_pipeline="speech"):
                user_tts = tts_service_from_config(
                    scenario.user_speech, cache_dir=cache_dir, use_cache=use_cache
                )

        wants_response = any(exp.event == "response" for turn in turns for exp in turn.expect)
        if bot_stt is None and wants_response and scenario.bot_audio:
            with logger.contextualize(eval_pipeline="transcription"):
                bot_stt = stt_service_from_config(scenario.transcriber)

        session = cls(
            scenario,
            bot_url,
            connect_timeout_s=connect_timeout_s,
            default_timeout_ms=default_timeout_ms,
            record_path=record_path,
            stop_bot=stop_bot,
            trigger_disconnect=trigger_disconnect,
            judge=judge,
            user_tts=user_tts,
            bot_stt=bot_stt,
        )
        if on_progress is not None:
            session._add_legacy_progress_callback(on_progress)
        return session

    async def run(self) -> EvalResult:
        """Connect, drive the scenario, and return the result."""
        started = time.monotonic()
        self._trace.start()
        self._trace.log(f"run: scenario {self._scenario.name!r} -> {self._bot_url}")
        # Record which speech / transcription / judge services and models were used,
        # so a saved eval.log is self-describing (no need to cross-reference config).
        for line in describe_config(self._scenario).splitlines():
            self._trace.log(line)

        # One record per scenario turn, filled in as the turns are driven. They
        # start as not_run and stay that way on every path that ends the run
        # early, so the result always says which turns were actually scored.
        turns = [EvalTurnResult(turn_index=i) for i in range(len(self._scenario.turns))]

        # The `response` transcription needs the bot's actual audio; without audio
        # mode there's nothing to transcribe, so skip rather than fail. (Normally
        # unreachable: EvalScenario.load resolves `response` to llm_response in text
        # modality; this guards Scenarios built directly.)
        if self._wants_response and not self._scenario.bot_audio:
            reason = "asserts 'response' transcription but judge modality is text (no audio)"
            logger.warning(f"Eval '{self._scenario.name}': {reason}; skipping")
            return EvalResult(
                scenario_name=self._scenario.name,
                passed=False,
                turns=turns,
                skipped=reason,
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        # Readiness wait: the suite spawns a bot and immediately drives it, so retry a
        # lightweight TCP connect until the bot's server is accepting (this is the only
        # readiness wait — see EvalSuite). It deliberately does *not* complete the
        # WebSocket/RTVI handshake: a full connect would fire the bot's
        # on_client_connected (kicking off a greeting and mutating its context) and then
        # throw it away, leaving the real session below with a duplicated opening. The
        # transport owns the one real connection. A bot that never accepts is a clean
        # <connect> failure.
        u = urlsplit(self._bot_url)
        host, port = u.hostname or "localhost", u.port or (443 if u.scheme == "wss" else 80)
        deadline = time.monotonic() + self._connect_timeout_s
        connect_error: Exception | None = None
        ready = False
        while not ready and time.monotonic() < deadline:
            try:
                _reader, writer = await asyncio.open_connection(host, port)
                writer.close()
                try:
                    await writer.wait_closed()
                except OSError:
                    pass
                ready = True
            except OSError as e:  # not accepting connections yet
                connect_error = e
                await asyncio.sleep(0.25)
        if not ready:
            e = connect_error or TimeoutError("timed out")
            return EvalResult(
                scenario_name=self._scenario.name,
                passed=False,
                failures=[
                    EvalAssertionFailure(
                        turn_index=-1,
                        expectation_index=-1,
                        event_name="<connect>",
                        reason=f"failed to connect to {self._bot_url}: {e.__class__.__name__}",
                        kind="connect_failed",
                    )
                ],
                turns=turns,
                duration_ms=int((time.monotonic() - started) * 1000),
            )

        # Build one eval pipeline for both modes:
        #
        #   input -> [STT -> user aggregator] -> sink -> [user TTS] -> output
        #
        # The STT + aggregator (with the aggregator's own VAD) transcribe the bot's
        # audio into the `response`; the user TTS turns TTSSpeakFrames into the audio
        # sent to the bot. Each bracketed stage is present only when its service was
        # built (audio scenarios); in text mode they're simply absent and no audio
        # flows. The sink turns the bot's frames into matcher events either way.
        user_audio_rate = (
            tts_sample_rate(self._scenario.user_speech) if self._scenario.user_speech else 0
        )
        params = WebsocketClientParams(
            audio_in_enabled=self._scenario.bot_audio,
            audio_out_enabled=self._sends_user_audio,
            audio_in_sample_rate=HARNESS_STT_SAMPLE_RATE if self._scenario.bot_audio else 0,
            audio_out_sample_rate=user_audio_rate,
            serializer=RTVIHarnessSerializer(),
        )
        # Record the conversation from the *raw* audio the transport sees on each
        # edge (the user TTS as produced, the bot's chunks as received), not the
        # paced/filled pipeline frames: Python can't hold the 40ms pacing tick
        # precisely, and recording the paced streams stutters. The recorder
        # reconstructs gapless turns and pads only the real between-turn pauses.
        if self._record_path and self._scenario.bot_audio:
            self._recorder = HarnessRecorder(user_audio_rate or HARNESS_STT_SAMPLE_RATE)
        # EvalHarnessTransport reshapes both audio edges into the continuous
        # real-time stream VAD/STT expect: its output paces the user TTS to the bot
        # and its input fills gaps in the bot's audio (both audio-mode only). When a
        # recorder is set, both edges also feed it the raw audio for the recording.
        transport = EvalHarnessTransport(self._connect_url(), params, recorder=self._recorder)

        @transport.event_handler("on_bot_ready")
        async def _on_bot_ready(_transport):
            self._bot_ready_event.set()

        sink = _BotFrameSink(self._stream)
        processors: list = [transport.input()]
        if self._bot_stt is not None:
            user_aggregator = LLMContextAggregatorPair(
                LLMContext(),
                user_params=LLMUserAggregatorParams(
                    vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.4))
                ),
            ).user()

            # The aggregator consumes the STT's TranscriptionFrames to build the
            # bot's turn, so the response comes from the aggregated turn text here
            # (not from a frame at the sink). This is also the judge hook for sims.
            # Skip while awaiting an LLM restart: an interrupted turn finalizes
            # *after* the interruption, and that straggler must not be matched
            # against the next turn (mirrors the llm_response suppression).
            @user_aggregator.event_handler("on_user_turn_stopped")
            async def _on_user_turn_stopped(_aggregator, _strategy, message):
                if message.content and not self._stream.awaiting_llm_restart:
                    await self._stream.append({"type": "response", "text": message.content})

            processors += [self._bot_stt, user_aggregator]
        processors.append(sink)
        if self._user_tts is not None:
            # The user TTS speaks only the harness's TTSSpeakFrames; the bot's text
            # flowing past it is passed through unspoken (see CachingTTSService), so
            # no echo and no gate needed.
            processors.append(self._user_tts)
        processors.append(transport.output())
        pipeline = Pipeline(processors)
        # The StartFrame's rates drive the in-pipeline services: the bot-audio STT
        # reads `audio_in_sample_rate`, the user TTS produces `audio_out_sample_rate`.
        # Set them from the scenario so the user TTS synthesizes at the configured
        # user_audio rate rather than the PipelineParams `audio_out` default (a
        # mismatch that would mislabel the cached audio). Text-mode scenarios have no
        # audio in/out, so the STT rate is a harmless placeholder for `audio_out`.
        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=HARNESS_STT_SAMPLE_RATE,
                audio_out_sample_rate=user_audio_rate or HARNESS_STT_SAMPLE_RATE,
            ),
            enable_rtvi=False,
            cancel_on_idle_timeout=False,
        )
        self._worker = worker
        runner = WorkerRunner()
        await runner.add_workers(worker)
        run_task = asyncio.create_task(runner.run())

        failures: list[EvalAssertionFailure] = []
        try:
            # The STT and user TTS run inside the pipeline (started by the worker
            # above); the judge runs out-of-band during matching. Everything below
            # is under this `try` so a service that fails to start (e.g. a local
            # model under load) surfaces as a failure rather than propagating raw.
            self._trace.log("connected")
            try:
                await self._handshake()
                self._trace.log("handshake: ok (bot-ready)")
            except TimeoutError:
                self._trace.log("handshake: failed (bot-ready not received)")
                failures.append(
                    EvalAssertionFailure(
                        turn_index=-1,
                        expectation_index=-1,
                        event_name="<bot-ready>",
                        reason=f"bot-ready not received within {int(BOT_READY_TIMEOUT_S * 1000)}ms",
                        kind="handshake_timeout",
                    )
                )
            else:
                for turn_idx, turn in enumerate(self._scenario.turns):
                    self._trace.turn = turn_idx
                    self._trace.log(f"--- turn {turn_idx}: {turn.user!r}")
                    turn_started = time.monotonic()
                    turn_failures = await self._run_turn(turn, turn_idx)
                    record = turns[turn_idx]
                    record.status = "failed" if turn_failures else "passed"
                    record.failures = turn_failures
                    record.duration_ms = int((time.monotonic() - turn_started) * 1000)
                    failures.extend(turn_failures)
                    if turn_failures:
                        # By default a failed turn ends the scenario: it leaves the
                        # conversation in an unknown state, so running the rest just
                        # burns another timeout per turn (e.g. a broken greeting turn
                        # shouldn't cost the full budget here and again on the
                        # question). A scenario whose turns are scored independently
                        # sets stop_on_failure: false and drives all of them.
                        if self._scenario.stop_on_failure:
                            self._trace.log(
                                f"turn {turn_idx} failed; stopping scenario (stop_on_failure)"
                            )
                            break
                        self._trace.log(
                            f"turn {turn_idx} failed; continuing (stop_on_failure: false)"
                        )
        except Exception as e:
            # An unexpected harness-side error (a sub-pipeline failing to start
            # under load, a judge/transcriber raising mid-turn, ...) would
            # otherwise propagate up to the suite and be swallowed as a bare
            # "error: <str>" with no eval.log. Capture it as a failure so the
            # reason and full traceback land in the result's debug trace (saved
            # to <bot>.eval.log) and the run still reports a structured outcome.
            self._trace.log(f"error: {type(e).__name__}: {e}")
            for line in traceback.format_exc().rstrip().splitlines():
                self._trace.log(line)
            failure = EvalAssertionFailure(
                turn_index=self._trace.turn,
                expectation_index=-1,
                event_name="<error>",
                reason=f"{type(e).__name__}: {e}",
                kind="harness_error",
            )
            failures.append(failure)
            # The raise happened either inside a turn — which is that turn's
            # failure — or before any of them started (a sub-pipeline that never
            # came up), where the trace's turn is still -1 and every turn is not_run.
            if 0 <= self._trace.turn < len(turns):
                record = turns[self._trace.turn]
                record.status = "failed"
                record.failures.append(failure)
        finally:
            # Write the recording now that the conversation is done (the recorder is
            # harness-owned, fed raw audio by the transport, so nothing in teardown
            # clears it -- but write here so it lands even if teardown below raises).
            await self._write_recording()
            # Optionally ask the bot to tear its pipeline down gracefully so it exits
            # on its own (best-effort; skipped by default so it stays up for more
            # scenarios).
            if self._stop_bot:
                await self._send_cancel()
            # Stop the eval pipeline: end the worker (which disconnects the
            # transport), falling back to cancel if it doesn't wind down cleanly.
            try:
                await self._worker.queue_frame(EndFrame())
                await asyncio.wait_for(run_task, timeout=5.0)
            except (TimeoutError, asyncio.CancelledError, Exception):
                run_task.cancel()
                try:
                    await run_task
                except (asyncio.CancelledError, Exception):
                    pass
            # Progress handlers run as tasks; wait them out so every record is
            # delivered before the caller has the result in hand.
            await self.cleanup()

        self._trace.log(f"done: {'PASS' if not failures else 'FAIL'} ({len(failures)} failure(s))")
        return EvalResult(
            scenario_name=self._scenario.name,
            passed=not failures,
            failures=failures,
            turns=turns,
            duration_ms=int((time.monotonic() - started) * 1000),
            events_seen=self._stream.events_seen,
            debug_log=self._trace.lines,
        )

    @property
    def _sends_user_audio(self) -> bool:
        """Whether any user turn reaches the bot as audio, synthesized or from a file."""
        return self._user_tts is not None or any(t.audio for t in self._scenario.turns)

    async def _write_recording(self) -> None:
        """Write the recorded conversation audio (bot + user) to ``record_path``."""
        if self._recorder is None or not self._record_path or not self._recorder.has_audio():
            return
        if await self._recorder.write(self._record_path):
            self._trace.log(f"recording saved: {self._record_path}")

    def _connect_url(self) -> str:
        """Bot URL with the per-connection eval query flags.

        ``skip_tts`` (text mode) silences the bot before any LLM runs; the eval
        transport must read it at connect time because frames are ordered and a
        later message can't precede an on-connect greeting (see
        :mod:`pipecat.evals.transport`). ``capture_bot_audio`` makes the bot forward
        its synthesized audio to the harness, both for ``response``/``tts_response``
        transcription and so the harness can record the bot's side (recording itself
        is harness-side now). ``trigger_disconnect`` asks the transport to fire the
        bot's ``on_client_disconnected`` handler when the connection ends (off by
        default, since bots often cancel there).
        """
        flags = []
        if not self._scenario.bot_audio:
            flags.append("skip_tts=true")
        # Forward the bot's audio when a scenario asserts on it (response) or when
        # recording an audio scenario (the harness records the bot's side from it).
        if self._wants_response or (self._record_path and self._scenario.bot_audio):
            flags.append("capture_bot_audio=true")
        if self._trigger_disconnect:
            flags.append("trigger_disconnect=true")
        if not flags:
            return self._bot_url
        sep = "&" if "?" in self._bot_url else "?"
        return f"{self._bot_url}{sep}{'&'.join(flags)}"

    def _message_id(self) -> str:
        self._next_id += 1
        return str(self._next_id)

    def _required_report_level(self) -> str | None:
        """Minimal function-call report level the scenario's assertions need.

        Returns ``"full"`` if any ``function_call`` expectation checks ``args``,
        ``"name"`` if one checks ``name`` only, else ``None`` (no elevation —
        the bot's default applies and a function_call event still arrives).
        """
        needs_name = False
        for turn in self._scenario.turns:
            for exp in turn.expect:
                if exp.event not in FUNCTION_CALL_EVENTS:
                    continue
                # name/args live in exp.calls (the parser normalizes the single
                # name:/args: shorthand into it too).
                for call in exp.calls or []:
                    if call.args is not None:
                        return "full"
                    if call.name is not None:
                        needs_name = True
        return "name" if needs_name else None

    def _needs_vad_events(self) -> bool:
        """Whether the scenario references raw VAD speaking events.

        These (``vad_user_started_speaking`` / ``vad_user_stopped_speaking``) are
        off by default; the harness asks the bot's RTVIObserver to emit them only
        when a scenario asserts on or schedules from them.
        """
        vad_events = {"vad_user_started_speaking", "vad_user_stopped_speaking"}
        for turn in self._scenario.turns:
            if turn.send_after is not None and turn.send_after.event in vad_events:
                return True
            if any(exp.event in vad_events for exp in turn.expect):
                return True
        return False

    async def _handshake(self) -> None:
        """Send client-ready, wait for bot-ready, then optionally seed context.

        ``bot-ready`` is a hard gate: the eval framework requires an RTVI bot, so a
        bot that never announces readiness either isn't a valid eval target or
        hasn't finished starting (services still connecting). Rather than fire
        turns at a half-started bot — which produces flaky, hard-to-read failures —
        we raise :class:`TimeoutError` so the caller reports a clean connect-level
        failure.
        """
        # The transport sends client-ready on connect and fires on_bot_ready when
        # the bot answers; our handler sets _bot_ready_event. Hard gate — raises
        # TimeoutError if the bot never announces readiness.
        await asyncio.wait_for(self._bot_ready_event.wait(), timeout=BOT_READY_TIMEOUT_S)

        # Ask the bot's RTVIObserver to expose what this scenario needs, for the
        # duration of this eval only (bots keep their defaults; only the eval
        # transport understands this): raise the function-call report level if it
        # asserts on call name/args, and enable raw VAD speaking events if it uses
        # them.
        level = self._required_report_level()
        vad = self._needs_vad_events()
        if level is not None or vad:
            config: dict = {}
            if level is not None:
                config["function_call_report_level"] = {"*": level}
            if vad:
                config["vad_user_speaking"] = True
            configure = RTVI.Message(
                type="client-message",
                id=self._message_id(),
                data={"t": EVAL_CONFIGURE_MESSAGE_TYPE, "d": config},
            )
            await self._send(configure)

        # Only send the eval-context when the scenario provides starting context.
        # An implicit empty one would race with bot startup flows (e.g. a greeting
        # added in on_client_connected), wiping the bot's context right after it
        # set it up.
        if self._scenario.context:
            context_message = RTVI.Message(
                type="client-message",
                id=self._message_id(),
                data={"t": EVAL_CONTEXT_MESSAGE_TYPE, "d": {"messages": self._scenario.context}},
            )
            await self._send(context_message)

    async def _send(self, message: RTVI.Message) -> None:
        """Send an RTVI client message to the bot through the transport pipeline."""
        assert self._worker is not None  # pipeline built before any send
        await self._worker.queue_frame(
            OutputTransportMessageUrgentFrame(message=message.model_dump())
        )

    async def _send_cancel(self) -> None:
        """Ask the bot to cancel its pipeline so it shuts down gracefully.

        Best-effort: the connection may already be gone, in which case the
        orchestrator's kill fallback handles teardown.
        """
        try:
            message = RTVI.Message(
                type="client-message",
                id=self._message_id(),
                data={"t": EVAL_CANCEL_MESSAGE_TYPE, "d": {}},
            )
            await self._send(message)
        except Exception:
            pass

    @staticmethod
    def _match_summary(event: dict) -> str:
        """A short human label for a matched event, for verbose progress.

        For ``function_call`` it's the call signature (``name(arg=value, ...)``);
        for everything else it's the event's text content (or empty).
        """
        if event.get("type") == "function_call":
            args = event.get("args") or {}
            sig = ", ".join(f"{k}={v}" for k, v in args.items())
            return f"{event.get('name') or '?'}({sig})"
        return _event_text(event)

    def _add_legacy_progress_callback(
        self, on_progress: Callable[[EvalTurnProgress], None]
    ) -> None:
        """Register a bare ``on_progress`` callback as an ``on_progress`` handler.

        The callback takes only the record, so it is wrapped to drop the session
        that event handlers receive as their first argument.
        """
        warnings.warn(
            "`on_progress` is deprecated since 1.9.0 and will be removed in 2.0.0. "
            "Use the `on_progress` event handler instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        self.add_event_handler("on_progress", lambda _session, record: on_progress(record))

    async def _progress(self, record: EvalTurnProgress) -> None:
        """Emit a progress record to the ``on_progress`` handlers."""
        await self._call_event_handler("on_progress", record)

    async def _run_turn(self, turn: EvalTurn, turn_idx: int) -> list[EvalAssertionFailure]:
        """Drive one turn: optionally honor send_after, send user input, match expectations.

        The user turn is sent as ``send-text`` (text mode) or, in audio mode, as
        chunked ``raw-audio`` messages that the bot's STT transcribes for real --
        the turn's ``audio:`` recording when it names one, otherwise its text
        synthesized by the user TTS.
        """
        failures: list[EvalAssertionFailure] = []
        # The turn's function calls match by name in any order; start each turn
        # with an empty buffer so a prior turn's calls can't carry over.
        self._pending_function_calls = []

        if turn.send_after is not None:
            try:
                await self._wait_send_after(turn.send_after)
            except TimeoutError as e:
                # Only the event-anchored wait can time out; the pure-delay form
                # just sleeps. So event is never None here, but fall back for typing.
                event_name = turn.send_after.event or "send_after"
                failures.append(
                    EvalAssertionFailure(
                        turn_index=turn_idx,
                        expectation_index=-1,
                        event_name=event_name,
                        reason=f"send_after never fired: {e}",
                        kind="send_after_timeout",
                    )
                )
                self._trace.log(f"FAIL: {event_name}: {failures[-1].reason}")
                await self._progress(
                    EvalTurnProgress(turn_idx, -1, event_name, "timeout", failures[-1].reason)
                )
                return failures

        # Register the turn's image (if any) before the user input, so the bot can
        # serve it when it requests a user image during the turn.
        if turn.image is not None:
            await self._send_image(turn.image)

        # Anything still queued belongs to an earlier turn: this turn's input hasn't
        # been sent, so the bot cannot have responded to it yet. Drop it, or an
        # expectation here can match — and a judge can rule on — output the bot
        # produced for a previous turn. The bot's own interruption events close this
        # window too, but only once the input reaches it, which is far too late when
        # `send_after` holds the send back for seconds.
        #
        # Before the send, not after: by the time the input has streamed, the bot has
        # begun reacting to it, and this turn's own `user_started_speaking` /
        # `bot_interrupted` would be dropped along with the stale output. Turns that
        # send nothing are observation-only and exist to match exactly this pending
        # output (a bot-first greeting), so they keep it.
        if turn.user is not None or turn.dtmf is not None:
            self._stream.drop_pending_bot_output("before send")

        if turn.user is not None:
            how = turn.audio or ("audio" if self._user_tts is not None else "text")
            self._trace.log(f"send: {turn.user!r} ({how})")
            if turn.audio is not None:
                await self._send_audio_file(turn.audio)
            elif self._user_tts is not None:
                await self._send_user_audio(turn.user)
            else:
                await self._send_user_text(turn.user, self._scenario.bot_audio)
            # Record the user turn in the judge's conversation, so a later reply is
            # judged in context (e.g. a terse "That's four" answering this question).
            if self._judge is not None:
                self._judge.add_user_message(turn.user)
        elif turn.dtmf is not None:
            self._trace.log(f"send: dtmf {turn.dtmf!r}")
            await self._send_user_dtmf(turn.dtmf)
            # Record the keypresses for judge context, so the bot's reply is judged
            # knowing what was pressed.
            if self._judge is not None:
                self._judge.add_user_message(f"(DTMF keypad input: {turn.dtmf})")

        if turn.user is not None or turn.dtmf is not None:
            # Suppress in-flight stragglers until the bot's fresh response begins
            # (bot-llm-started clears the flag), so this turn matches only what the
            # bot says in reply to this input.
            self._stream.awaiting_llm_restart = True

        await self._progress(EvalTurnProgress(turn_idx, -1, turn.user or turn.dtmf or "", "turn"))

        # All of a turn's expectations share one deadline anchored at the send, so a
        # stalled turn fails within a single ``within_ms`` budget instead of spending
        # a fresh budget per expectation — e.g. a missing function call followed by a
        # missing response fails in 60s total, not 120s.
        anchor = time.monotonic()
        for exp_idx, expectation in enumerate(turn.expect):
            budget_ms = expectation.within_ms or self._default_timeout_ms

            try:
                failure = await self._match_and_verify(
                    expectation, anchor, budget_ms, turn_idx, exp_idx
                )
            except TimeoutError:
                reason = f"no matching {expectation.event!r} event arrived within {budget_ms}ms"
                failures.append(
                    EvalAssertionFailure(
                        turn_index=turn_idx,
                        expectation_index=exp_idx,
                        event_name=expectation.event,
                        reason=reason,
                        kind="timeout",
                    )
                )
                self._trace.log(f"FAIL: {expectation.event}: {reason}")
                await self._progress(
                    EvalTurnProgress(turn_idx, exp_idx, expectation.event, "timeout", reason)
                )
                break

            if failure:
                failures.append(failure)
                self._trace.log(f"FAIL: {expectation.event}: {failure.reason}")
                await self._progress(
                    EvalTurnProgress(turn_idx, exp_idx, expectation.event, "failed", failure.reason)
                )
            else:
                await self._progress(
                    EvalTurnProgress(
                        turn_idx, exp_idx, expectation.event, "matched", self._last_match_text
                    )
                )

        return failures

    async def _send_user_text(self, text: str, bot_audio: bool) -> None:
        """Send a text user turn via the RTVI ``send-text`` message.

        ``audio_response`` mirrors the scenario's ``bot_audio``: when False the
        LLM bypasses TTS for this turn (content-only evals).
        """
        message = RTVI.Message(
            type="send-text",
            id=self._message_id(),
            data=RTVI.SendTextData(
                content=text,
                options=RTVI.SendTextOptions(run_immediately=True, audio_response=bot_audio),
            ).model_dump(),
        )
        await self._send(message)

    async def _send_user_dtmf(self, keys: str) -> None:
        """Send a DTMF keypress turn as one RTVI ``dtmf`` message.

        The bot's ``RTVIProcessor`` turns each key into an ``InputDTMFFrame``
        pushed downstream, the same path a telephony transport's keypress takes.
        The bot's ``DTMFAggregator`` (if any) accumulates them and flushes — on
        the ``#`` terminator or its idle timeout — into a transcription the bot
        reacts to. Use ``send_after`` across turns to pace key sequences.
        """
        message = RTVI.Message(
            type="dtmf",
            id=self._message_id(),
            data={"buttons": list(keys)},
        )
        await self._send(message)

    async def _send_image(self, image_path: str) -> None:
        """Register an image (base64, with its MIME type) for the current turn.

        The eval transport serves it back as a ``UserImageRawFrame`` when the bot
        requests a user image. The file is sent as-is (already PNG/JPEG/...), so
        nothing is decoded or re-encoded.
        """
        path = Path(image_path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        self._trace.log(f"send: image {path.name} ({mime})")
        message = RTVI.Message(
            type="client-message",
            id=self._message_id(),
            data={"t": EVAL_IMAGE_MESSAGE_TYPE, "d": {"image": encoded, "format": mime}},
        )
        await self._send(message)

    async def _send_user_audio(self, text: str) -> None:
        """Speak ``text`` as the user by pushing a ``TTSSpeakFrame`` into the pipeline.

        The user TTS (:class:`~pipecat.evals.tts.CachingTTSService`) renders it to
        audio (cached), which the output transport
        (:class:`~pipecat.evals.client_transport.EvalHarnessOutputTransport`) paces
        to the bot as a continuous real-time stream.
        """
        assert self._worker is not None  # pipeline built before any send
        await self._worker.queue_frame(TTSSpeakFrame(text))

    async def _send_audio_file(self, path: str) -> None:
        """Play a turn's ``audio:`` recording to the bot in place of synthesizing it.

        The recording is spoken exactly like a user TTS utterance: one audio frame
        bracketed by ``TTSStartedFrame`` / ``TTSStoppedFrame``, pushed into the
        pipeline. The output transport resamples it to the user-audio rate, paces
        it to the bot, flushes its final partial chunk on the stop frame, and
        records it.
        """
        assert self._worker is not None  # pipeline built before any send
        pcm, sample_rate = await load_user_audio(path)
        for frame in (
            TTSStartedFrame(),
            TTSAudioRawFrame(audio=pcm, sample_rate=sample_rate, num_channels=1),
            TTSStoppedFrame(),
        ):
            await self._worker.queue_frame(frame)

    async def _wait_send_after(self, send_after: EvalSendAfter) -> None:
        """Block until ``send_after.event`` has been seen + ``delay_ms`` has elapsed.

        If the event was seen earlier in the run, anchor on that time (potentially
        fire immediately). Otherwise, poll the latest_event_times map until the
        event arrives, then anchor on that.

        With no event (``send_after.event is None``), it's a pure time delay:
        sleep ``delay_ms`` from now (i.e. from the previous turn's send).
        """
        target_delay_s = send_after.delay_ms / 1000.0

        if send_after.event is None:
            self._trace.log(f"send_after: waiting {send_after.delay_ms}ms")
            await asyncio.sleep(target_delay_s)
            return

        deadline = time.monotonic() + SEND_AFTER_MAX_WAIT_S
        self._trace.log(f"send_after: waiting for {send_after.event!r} + {send_after.delay_ms}ms")

        while True:
            seen_at = self._stream.latest_event_times.get(send_after.event)
            if seen_at is not None:
                wait_s = max(0.0, (seen_at + target_delay_s) - time.monotonic())
                await asyncio.sleep(wait_s)
                return

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"event {send_after.event!r} not seen within "
                    f"{int(SEND_AFTER_MAX_WAIT_S * 1000)}ms"
                )

            await asyncio.sleep(SEND_AFTER_POLL_S)

    async def _match_and_verify(
        self,
        expectation: EvalExpectation,
        anchor: float,
        budget_ms: int,
        turn_idx: int,
        exp_idx: int,
    ) -> EvalAssertionFailure | None:
        """Wait for the expected event and verify it. Returns a failure or None.

        Most events match a single event and are checked once. An ``llm_response``
        carrying a content check (``text_contains`` / ``eval:``) instead
        *aggregates*: it accumulates the text of successive response segments and
        re-checks on each new segment until the check passes, the judge
        affirmatively rejects, or the ``within_ms`` budget expires. A
        ``user_transcription`` with ``text_contains`` aggregates the same way,
        since an STT may finalize one utterance in several pieces.

        Output that predates this turn (the greeting, or a turn that was
        interrupted) isn't specially filtered: in audio mode the reader already
        drops it from the queue when ``user-started-speaking`` fires, and anything
        that slips through (e.g. a text-mode greeting) is harmless — the judge
        returns "continue" until the turn's real answer is aggregated in.

        Raises:
            TimeoutError: when no matching event arrives at all (so the caller can
                report "no matching event arrived"). A response that arrives but
                never satisfies the content check returns a failure instead.
        """
        deadline = anchor + (budget_ms / 1000.0)
        self._last_match_text = ""

        if expectation.absent:
            return await self._match_absent(expectation, deadline, budget_ms, turn_idx, exp_idx)

        aggregates = (
            expectation.event in ("response", "llm_response", "tts_response")
            and (expectation.text_contains is not None or expectation.eval is not None)
        ) or (
            expectation.event == "user_transcription"
            and expectation.text_contains is not None
            and expectation.eval is None
        )
        if not aggregates:
            if expectation.event in FUNCTION_CALL_EVENTS:
                # A call expectation holds the set of calls the turn should make;
                # it completes only when all are found, in any order (a response
                # arriving doesn't short-circuit it).
                return await self._match_function_calls(expectation, deadline, turn_idx, exp_idx)
            self._trace.log(f"match: waiting for {expectation.event!r}")
            event = await self._stream.next_event(expectation.event, deadline)
            payload_failure = self._check_payload(event, expectation, turn_idx, exp_idx)
            if payload_failure:
                return payload_failure
            judge_failure = await self._check_judge(event, expectation, turn_idx, exp_idx)
            if judge_failure is None:
                self._last_match_text = self._match_summary(event)
            return judge_failure

        def fail(reason: str, kind: str) -> EvalAssertionFailure:
            return EvalAssertionFailure(turn_idx, exp_idx, expectation.event, reason, kind)

        if expectation.eval is not None and self._judge is None:
            return fail("scenario uses 'eval:' but no judge could be built", "no_judge")

        check = "+".join(
            name
            for name, val in (
                ("text_contains", expectation.text_contains),
                ("eval", expectation.eval),
            )
            if val is not None
        )
        self._trace.log(f"match: waiting for {expectation.event!r} ({check})")
        aggregate = ""
        last_reason = ""
        seen_any = False
        while True:
            try:
                event = await self._stream.next_event(expectation.event, deadline)
            except TimeoutError:
                if not seen_any:
                    raise  # no response at all → caller logs "no matching event arrived"
                self._trace.log(f"eval: timeout, not satisfied: {last_reason}")
                # Without `eval:` the only way to be unsatisfied is a missing
                # substring: `text_contains` is monotonic, so it holds out for more
                # text rather than failing outright.
                return fail(
                    f"not satisfied within {budget_ms}ms: {last_reason}",
                    "judge_continue" if expectation.eval is not None else "text_mismatch",
                )

            seen_any = True
            delta = _event_text(event)
            aggregate += delta
            # Feed each segment to the judge as its own assistant message, so it
            # judges the bot's reply in the conversation's context (the cumulative
            # `aggregate` is kept only for text_contains and the match summary).
            if expectation.eval is not None and self._judge is not None:
                self._judge.add_assistant_message(delta)
            status, reason = await self._evaluate_aggregate(aggregate, expectation)
            self._trace.log(f"eval: {status} (aggregate={aggregate.strip()!r}) {reason}")
            if status == "pass":
                self._last_match_text = aggregate
                return None
            if status == "fail":
                # Only the judge can affirmatively fail an aggregate.
                return fail(reason, "judge_no")
            # "continue": wait for the next segment, separated by a space so
            # sentences don't run together (e.g. "...that. The weather...").
            aggregate += " "
            last_reason = reason

    async def _match_absent(
        self,
        expectation: EvalExpectation,
        deadline: float,
        budget_ms: int,
        turn_idx: int,
        exp_idx: int,
    ) -> EvalAssertionFailure | None:
        """Inverted match: pass when NO event of this type arrives before the deadline.

        The budget is the whole point here — the expectation holds the turn open
        for ``within_ms`` and succeeds only if the event type stays absent for
        that entire window. An arriving event fails immediately with its content
        in the reason, so a duplicate-output regression shows what the bot said.
        """
        self._trace.log(f"match: expecting NO {expectation.event!r} for {budget_ms}ms")
        try:
            event = await self._stream.next_event(expectation.event, deadline)
        except TimeoutError:
            # The quiet window held: absence confirmed.
            self._last_match_text = f"no {expectation.event!r} for {budget_ms}ms"
            return None
        return EvalAssertionFailure(
            turn_index=turn_idx,
            expectation_index=exp_idx,
            event_name=expectation.event,
            reason=(
                f"expected no {expectation.event!r} within {budget_ms}ms, "
                f"but one arrived: {self._match_summary(event)}"
            ),
            kind="unexpected_event",
        )

    async def _match_function_calls(
        self,
        expectation: EvalExpectation,
        deadline: float,
        turn_idx: int,
        exp_idx: int,
    ) -> EvalAssertionFailure | None:
        """Match every call in a ``function_call`` expectation, in any order.

        Iterates the expectation's ``calls`` (each a name + optional args), claiming
        a matching call for each from the turn's calls (buffered + still arriving).
        Passes only when all are claimed within the budget; otherwise returns a
        failure naming the call that was missing or whose args didn't match.
        """

        def fail(reason: str, kind: str) -> EvalAssertionFailure:
            return EvalAssertionFailure(turn_idx, exp_idx, expectation.event, reason, kind)

        def spec_sig(spec) -> str:
            name = spec.name or "any function"
            if not spec.args:
                return name
            args = ", ".join(f"{k}={v!r}" for k, v in spec.args.items())
            return f"{name}({args})"

        matched: list[str] = []
        for spec in expectation.calls or []:
            want = spec.args or None
            self._trace.log(f"match: waiting for {expectation.event!r} ({spec_sig(spec)})")
            try:
                event = await self._next_function_call(spec.name, deadline, want, expectation.event)
            except TimeoutError:
                # A call of the right name with the wrong arguments is a different
                # failure from the call never being made, and the bot's arguments
                # are what the reader needs to see.
                near = [
                    ev.get("args")
                    for ev in self._pending_function_calls
                    if ev.get("type") == expectation.event
                    and (spec.name is None or ev.get("name") == spec.name)
                ]
                if want is not None and near:
                    return fail(
                        f"no {spec.name!r} call had args {want!r} (saw {near!r})",
                        "function_args_mismatch",
                    )
                missing = spec.name or "any function"
                seen = ", ".join(matched) if matched else "none"
                return fail(
                    f"function call {missing!r} not seen (matched: {seen})",
                    "missing_function_call",
                )
            matched.append(str(event.get("name")))

        self._last_match_text = ", ".join(matched) or "function call"
        return None

    async def _next_function_call(
        self,
        name: str | None,
        deadline: float,
        args: dict | None = None,
        event_type: str = "function_call",
    ) -> dict:
        """Return a call event of ``event_type`` matching ``name`` (``None`` = any) and ``args``.

        A turn's function calls can arrive in any order, so match against the
        per-turn buffer of calls seen but not yet claimed, plus newly arriving
        ones; a call that doesn't match is buffered so another expected call can
        claim it. ``args`` is a subset check and participates in matching, so the
        turn is satisfied by any call matching both name and arguments rather
        than by the first to share the name — which is what lets a call the LLM
        corrects and repeats satisfy it. Other event types are dropped, as in
        :meth:`~pipecat.evals.events.EvalEventStream.next_event`. Raises
        TimeoutError once ``deadline`` passes.
        """

        def matches(ev: dict) -> bool:
            if ev.get("type") != event_type:
                return False
            if name is not None and ev.get("name") != name:
                return False
            if args is None:
                return True
            actual = ev.get("args") or {}
            return all(actual.get(k) == v for k, v in args.items())

        for i, ev in enumerate(self._pending_function_calls):
            if matches(ev):
                return self._pending_function_calls.pop(i)

        while True:
            event = await self._stream.next_any(deadline)
            if event.get("type") not in FUNCTION_CALL_EVENTS:
                continue
            if matches(event):
                return event
            self._pending_function_calls.append(event)

    async def _evaluate_aggregate(
        self, aggregate: str, expectation: EvalExpectation
    ) -> tuple[str, str]:
        """Evaluate the accumulated response text. Returns ``(status, reason)``.

        ``status`` is ``"pass"``, ``"fail"``, or ``"continue"``. ``text_contains``
        is monotonic, so a missing substring is ``"continue"`` (more text may
        arrive); only the judge can affirmatively ``"fail"``.
        """
        if expectation.text_contains is not None and not _text_contains(
            aggregate, expectation.text_contains
        ):
            return ("continue", f"does not contain {expectation.text_contains!r}")

        if expectation.eval is not None:
            if not aggregate.strip():
                return ("continue", "no response text yet")
            # _match_and_verify guarantees a judge exists before aggregating eval:.
            assert self._judge is not None
            with logger.contextualize(eval_pipeline="judge"):
                # The reply segments were added to the judge's conversation in the
                # aggregation loop; the judge evaluates that context, not `aggregate`.
                verdict = await self._judge.evaluate(expectation.eval)
            if verdict.verdict == "no":
                return ("fail", f"judge said no: {verdict.reason}")
            if verdict.verdict == "continue":
                return ("continue", f"judge said continue: {verdict.reason}")
            return ("pass", f"judge said yes: {verdict.reason}")

        return ("pass", "")

    @staticmethod
    def _check_payload(
        event: dict,
        expectation: EvalExpectation,
        turn_idx: int,
        exp_idx: int,
    ) -> EvalAssertionFailure | None:
        """Apply payload-level checks to a matched event. Returns the first failure or None."""

        def fail(reason: str) -> EvalAssertionFailure:
            return EvalAssertionFailure(
                turn_index=turn_idx,
                expectation_index=exp_idx,
                event_name=expectation.event,
                reason=reason,
                kind="text_mismatch",
            )

        if expectation.text_contains is not None:
            content = _event_text(event)
            if not _text_contains(content, expectation.text_contains):
                return fail(f"text {content!r} does not contain {expectation.text_contains!r}")

        return None

    async def _check_judge(
        self,
        event: dict,
        expectation: EvalExpectation,
        turn_idx: int,
        exp_idx: int,
    ) -> EvalAssertionFailure | None:
        """Run the judge assertion if ``eval:`` was set on this expectation."""
        if expectation.eval is None:
            return None

        if self._judge is None:
            return EvalAssertionFailure(
                turn_index=turn_idx,
                expectation_index=exp_idx,
                event_name=expectation.event,
                reason="scenario uses 'eval:' but no judge could be built",
                kind="no_judge",
            )

        content = event.get("text") or event.get("transcript")
        if not content:
            return EvalAssertionFailure(
                turn_index=turn_idx,
                expectation_index=exp_idx,
                event_name=expectation.event,
                reason=f"event has no text/transcript to judge: {event!r}",
                kind="no_content",
            )

        self._judge.add_assistant_message(content)
        verdict = await self._judge.evaluate(expectation.eval)
        if not verdict.passed:
            return EvalAssertionFailure(
                turn_index=turn_idx,
                expectation_index=exp_idx,
                event_name=expectation.event,
                reason=f"eval {expectation.eval!r}: judge said no — {verdict.reason}",
                kind="judge_no",
            )

        return None


def _event_text(event: dict) -> str:
    """The text an event carries: reply events use ``text``, ``user_transcription`` ``transcript``."""
    return event.get("text") or event.get("transcript") or ""


def _text_contains(content: str, needle: str) -> bool:
    """Whether ``needle`` occurs in ``content``, ignoring how either is spaced.

    Aggregated text joins segments with spaces and an STT's pieces may carry
    their own, so a phrase is matched on collapsed whitespace.
    """
    return " ".join(needle.split()) in " ".join(content.split())

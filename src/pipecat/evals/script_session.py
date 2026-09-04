#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Eval session: runs a scripted scenario against a bot and asserts on its behavior.

An :class:`EvalScriptSession` runs an :class:`~pipecat.evals.script.EvalScriptScenario`
over the :class:`~pipecat.evals.base_session.BaseEvalSession` runtime with the
:class:`~pipecat.evals.script_driver.EvalScriptDriver`, which plays the scenario's turns
and matches each turn's expectations, and returns an
:class:`~pipecat.evals.results.EvalScriptResult`.

Example::

    scenario = EvalScriptScenario.load("scenarios/greeting.yaml")
    result = await EvalScriptSession.from_scenario(scenario, "ws://localhost:7860").run()
    if result.passed:
        print("PASS")
    else:
        for f in result.failures:
            print(f"  {f}")

    # Per-turn outcomes, for a scenario scored a turn at a time.
    scored = [t for t in result.turns if t.status != "not_run"]
    print(f"{sum(1 for t in scored if t.status == 'passed')}/{len(scored)} turns")
"""

import warnings
from collections.abc import Callable

from loguru import logger

from pipecat.evals.base_driver import BaseEvalDriver
from pipecat.evals.base_session import BaseEvalSession
from pipecat.evals.client import EvalClient
from pipecat.evals.events import EvalEventStream
from pipecat.evals.judge import EvalJudge
from pipecat.evals.results import EvalScriptResult, EvalScriptTurnProgress
from pipecat.evals.scenario_config import describe_config
from pipecat.evals.script import EvalScriptScenario
from pipecat.evals.script_driver import EvalScriptDriver
from pipecat.evals.services import stt_service_from_config, tts_service_from_config
from pipecat.evals.tts import CachingTTSService
from pipecat.services.stt_service import STTService
from pipecat.utils.deprecation import deprecated

# Generous default so an expectation without an explicit ``within_ms`` waits
# long enough for slow LLM/TTS responses (and function-call round-trips) rather
# than failing on latency. Set ``within_ms`` explicitly to assert on timing.
DEFAULT_EVENT_TIMEOUT_MS = 60000


class EvalScriptSession(BaseEvalSession[EvalScriptResult]):
    """Runs one :class:`~pipecat.evals.script.EvalScriptScenario` against a bot.

    Connects as an RTVI client, drives each turn (sending ``send-text``,
    ``raw-audio``, or ``dtmf``), collects the RTVI events the bot emits, and
    asserts on them. Build one with :meth:`from_scenario` (which constructs the
    judge, user TTS, and STT the scenario needs), then await :meth:`run`.

    Example::

        @session.event_handler("on_progress")
        async def on_progress(session, progress):
            print(progress.event_name, progress.status)
    """

    def __init__(
        self,
        scenario: EvalScriptScenario,
        bot_url: str,
        *,
        connect_timeout_s: float = 5.0,
        default_timeout_ms: int = DEFAULT_EVENT_TIMEOUT_MS,
        on_progress: Callable[[EvalScriptTurnProgress], None] | None = None,
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
            on_progress: Optional callback invoked with a :class:`EvalScriptTurnProgress`
                as each turn and expectation resolves (used for verbose output).

                .. deprecated:: 1.9.0
                    Use the ``on_progress`` event handler instead.
                    Will be removed in 2.0.0.

            record_path: When set (and the scenario is audio mode), the
                conversation audio (both sides) is recorded to this path.
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
        super().__init__(kind="script", name=scenario.name, bot_url=bot_url)
        self._scenario = scenario
        # The bot's output as events: fed by the client's pipeline, read by the driver.
        self._stream = EvalEventStream(bot_audio=scenario.bot_audio, trace=self._trace)
        # The connection to the bot: the eval pipeline and the user's sends.
        self._client = EvalClient.for_scenario(
            scenario,
            bot_url=bot_url,
            stream=self._stream,
            trace=self._trace,
            connect_timeout_s=connect_timeout_s,
            record_path=record_path,
            stop_bot=stop_bot,
            trigger_disconnect=trigger_disconnect,
            user_tts=user_tts,
            bot_stt=bot_stt,
        )
        # What the user says next and how the outcome is scored: a scenario is
        # played by the eval driver.
        self._driver: BaseEvalDriver[EvalScriptResult] = EvalScriptDriver(
            scenario=scenario,
            default_timeout_ms=default_timeout_ms,
            client=self._client,
            stream=self._stream,
            judge=judge,
            trace=self._trace,
            progress=self._progress,
        )
        if on_progress is not None:
            self._add_legacy_progress_callback(on_progress)

    @classmethod
    def from_scenario(
        cls,
        scenario: EvalScriptScenario,
        bot_url: str,
        *,
        connect_timeout_s: float = 5.0,
        default_timeout_ms: int = DEFAULT_EVENT_TIMEOUT_MS,
        on_progress: Callable[[EvalScriptTurnProgress], None] | None = None,
        record_path: str | None = None,
        cache_dir: str | None = None,
        use_cache: bool = True,
        stop_bot: bool = False,
        trigger_disconnect: bool = False,
        judge: EvalJudge | None = None,
        user_tts: CachingTTSService | None = None,
        bot_stt: STTService | None = None,
    ) -> "EvalScriptSession":
        """Build a ready-to-run session from a scenario, constructing what it needs.

        Builds the judge, user TTS, and STT the scenario calls for and injects them
        into a new session. Pass ``judge`` /
        ``user_tts`` / ``bot_stt`` to override any of them with your own pre-built
        instance. Then await :meth:`run`::

            session = EvalScriptSession.from_scenario(scenario, "ws://localhost:7860")
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

        if bot_stt is None and scenario.wants_response() and scenario.bot_audio:
            with logger.contextualize(eval_pipeline="transcription"):
                bot_stt = stt_service_from_config(scenario.transcriber)

        return cls(
            scenario,
            bot_url,
            connect_timeout_s=connect_timeout_s,
            default_timeout_ms=default_timeout_ms,
            on_progress=on_progress,
            record_path=record_path,
            stop_bot=stop_bot,
            trigger_disconnect=trigger_disconnect,
            judge=judge,
            user_tts=user_tts,
            bot_stt=bot_stt,
        )

    def _describe(self) -> str:
        return describe_config(self._scenario)

    def _skip_reason(self) -> str | None:
        # The `response` transcription needs the bot's actual audio; without audio
        # mode there's nothing to transcribe, so skip rather than fail. (Normally
        # unreachable: EvalScriptScenario.load resolves `response` to llm_response in text
        # modality; this guards Scenarios built directly.)
        if self._scenario.wants_response() and not self._scenario.bot_audio:
            return "asserts 'response' transcription but judge modality is text (no audio)"
        return None

    def _add_legacy_progress_callback(
        self, on_progress: Callable[[EvalScriptTurnProgress], None]
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


@deprecated(
    "`EvalSession` is deprecated since 1.9.0 and will be removed in 2.0.0. "
    "Use `EvalScriptSession` instead."
)
class EvalSession(EvalScriptSession):
    """Deprecated alias for :class:`EvalScriptSession`.

    .. deprecated:: 1.9.0
        Use :class:`EvalScriptSession` instead. Will be removed in 2.0.0.
    """

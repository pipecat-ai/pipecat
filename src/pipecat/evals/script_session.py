#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The scripted session: runs a scripted scenario against a bot.

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
from pipecat.evals.client import EvalClient, EvalClientParams
from pipecat.evals.events import EvalEventStream
from pipecat.evals.judge import EvalJudge
from pipecat.evals.results import EvalScriptResult, EvalScriptTurnProgress
from pipecat.evals.scenario import EvalKind
from pipecat.evals.scenario_config import describe_config
from pipecat.evals.script import EvalScriptScenario
from pipecat.evals.script_driver import EvalScriptDriver
from pipecat.evals.services import stt_service_from_config, tts_service_from_config
from pipecat.evals.session import EvalSession
from pipecat.evals.tts import CachingTTSService
from pipecat.services.stt_service import STTService

# Generous default so an expectation without an explicit ``within_ms`` waits
# long enough for slow LLM/TTS responses (and function-call round-trips) rather
# than failing on latency. Set ``within_ms`` explicitly to assert on timing.
DEFAULT_EVENT_TIMEOUT_MS = 60000


class EvalScriptSession(EvalSession[EvalScriptResult]):
    """Runs one :class:`~pipecat.evals.script.EvalScriptScenario` against a bot.

    Build one with :meth:`from_scenario`, which constructs the judge, the user
    TTS, and the STT the scenario needs, then await :meth:`run`.

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
        params: EvalClientParams | None = None,
        default_timeout_ms: int = DEFAULT_EVENT_TIMEOUT_MS,
        on_progress: Callable[[EvalScriptTurnProgress], None] | None = None,
        judge: EvalJudge | None = None,
    ):
        """Initialize the eval session.

        The services come pre-built: :meth:`from_scenario` constructs the
        defaults, the judge here and the user TTS and bot STT in ``params``. Pass
        your own to override them.

        Args:
            scenario: The parsed scenario to run.
            bot_url: WebSocket URL of the bot's eval transport.
            params: How the run talks to the bot (timeouts, recording, teardown)
                and the services in its pipeline, an
                :class:`~pipecat.evals.client.EvalClientParams`.
            default_timeout_ms: Per-expectation latency budget for expectations
                without their own ``within_ms`` (the turn's expectations share one
                deadline anchored at the send). Defaults to 60s.
            on_progress: Optional callback invoked with a :class:`EvalScriptTurnProgress`
                as each turn and expectation resolves (used for verbose output).

                .. deprecated:: 1.9.0
                    Use the ``on_progress`` event handler instead.
                    Will be removed in 2.0.0.

            judge: The :class:`~pipecat.evals.judge.EvalJudge` for ``eval:``
                assertions, or ``None`` if the scenario has none.
        """
        super().__init__(kind=EvalKind.SCRIPT, name=scenario.name, bot_url=bot_url)
        self._scenario = scenario
        # The bot's output as events: fed by the client's pipeline, read by the driver.
        self._stream = EvalEventStream(bot_audio=scenario.bot_audio, trace=self._trace)
        # The connection to the bot: the eval pipeline and the user's sends.
        self._client = EvalClient.for_scenario(
            scenario,
            bot_url,
            params=params,
            stream=self._stream,
            trace=self._trace,
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
        """Build a ready-to-run session from a scenario, constructing the services it needs.

        Pass ``judge``, ``user_tts``, or ``bot_stt`` to use your own. Then await
        :meth:`run`::

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

        params = EvalClientParams(
            connect_timeout_s=connect_timeout_s,
            record_path=record_path,
            stop_bot=stop_bot,
            trigger_disconnect=trigger_disconnect,
            user_tts=user_tts,
            bot_stt=bot_stt,
        )
        return cls(
            scenario,
            bot_url,
            params=params,
            default_timeout_ms=default_timeout_ms,
            on_progress=on_progress,
            judge=judge,
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
        """Register a bare ``on_progress`` callback as an event handler, dropping the session argument."""
        warnings.warn(
            "`on_progress` is deprecated since 1.9.0 and will be removed in 2.0.0. "
            "Use the `on_progress` event handler instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        self.add_event_handler("on_progress", lambda _session, record: on_progress(record))

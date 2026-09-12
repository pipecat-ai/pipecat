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
from pipecat.evals.session import EvalSession, EvalSessionParams, _params_with_deprecated_knobs
from pipecat.evals.timing import EvalTimingObserver
from pipecat.evals.tts import CachingTTSService
from pipecat.services.stt_service import STTService


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
        params: EvalSessionParams | None = None,
        on_progress: Callable[[EvalScriptTurnProgress], None] | None = None,
        judge: EvalJudge | None = None,
        user_tts: CachingTTSService | None = None,
        bot_stt: STTService | None = None,
    ):
        """Initialize the eval session.

        The services come pre-built, or ``None`` where the scenario has no use
        for them; :meth:`from_scenario` constructs the ones a scenario needs.

        Args:
            scenario: The parsed scenario to run.
            bot_url: WebSocket URL of the bot's eval transport.
            params: How the run behaves; ``None`` for the defaults.
            on_progress: Optional callback invoked with a :class:`EvalScriptTurnProgress`
                as each turn and expectation resolves (used for verbose output).

                .. deprecated:: 1.9.0
                    Use the ``on_progress`` event handler instead.
                    Will be removed in 2.0.0.

            judge: The :class:`~pipecat.evals.judge.EvalJudge` for ``eval:``
                assertions, or ``None`` if the scenario has none.
            user_tts: The user-audio TTS, or ``None`` for text mode.
            bot_stt: The bot-audio STT for the ``response`` transcription, or
                ``None`` when the scenario has none.
        """
        super().__init__(kind=EvalKind.SCRIPT, name=scenario.name, bot_url=bot_url, params=params)
        self._scenario = scenario
        # The bot's output as events: fed by the client's pipeline, read by the driver.
        self._stream = EvalEventStream(bot_audio=scenario.bot_audio, trace=self._trace)
        # When each turn's reply happened: measured on the client's pipeline,
        # told by the driver when a turn's input went out.
        self._timing = EvalTimingObserver()
        # The connection to the bot: the eval pipeline and the user's sends,
        # asking the bot for what the scenario's assertions need.
        self._client = EvalClient(
            bot_url,
            params=EvalClientParams(
                bot_audio=scenario.bot_audio,
                user_audio=scenario.user_audio,
                user_speech=scenario.user_speech,
                capture_bot_audio=scenario.wants_response(),
                report_level=scenario.required_report_level(),
                vad_events=scenario.needs_vad_events(),
                context=list(scenario.context or []),
                trigger_disconnect=scenario.trigger_disconnect,
            ),
            session_params=self._params,
            stream=self._stream,
            trace=self._trace,
            user_tts=user_tts,
            bot_stt=bot_stt,
            observers=[self._timing],
        )
        # What the user says next and how the outcome is scored: a scenario is
        # played by the eval driver.
        self._driver: BaseEvalDriver[EvalScriptResult] = EvalScriptDriver(
            scenario=scenario,
            default_timeout_ms=self._params.default_timeout_ms,
            client=self._client,
            stream=self._stream,
            judge=judge,
            trace=self._trace,
            progress=self._progress,
            timing=self._timing,
        )
        if on_progress is not None:
            self._add_legacy_progress_callback(on_progress)

    @classmethod
    def from_scenario(
        cls,
        scenario: EvalScriptScenario,
        bot_url: str,
        *,
        params: EvalSessionParams | None = None,
        on_progress: Callable[[EvalScriptTurnProgress], None] | None = None,
        judge: EvalJudge | None = None,
        user_tts: CachingTTSService | None = None,
        bot_stt: STTService | None = None,
        connect_timeout_s: float | None = None,
        default_timeout_ms: int | None = None,
        record_path: str | None = None,
        cache_dir: str | None = None,
        use_cache: bool | None = None,
        stop_bot: bool | None = None,
        trigger_disconnect: bool | None = None,
    ) -> "EvalScriptSession":
        """Build a ready-to-run session from a scenario, constructing the services it needs.

        Pass ``judge``, ``user_tts``, or ``bot_stt`` to use your own. Then await
        :meth:`run`::

            session = EvalScriptSession.from_scenario(scenario, "ws://localhost:7860")
            result = await session.run()

        Args:
            scenario: The parsed scenario to run.
            bot_url: WebSocket URL of the bot's eval transport.
            params: How the run behaves; ``None`` for the defaults.
            on_progress: Optional per-turn/expectation progress callback (verbose).

                .. deprecated:: 1.9.0
                    Use the ``on_progress`` event handler instead.
                    Will be removed in 2.0.0.

            judge: Override the judge (default: built from ``scenario.judge`` when the
                scenario has ``eval:`` assertions).
            user_tts: Override the user-audio TTS (default: built from
                ``scenario.user_speech`` in audio mode).
            bot_stt: Override the bot-audio STT (default: built from
                ``scenario.transcriber`` when the scenario asserts ``response``).
            connect_timeout_s: The ``params`` field of the same name.

                .. deprecated:: 1.9.0
                    Use ``params`` instead. Will be removed in 2.0.0.

            default_timeout_ms: The ``params`` field of the same name.

                .. deprecated:: 1.9.0
                    Use ``params`` instead. Will be removed in 2.0.0.

            record_path: The ``params`` field of the same name.

                .. deprecated:: 1.9.0
                    Use ``params`` instead. Will be removed in 2.0.0.

            cache_dir: The ``params`` field of the same name.

                .. deprecated:: 1.9.0
                    Use ``params`` instead. Will be removed in 2.0.0.

            use_cache: The ``params`` field of the same name.

                .. deprecated:: 1.9.0
                    Use ``params`` instead. Will be removed in 2.0.0.

            stop_bot: The ``params`` field of the same name.

                .. deprecated:: 1.9.0
                    Use ``params`` instead. Will be removed in 2.0.0.

            trigger_disconnect: The ``params`` field of the same name.

                .. deprecated:: 1.9.0
                    Use ``params`` instead. Will be removed in 2.0.0.

        Returns:
            A configured session, ready for :meth:`run`.
        """
        params = _params_with_deprecated_knobs(
            params,
            "EvalScriptSession.from_scenario",
            connect_timeout_s=connect_timeout_s,
            default_timeout_ms=default_timeout_ms,
            record_path=record_path,
            cache_dir=cache_dir,
            use_cache=use_cache,
            stop_bot=stop_bot,
            trigger_disconnect=trigger_disconnect,
        )
        turns = scenario.turns
        if judge is None and any(exp.eval is not None for turn in turns for exp in turn.expect):
            with logger.contextualize(eval_pipeline="judge"):
                judge = EvalJudge.from_config(scenario.judge)

        if user_tts is None and scenario.user_speech is not None:
            with logger.contextualize(eval_pipeline="speech"):
                user_tts = tts_service_from_config(
                    scenario.user_speech, cache_dir=params.cache_dir, use_cache=params.use_cache
                )

        if bot_stt is None and scenario.wants_response() and scenario.bot_audio:
            with logger.contextualize(eval_pipeline="transcription"):
                bot_stt = stt_service_from_config(scenario.transcriber)

        return cls(
            scenario,
            bot_url,
            params=params,
            on_progress=on_progress,
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
        """Register a bare ``on_progress`` callback as an event handler, dropping the session argument."""
        warnings.warn(
            "`on_progress` is deprecated since 1.9.0 and will be removed in 2.0.0. "
            "Use the `on_progress` event handler instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        self.add_event_handler("on_progress", lambda _session, record: on_progress(record))

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Simulation session: lets a persona hold a conversation with a bot and judges it.

A :class:`EvalSimulationSession` runs an
:class:`~pipecat.evals.simulation.EvalSimulationScenario` over the
:class:`~pipecat.evals.base_session.BaseEvalSession` runtime with the
:class:`~pipecat.evals.simulation_driver.EvalSimulationDriver`: the persona LLM rides in the
pipeline and answers the bot on its own, the judge decides the goal and the
quality criteria, and the result is a
:class:`~pipecat.evals.results.EvalSimulationResult`.

Example::

    scenario = EvalSimulationScenario.load("scenarios/simulated/curious_caller.yaml")
    run = await EvalSimulationSession.from_scenario(scenario, "ws://localhost:7860").run()
    print(f"{'succeeded' if run.succeeded else 'failed'}: {run.reason}")
"""

from loguru import logger

from pipecat.evals.base_driver import BaseEvalDriver
from pipecat.evals.base_session import BaseEvalSession
from pipecat.evals.client import EvalClient, EvalClientParams
from pipecat.evals.events import EvalEventStream
from pipecat.evals.judge import EvalJudge
from pipecat.evals.persona import EvalPersona
from pipecat.evals.results import EvalSimulationResult
from pipecat.evals.scenario import EvalKind
from pipecat.evals.services import (
    llm_service_from_config,
    stt_service_from_config,
    tts_service_from_config,
)
from pipecat.evals.simulation import EvalSimulationScenario, describe_simulation
from pipecat.evals.simulation_driver import EvalSimulationDriver
from pipecat.evals.tts import CachingTTSService
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService


class EvalSimulationSession(BaseEvalSession[EvalSimulationResult]):
    """Runs one :class:`~pipecat.evals.simulation.EvalSimulationScenario` against a bot.

    The persona LLM rides in the client's pipeline and answers the bot on its
    own; the session watches for the conversation's end and has the judge
    decide the goal and the quality criteria. Build one with
    :meth:`from_scenario` (which constructs the persona LLM, judge, user TTS,
    and STT the simulation needs), then await :meth:`run`.
    """

    def __init__(
        self,
        scenario: EvalSimulationScenario,
        bot_url: str,
        *,
        params: EvalClientParams | None = None,
        persona_llm: LLMService,
        judge: EvalJudge | None,
        user_tts: CachingTTSService | None = None,
        bot_stt: STTService | None = None,
    ):
        """Initialize the simulation session.

        Args:
            scenario: The parsed simulation to run.
            bot_url: WebSocket URL of the bot's eval transport.
            params: How the run talks to the bot (timeouts, recording,
                teardown), an :class:`~pipecat.evals.client.EvalClientParams`.
            persona_llm: The persona LLM service, run inside the eval pipeline.
            judge: The judge for the goal and the quality criteria, or ``None``
                (the run then reports no verdict).
            user_tts: The TTS that speaks the persona's turns in audio mode, or
                ``None`` for text mode.
            bot_stt: The STT that transcribes the bot's audio for the persona in
                audio mode, or ``None`` for text mode.
        """
        super().__init__(kind=EvalKind.SIMULATION, name=scenario.name, bot_url=bot_url)
        self._scenario = scenario
        persona = EvalPersona(scenario.persona, scenario.goal, persona_llm)
        self._stream = EvalEventStream(bot_audio=scenario.bot_audio, trace=self._trace)
        self._client = EvalClient.for_simulation(
            scenario,
            bot_url=bot_url,
            stream=self._stream,
            trace=self._trace,
            params=params,
            user_tts=user_tts,
            bot_stt=bot_stt,
            persona=persona,
        )
        self._driver: BaseEvalDriver[EvalSimulationResult] = EvalSimulationDriver(
            simulation=scenario,
            persona=persona,
            client=self._client,
            stream=self._stream,
            judge=judge,
            trace=self._trace,
            progress=self._progress,
        )

    @classmethod
    def from_scenario(
        cls,
        scenario: EvalSimulationScenario,
        bot_url: str,
        *,
        connect_timeout_s: float = 5.0,
        record_path: str | None = None,
        cache_dir: str | None = None,
        use_cache: bool = True,
        stop_bot: bool = False,
        trigger_disconnect: bool = False,
        persona_llm: LLMService | None = None,
        judge: EvalJudge | None = None,
        user_tts: CachingTTSService | None = None,
        bot_stt: STTService | None = None,
    ) -> "EvalSimulationSession":
        """Build a ready-to-run session from a scenario, constructing what it needs.

        Builds the persona LLM, the judge, and in audio mode the user TTS and
        the STT from the simulation's config; pass any of them to use your own.

        Args:
            scenario: The parsed simulation to run.
            bot_url: WebSocket URL of the bot's eval transport.
            connect_timeout_s: How long to wait for the bot to accept the WS
                connection before giving up.
            record_path: Optional path to record the conversation audio (audio mode).
            cache_dir: Optional directory for cached synthesized user audio.
            use_cache: When False, ignore cached user audio and force fresh synthesis.
            stop_bot: When True, ask the bot to cancel its pipeline on teardown.
            trigger_disconnect: When True, fire the bot's ``on_client_disconnected``
                handler when the connection ends.
            persona_llm: Override the persona LLM (default: built from
                ``simulation.simulator``).
            judge: Override the judge (default: built from ``simulation.judge``).
            user_tts: Override the user-audio TTS (default: built from
                ``simulation.user_speech`` in audio mode).
            bot_stt: Override the bot-audio STT (default: built from
                ``simulation.transcriber`` in audio mode).

        Returns:
            A configured session, ready for :meth:`run`.
        """
        if persona_llm is None:
            with logger.contextualize(eval_pipeline="persona"):
                persona_llm = llm_service_from_config(scenario.simulator, where="simulator")
        if judge is None:
            with logger.contextualize(eval_pipeline="judge"):
                judge = EvalJudge.from_config(scenario.judge)
        if user_tts is None and scenario.user_speech is not None:
            with logger.contextualize(eval_pipeline="speech"):
                user_tts = tts_service_from_config(
                    scenario.user_speech, cache_dir=cache_dir, use_cache=use_cache
                )
        if bot_stt is None and scenario.bot_audio:
            with logger.contextualize(eval_pipeline="transcription"):
                bot_stt = stt_service_from_config(scenario.transcriber)
        params = EvalClientParams(
            connect_timeout_s=connect_timeout_s,
            record_path=record_path,
            stop_bot=stop_bot,
            trigger_disconnect=trigger_disconnect,
        )
        return cls(
            scenario,
            bot_url,
            params=params,
            persona_llm=persona_llm,
            judge=judge,
            user_tts=user_tts,
            bot_stt=bot_stt,
        )

    def _describe(self) -> str:
        return describe_simulation(self._scenario)

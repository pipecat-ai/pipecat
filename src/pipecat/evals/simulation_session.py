#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The simulation session: runs a simulated scenario against a bot.

Example::

    scenario = EvalSimulationScenario.load("scenarios/simulated/curious_caller.yaml")
    run = await EvalSimulationSession.from_scenario(scenario, "ws://localhost:7860").run()
    print(f"{'succeeded' if run.succeeded else 'failed'}: {run.reason}")
"""

from loguru import logger

from pipecat.evals.base_driver import BaseEvalDriver
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
from pipecat.evals.session import EvalSession, EvalSessionParams
from pipecat.evals.simulation import EvalSimulationScenario, describe_simulation
from pipecat.evals.simulation_driver import EvalSimulationDriver
from pipecat.evals.tts import CachingTTSService
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService


class EvalSimulationSession(EvalSession[EvalSimulationResult]):
    """Runs one :class:`~pipecat.evals.simulation.EvalSimulationScenario` against a bot.

    The persona LLM answers the bot on its own inside the client's pipeline,
    and the judge decides the goal and the criteria at the end. Build one
    with :meth:`from_scenario`, which constructs the persona LLM, the judge,
    and in audio mode the user TTS and the STT, then await :meth:`run`.
    """

    def __init__(
        self,
        scenario: EvalSimulationScenario,
        bot_url: str,
        *,
        params: EvalSessionParams | None = None,
        persona_llm: LLMService,
        judge: EvalJudge | None,
        user_tts: CachingTTSService | None = None,
        bot_stt: STTService | None = None,
    ):
        """Initialize the simulation session.

        Args:
            scenario: The parsed simulation to run.
            bot_url: WebSocket URL of the bot's eval transport.
            params: How the run behaves; ``None`` for the defaults.
            persona_llm: The persona LLM service, run inside the eval pipeline.
            judge: The judge for the goal and the quality criteria, or ``None``
                (the run then reports no verdict).
            user_tts: The user-audio TTS, or ``None`` for text mode.
            bot_stt: The bot-audio STT, or ``None`` in text mode.
        """
        super().__init__(
            kind=EvalKind.SIMULATION, name=scenario.name, bot_url=bot_url, params=params
        )
        self._scenario = scenario
        persona = EvalPersona(scenario.persona, scenario.goal, persona_llm)
        self._stream = EvalEventStream(bot_audio=scenario.bot_audio, trace=self._trace)
        # The persona hears the bot, and the judge sees its tool calls.
        self._client = EvalClient(
            bot_url,
            params=EvalClientParams(
                bot_audio=scenario.bot_audio,
                user_audio=scenario.user_audio,
                user_speech=scenario.user_speech,
                capture_bot_audio=scenario.bot_audio,
                # The bot's function calls, with their arguments, are the judge's
                # evidence of what the bot actually did.
                report_level="full",
                trigger_disconnect=scenario.trigger_disconnect,
            ),
            session_params=self._params,
            stream=self._stream,
            trace=self._trace,
            persona=persona,
            user_tts=user_tts,
            bot_stt=bot_stt,
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
        params: EvalSessionParams | None = None,
        persona_llm: LLMService | None = None,
        judge: EvalJudge | None = None,
        user_tts: CachingTTSService | None = None,
        bot_stt: STTService | None = None,
    ) -> "EvalSimulationSession":
        """Build a ready-to-run session from a scenario, constructing the services it needs.

        Pass ``persona_llm``, ``judge``, ``user_tts``, or ``bot_stt`` to use your own.

        Args:
            scenario: The parsed simulation to run.
            bot_url: WebSocket URL of the bot's eval transport.
            params: How the run behaves; ``None`` for the defaults.
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
        params = params or EvalSessionParams()
        if persona_llm is None:
            with logger.contextualize(eval_pipeline="persona"):
                persona_llm = llm_service_from_config(scenario.simulator, where="simulator")
        if judge is None:
            with logger.contextualize(eval_pipeline="judge"):
                judge = EvalJudge.from_config(scenario.judge)
        if user_tts is None and scenario.user_speech is not None:
            with logger.contextualize(eval_pipeline="speech"):
                user_tts = tts_service_from_config(
                    scenario.user_speech, cache_dir=params.cache_dir, use_cache=params.use_cache
                )
        if bot_stt is None and scenario.bot_audio:
            with logger.contextualize(eval_pipeline="transcription"):
                bot_stt = stt_service_from_config(scenario.transcriber)
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

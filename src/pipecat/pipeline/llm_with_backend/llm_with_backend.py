#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The :class:`LLMWithBackend` pipeline, which puts a frontend and its backend where an LLM goes."""

import asyncio
from typing import Any

from loguru import logger

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.pipeline.llm_with_backend.connector import BackendConnector, ConnectorContext
from pipecat.pipeline.llm_with_backend.predictive_delegation import PredictiveDelegation
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.llm_service import LLMService
from pipecat.workers.llm.backend_llm_worker import BackendLLMWorker


class LLMWithBackend(Pipeline):
    """A conversational LLM, kept fast and light, with a backend that does the work it hands off.

    **Alpha.** The API is unstable: names, arguments, the strategies and the
    job contract are likely to change between releases.

    Put it where the LLM goes in a pipeline. It is a :class:`Pipeline`, not an
    :class:`LLMService`: functions, settings updates, event handlers and a
    ``FlowManager`` go on :attr:`frontend`. It wraps the frontend service,
    installs the connector's tools on it as built-in tools, appends the
    connector's guidance to the frontend's system instruction, adds a local
    backend worker to the pipeline worker so the app never wires it up, and
    holds the session with the backend for the pipeline's life. A built-in
    tool is sent on every inference beside whatever tools the frontend has,
    and never enters the context's tool set, so a tool change announced to
    the model never mentions it. The tools are the backend's; the frontend
    has ``delegate`` and no more, though a tool
    it must keep, in its context or configured on the service, stays.

    The guidance is the connector's pairing instruction (how to work with a
    backend: when and how to delegate, and what to do with the backend's
    messages) with the strategy's (how the request is worded) in place. The
    frontend's own system instruction is the place to say what the backend is
    for, in plain words, as the backend's own instruction does: no tool names,
    nothing to update when a tool changes.

    Example::

        llm = LLMWithBackend(
            frontend=OpenAILLMService(...),
            backend=BackendLLMWorker(
                llm=AnthropicLLMService(...),
                context=LLMContext(tools=[get_weather]),
            ),
        )
        pipeline = Pipeline([transport.input(), stt, user_agg, llm, tts, transport.output(), assistant_agg])
    """

    def __init__(
        self,
        *,
        frontend: LLMService[Any],
        backend: BackendLLMWorker | str,
        connector: BackendConnector | None = None,
        predictive_delegation: PredictiveDelegation | None = None,
    ):
        """Initialize the service.

        Args:
            frontend: The conversational LLM service, text or speech-to-speech.
            backend: The backend worker, or the name of one registered
                elsewhere (in the app, or in another process on a shared bus).
                A worker given here is added to the pipeline worker at setup.
            connector: How delegation works. A default
                :class:`BackendConnector` when omitted.
            predictive_delegation: A :class:`PredictiveDelegation`, to speed
                up responses that need backend involvement: its classifier
                determines whether to wait for the frontend's inference or go
                ahead and trigger an early backend delegation. Only for a text
                frontend.

        Raises:
            ValueError: If the frontend service declines the role; if the
                connector's request strategy reads the frontend's context and
                the frontend is speech-to-speech; or if predictive delegation
                is given with a speech-to-speech frontend or with a request
                strategy whose ``delegate`` call needs arguments.
        """
        logger.warning("LLMWithBackend is alpha: its API is likely to change between releases.")
        if objection := frontend.llm_with_backend_role_objection("frontend"):
            raise ValueError(objection)
        self._frontend = frontend
        self._backend = backend
        self._connector = connector or BackendConnector()
        frontend_is_realtime = frontend.service_metadata_frame().is_realtime_service
        self._connector.bind(
            ConnectorContext(
                backend_name=backend if isinstance(backend, str) else backend.name,
                frontend_is_realtime=frontend_is_realtime,
            )
        )
        if predictive_delegation is not None:
            if frontend_is_realtime:
                raise ValueError(
                    "Predictive delegation needs a text frontend: a speech-to-speech frontend "
                    "replies to the user's audio before the turn's transcript reaches its "
                    "context, so there is no turn to classify in time."
                )
            if self._connector.request_strategy.delegate_needs_arguments:
                raise ValueError(
                    f"Predictive delegation makes its delegate call with no arguments, which "
                    f"{type(self._connector.request_strategy).__name__} can't build a request "
                    "from. Use a strategy that needs none, such as "
                    "TranscriptBackendRequestStrategy."
                )
        if instruction := self._connector.frontend_instruction:
            frontend.append_system_instruction(instruction)
        for tool in self._connector.tools:
            frontend.register_function(tool.name, tool.handler, cancel_on_interruption=False)
            frontend.get_llm_adapter().builtin_tools[tool.name] = tool
        self._predictive_delegation = predictive_delegation
        self._session_task: asyncio.Task | None = None
        if predictive_delegation is not None:
            predictive_delegation._bind(frontend, self._connector)
            super().__init__([predictive_delegation, frontend])
        else:
            super().__init__([frontend])

    @property
    def frontend(self) -> LLMService[Any]:
        """The frontend LLM service."""
        return self._frontend

    @property
    def backend(self) -> BackendLLMWorker | str:
        """The backend worker, or its name."""
        return self._backend

    @property
    def connector(self) -> BackendConnector:
        """The connector joining the two layers."""
        return self._connector

    @property
    def predictive_delegation(self) -> PredictiveDelegation | None:
        """The PredictiveDelegation in front of the frontend, if there is one."""
        return self._predictive_delegation

    @property
    def tool(self) -> FunctionSchema:
        """The ``delegate`` tool installed on the frontend."""
        return self._connector.tool

    async def setup(self, setup: FrameProcessorSetup):
        """Register a local backend worker as a child of the pipeline worker, set up the frontend, and open the session."""
        # Before setting the frontend up: it gets the StartFrame as soon as its
        # own setup is done, and can delegate from then on.
        if isinstance(self._backend, BackendLLMWorker):
            await setup.pipeline_worker.add_workers(self._backend)
        await super().setup(setup)
        self._session_task = self.create_task(
            self._connector.run_session(self.pipeline_worker, self._frontend),
            f"{self}::backend_session",
        )

    async def cleanup(self):
        """Close the session with the backend, then clean up the frontend."""
        if self._session_task is not None:
            await self.cancel_task(self._session_task)
            self._session_task = None
        await super().cleanup()

#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM switcher for switching between different LLMs at runtime, with different switching strategies."""

from typing import Any, Generic, List, Optional, Type, TypeVar

from pipecat.pipeline.parallel_pipeline import ParallelPipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.filters.function_filter import FunctionFilter
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService


class LLMSwitcherStrategy:
    """Base class for LLM switching strategies."""

    def __init__(self, llms: List[LLMService]):
        """Initialize the LLM switcher strategy with a list of LLM services."""
        self.llms = llms
        self.active_llm: Optional[LLMService] = None

    def is_active(self, llm: LLMService) -> bool:
        """Determine if the given LLM is the currently active one.

        This method should be overridden by subclasses to implement specific logic.

        Args:
            llm: The LLM service to check.

        Returns:
            True if the given LLM is the active one, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement this method.")


StrategyType = TypeVar("StrategyType", bound=LLMSwitcherStrategy)


class LLMSwitcher(ParallelPipeline, Generic[StrategyType]):
    """A pipeline that switches between different LLMs at runtime."""

    def __init__(self, llms: List[LLMService], strategy_type: Type[StrategyType]):
        """Initialize the LLM switcher with a list of LLM services and a switching strategy."""
        strategy = strategy_type(llms)
        super().__init__(*LLMSwitcher._make_pipeline_definitions(llms, strategy))
        self.llms = llms
        self.strategy = strategy

    async def run_inference(
        self, context: LLMContext, system_instruction: Optional[str] = None
    ) -> Optional[str]:
        """Run a one-shot, out-of-band (i.e. out-of-pipeline) inference with the given LLM context, using the currently active LLM.

        Args:
            context: The LLM context containing conversation history.
            system_instruction: Optional system instruction to guide the LLM's
              behavior. You could also (again, optionally) provide a system
              instruction directly in the context. If both are provided, the
              one in the context takes precedence.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        if self.strategy.active_llm:
            return await self.strategy.active_llm.run_inference(
                context=context, system_instruction=system_instruction
            )
        return None

    def register_function(
        self,
        function_name: Optional[str],
        handler: Any,
        start_callback=None,
        *,
        cancel_on_interruption: bool = True,
    ):
        """Register a function handler for LLM function calls, on all LLMs, active or not.

        Args:
            function_name: The name of the function to handle. Use None to handle
                all function calls with a catch-all handler.
            handler: The function handler. Should accept a single FunctionCallParams
                parameter.
            start_callback: Legacy callback function (deprecated). Put initialization
                code at the top of your handler instead.

                .. deprecated:: 0.0.59
                    The `start_callback` parameter is deprecated and will be removed in a future version.

            cancel_on_interruption: Whether to cancel this function call when an
                interruption occurs. Defaults to True.
        """
        for llm in self.llms:
            llm.register_function(
                function_name=function_name,
                handler=handler,
                start_callback=start_callback,
                cancel_on_interruption=cancel_on_interruption,
            )

    @staticmethod
    def _make_pipeline_definitions(
        llms: List[LLMService], strategy: LLMSwitcherStrategy
    ) -> List[Any]:
        pipelines = []
        for llm in llms:
            pipelines.append(LLMSwitcher._make_pipeline_definition(llm, strategy))
        return pipelines

    @staticmethod
    def _make_pipeline_definition(llm: LLMService, strategy: LLMSwitcherStrategy) -> Any:
        async def filter(frame) -> bool:
            # frame is intentionally unused, but required by the interface
            _ = frame
            return strategy.is_active(llm)

        return [
            FunctionFilter(filter, direction=FrameDirection.DOWNSTREAM),
            llm,
            FunctionFilter(filter, direction=FrameDirection.UPSTREAM),
        ]


class LLMSwitcherStrategyManual(LLMSwitcherStrategy):
    """A strategy for switching between LLMs manually.

    This strategy allows the user to manually select which LLM is active.
    The initial active LLM is the first one in the list.
    """

    def __init__(self, llms: List[LLMService]):
        """Initialize the manual LLM switcher strategy with a list of LLM services."""
        super().__init__(llms)
        self.active_llm = llms[0] if llms else None

    def is_active(self, llm: LLMService) -> bool:
        """Check if the given LLM is the currently active one.

        Args:
            llm: The LLM service to check.

        Returns:
            True if the given LLM is the active one, False otherwise.
        """
        return llm == self.active_llm

    def set_active(self, llm: LLMService):
        """Set the active LLM to the given one.

        Args:
            llm: The LLM service to set as active.
        """
        if llm in self.llms:
            self.active_llm = llm
        else:
            raise ValueError(f"LLM {llm} is not in the list of available LLMs.")

#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""LLM switcher for switching between different LLMs at runtime, with different switching strategies."""

from typing import Any, List, Optional, Type

from pipecat.adapters.schemas.direct_function import DirectFunction
from pipecat.pipeline.service_switcher import ServiceSwitcher, StrategyType
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService


class LLMSwitcher(ServiceSwitcher[StrategyType]):
    """A pipeline that switches between different LLMs at runtime.

    Example::

        llm_switcher = LLMSwitcher(
            llms=[openai_llm, anthropic_llm],
            strategy_type=ServiceSwitcherStrategyManual
        )
    """

    def __init__(self, llms: List[LLMService], strategy_type: Type[StrategyType]):
        """Initialize the service switcher with a list of LLMs and a switching strategy.

        Args:
            llms: List of LLM services to switch between.
            strategy_type: The strategy class to use for switching between LLMs.
        """
        super().__init__(llms, strategy_type)

    @property
    def llms(self) -> List[LLMService]:
        """Get the list of LLMs managed by this switcher.

        Returns:
            List of LLM services managed by this switcher.
        """
        return self.services

    @property
    def active_llm(self) -> Optional[LLMService]:
        """Get the currently active LLM.

        Returns:
            The currently active LLM service, or None if no LLM is active.
        """
        return self.strategy.active_service

    async def run_inference(self, context: LLMContext) -> Optional[str]:
        """Run a one-shot, out-of-band (i.e. out-of-pipeline) inference with the given LLM context, using the currently active LLM.

        Args:
            context: The LLM context containing conversation history.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        if self.active_llm:
            return await self.active_llm.run_inference(context=context)
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

    def register_direct_function(
        self,
        handler: DirectFunction,
        *,
        cancel_on_interruption: bool = True,
    ):
        """Register a direct function handler for LLM function calls, on all LLMs, active or not.

        Args:
            handler: The direct function to register. Must follow DirectFunction protocol.
            cancel_on_interruption: Whether to cancel this function call when an
                interruption occurs. Defaults to True.
        """
        for llm in self.llms:
            llm.register_direct_function(
                handler=handler,
                cancel_on_interruption=cancel_on_interruption,
            )

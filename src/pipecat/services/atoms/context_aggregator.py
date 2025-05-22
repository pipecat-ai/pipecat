from pipecat.processors.aggregators.llm_response import (
    LLMAssistantContextAggregator,
    LLMUserContextAggregator,
)


class AtomsAgentUserContextAggregator(LLMUserContextAggregator):
    """This class is responsible for aggregating the user context from the agent."""

    pass


class AtomsAgentAssistantContextAggregator(LLMAssistantContextAggregator):
    """This class is responsible for aggregating the assistant context from the agent."""

    pass

class HoppingAgentService:
    """
    This service is responsible for hopping the agent.
    It will be used to hop the agent passed from the agent context aggregator service
    It can hop the agent to a new node or can remain at the same node
    """

    pass


class NodeActionService:
    """
    This service is responsible for taking actions on a node.
    It will be used to take actions on a GraphNode passed from the hopping agent service
    It can end the call
    It can transfer the call
    It can call a function on the node
    """

    pass


class AgentContextAggregatorService:
    """
    This service is responsible for aggregating the context of the agent.
    It will save all the context and pass it to the LLM
    It will be attached after the NodeActionService and after the AgentReponseService
    """

    pass


class AgentReponseService:
    """
    This service is responsible for calling LLM based on the prompt given by the node action service.
    Node action service can also bypass the LLM and send a direct response
    """

    pass


class KnowledgeBaseAdapter:
    """
    This service is responsible for adapting the knowledge base to the agent.
    It will be used to adapt the knowledge base to the agent passed from the hopping agent service
    It can adapt the knowledge base to the agent
    """

    pass

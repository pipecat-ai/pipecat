"""This module contains the initialize_conversational_agent function, which is used to initialize a conversational agent."""

import json
import os
from enum import Enum
from typing import Any, Dict

import httpx
from loguru import logger
from models.agent import AtomsLLMModels, CallData
from utils import get_unallowed_variable_names

from pipecat.processors.filters.custom_mute_filter import TransportInputFilter
from pipecat.services.atoms.agent import FlowGraphManager
from pipecat.services.atoms.llm_client import AzureOpenAIClient, OpenAIClient
from pipecat.services.atoms.pathways import ConversationalPathway
from pipecat.services.atoms.prompts import (
    FT_RESPONSE_MODEL_SYSTEM_PROMPT,
    GENERAL_RESPONSE_MODEL_SYSTEM_PROMPT,
)
from pipecat.services.atoms.utils import convert_old_to_new_format
from pipecat.services.documentdb.rag import DocumentDBVectorStore


class CallType(Enum):
    """Call type enum."""

    TELEPHONY_INBOUND = "telephony_inbound"
    TELEPHONY_OUTBOUND = "telephony_outbound"
    WEBCALL = "webcall"
    CHAT = "chat"


async def initialize_conversational_agent(
    *, agent_id: str, call_data: CallData, transport_input_filter: Any
) -> tuple[FlowGraphManager, Dict[str, Any]]:
    """Initialize a conversational agent with the specified configuration.

    Args:
        agent_id: ID of the agent to initialize
        call_id: Call ID for logging
        call_data: Contains variables and other call-related information
        initialize_first_message: Whether to initialize first message
        save_msgs_path: Path to save messages

    Returns:
        tuple: (initialized agent instance, agent configuration)

    Raises:
        ValueError: If agent_id is not provided
        Exception: If initialization fails or variables are not provided
    """
    if call_data.variables is None:
        raise Exception("Variables is required to initialize conversational agent")

    try:
        # Initialize conversational pathway
        conv_pathway_data, agent_config = await get_conv_pathway_graph(agent_id=agent_id)
        conv_pathway = ConversationalPathway()
        conv_pathway.build_from_json(conv_pathway_data)

        # Initialize variables
        initial_variables = call_data.variables.copy()
        default_variables = agent_config.get("default_variables", {})

        for unallowed_var_name in get_unallowed_variable_names():
            if unallowed_var_name in default_variables:
                raise Exception(
                    f"Default variable name '{unallowed_var_name}' is reserved and cannot be overridden."
                )

        for key, value in default_variables.items():
            initial_variables.setdefault(key, value)

        # Initialize LLM client and agent
        model_name = agent_config.get("model_name", AtomsLLMModels.ELECTRON.value)
        agent_gender = agent_config.get("synthesizer_args", {}).get("gender", "female")
        language_switching = agent_config.get("language_switching", False)
        agent_language = agent_config.get("default_language", "en")
        global_prompt = agent_config.get("global_prompt")
        global_kb_id = agent_config.get("global_knowledge_base_id")
        agent_persona = {"gender": agent_gender} if agent_gender else None

        assert model_name in [model.value for model in AtomsLLMModels], (
            f"Unknown model name '{model_name}'"
        )

        flow_model_client = OpenAIClient(
            model_id="atoms-flow-navigation",
            api_key=os.getenv("ATOMS_INFER_API_KEY"),
            base_url=f"{os.getenv('FLOW_MODEL_ENDPOINT')}/v1",
            default_response_kwargs={"temperature": 0.0},
        )

        if model_name == AtomsLLMModels.ELECTRON.value:
            response_model_client = OpenAIClient(
                model_id="atoms-responses",
                api_key=os.getenv("ATOMS_INFER_API_KEY"),
                base_url=f"{os.getenv('RESPONSE_MODEL_ENDPOINT')}/v1",
                default_response_kwargs={"temperature": 0.7},
            )
            system_prompt = FT_RESPONSE_MODEL_SYSTEM_PROMPT
        elif model_name == AtomsLLMModels.GPT_4O.value:
            response_model_client = AzureOpenAIClient(
                model_id="gpt-4o",
                api_version="2024-12-01-preview",
                azure_endpoint=os.getenv("AZURE_ENDPOINT"),
                api_key=os.getenv("AZURE_API_KEY"),
            )
            system_prompt = GENERAL_RESPONSE_MODEL_SYSTEM_PROMPT

        variable_extraction_client = AzureOpenAIClient(
            model_id="gpt-4o",
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
        )

        vector_datastore = DocumentDBVectorStore()

        if global_prompt and isinstance(global_prompt, str) and global_prompt.strip():
            system_prompt += f"\n\nSpecial Instructions:\n\n{global_prompt}"

        agent_config = {
            "model_name": model_name,
            "agent_gender": agent_gender,
            "language_switching": language_switching,
            "agent_language": agent_language,
            "global_prompt": global_prompt,
            "global_kb_id": global_kb_id,
            "agent_persona": agent_persona,
            "system_prompt": system_prompt,
        }

        flow_manager = FlowGraphManager(
            response_model_client=response_model_client,
            flow_model_client=flow_model_client,
            variable_extraction_client=variable_extraction_client,
            conversation_pathway=conv_pathway,
            transport_input_filter=transport_input_filter,
            vector_datastore=vector_datastore,
            agent_input_params=FlowGraphManager.AgentInputParams(
                initial_variables=initial_variables,
                agent_persona=agent_persona,
                current_language=agent_language,
                is_language_switching_enabled=language_switching,
                global_kb_id=global_kb_id,
            ),
        )
        return flow_manager, agent_config

    except Exception as e:
        raise Exception(f"Failed to initialize conversational agent: {str(e)}")


async def get_conv_pathway_graph(agent_id) -> tuple[str, dict]:
    """Fetch conversation pathway graph along with config from Admin API.

    Args:
        agent_id: ID of the agent
        call_id: Call ID for logging

    Returns:
        tuple[str, dict]: Processed workflow graph data and agent configuration

    Raises:
        Exception: If the graph cannot be fetched or is invalid
    """
    # Determine which identifier to use
    headers = {"X-API-Key": os.getenv("ADMIN_API_KEY"), "Content-Type": "application/json"}
    params = {"agentId": agent_id}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{os.getenv('ATOMS_BASE_URL')}/api/v1/admin/get-agent-details",
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            data: dict = response.json()

            agent_config: dict = data.get("agent", {})

            agent_config = {
                "language_switching": agent_config.get("languageSwitching", False),
                "default_language": agent_config.get("defaultLanguage", "en"),
                "synthesizer_type": agent_config.get("synthesizerType", "waves"),
                "synthesizer_args": agent_config.get("synthesizerArgs", {}),
                "synthesizer_speed": agent_config.get("synthesizerSpeed", 1.2),
                "model_name": agent_config.get("modelName", AtomsLLMModels.ELECTRON.value),
                "default_variables": agent_config.get("defaultVariables", {}),
                "allowed_idle_time_seconds": agent_config.get("allowedIdleTimeSeconds", 8),
                "num_check_human_present_times": agent_config.get("numCheckHumanPresentTimes", 2),
                "global_prompt": agent_config.get("globalPrompt"),
                "global_knowledge_base_id": agent_config.get("globalKnowledgeBaseId"),
                "synthesizer_consistency": agent_config.get("synthesizerConsistency", None),
                "synthesizer_similarity": agent_config.get("synthesizerSimilarity", None),
                "synthesizer_enhancement": agent_config.get("synthesizerEnhancement", None),
                "synthesizer_samplerate": agent_config.get("synthesizerSampleRate", None),
            }
            workflow_graph = data.get("workflowGraph") or data.get("workflow", {}).get(
                "workflowGraph"
            )

            if not workflow_graph:
                logger.error(
                    f"No workflow graph found for agent ID {agent_id}",
                )
                raise Exception("Workflow graph not found")

            processed_workflow = process_pathway_data(convert_old_to_new_format(workflow_graph))
            logger.info(
                f"Successfully fetched and processed graph for agent ID {agent_id}",
            )
            return processed_workflow, agent_config

    except httpx.HTTPError as e:
        logger.error(
            f"HTTP error for agent ID {agent_id}: {str(e)}",
            exc_info=True,
        )
        raise Exception("Failed to fetch workflow graph")
    except Exception as e:
        logger.error(
            f"Error processing graph for agent ID {agent_id}: {str(e)}",
            exc_info=True,
        )
        raise Exception("Failed to process workflow graph")


def process_pathway_data(pathway_data: list):
    for node in pathway_data:
        if node["type"] == "webhook":
            if node["api_body"] and isinstance(node["api_body"], str):
                node["api_body"] = json.loads(node["api_body"])
    return pathway_data

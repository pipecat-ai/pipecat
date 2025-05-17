import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import httpx
from pydantic import BaseModel, Field, field_validator

from pipecat.frames.frames import (
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    UserImageRawFrame,
    VisionImageRawFrame,
)

from .agent import AtomsConversationalAgent
from .base_agent import BaseConversationalAgent
from .llm_client import AzureOpenAIClient, OpenAIClient
from .pathways import ConversationalPathway
from .prompts import GENERAL_RESPONSE_MODEL_SYSTEM_PROMPT
from .utils import convert_old_to_new_format, get_abbreviations, get_unallowed_variable_names


class AtomsLLMModels(Enum):
    ELECTRON = "electron"
    GPT_4O = "gpt-4o"


class CallData(BaseModel):
    variables: Optional[Dict[str, Any]] = Field(default=None)

    @field_validator("variables")
    @classmethod
    def validate_required_variables(cls, v):
        if v is not None:
            required_keys = get_unallowed_variable_names()
            missing_keys = [key for key in required_keys if key not in v]
            if missing_keys:
                raise ValueError(f"Missing required keys in variables: {', '.join(missing_keys)}")
        return v


class CallType(Enum):
    TELEPHONY_INBOUND = "telephony_inbound"
    TELEPHONY_OUTBOUND = "telephony_outbound"
    WEBCALL = "webcall"
    CHAT = "chat"


class AgentConfig(BaseModel):  # type: ignore
    initial_message: str
    generate_responses: bool = True
    allowed_idle_time_seconds: Optional[float] = None
    num_check_human_present_times: int = 0
    allow_agent_to_be_cut_off: bool = True


class AtomsAgentConfig(AgentConfig):
    """Configuration for the Atoms conversational agent."""

    agent: BaseConversationalAgent
    llm_response_kwargs: Dict[str, Any] = {}
    sample_rate: int = 8000
    bytes_per_sample: int = 2
    agent_id: str
    conversation_id: str
    call_id: str
    user_number: Optional[str] = None
    call_type: CallType

    class Config:
        arbitrary_types_allowed = True


logger = logging.getLogger(__name__)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AtomsAgent(FrameProcessor):
    """Implementation of the Atoms conversational agent."""

    def __init__(self, agent_config: AtomsAgentConfig, **kwargs):
        super().__init__(**kwargs)
        self.atoms_agent = agent_config.agent
        self.llm_response_kwargs = agent_config.llm_response_kwargs
        self.initialized = False
        self.conversation_id = agent_config.conversation_id
        self.call_id = agent_config.call_id
        self.agent_id = agent_config.agent_id
        self.call_type = agent_config.call_type.value
        self.end_call = False

    def _update_settings(self, settings: LLMUpdateSettingsFrame):
        pass

    async def initialize(self):
        """Initialize the agent and start the logging worker"""
        self.initialized = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, VisionImageRawFrame):
            raise Exception("VisionImageRawFrame is not supported")

        await super().process_frame(frame, direction)
        # context = None
        # if isinstance(frame, LLMMessagesFrame):
        #     context = OpenAILLMContext.from_messages(frame.messages)
        # elif isinstance(frame, LLMUpdateSettingsFrame):
        #     await self._update_settings(frame.settings)
        # else:
        #     await self.push_frame(frame, direction)

        # if context:
        #     try:
        #         await self.push_frame(LLMFullResponseStartFrame())
        #         await self.start_processing_metrics()
        #         await self._process_context(context)
        #     except httpx.TimeoutException:
        #         await self._call_event_handler("on_completion_timeout")
        #     finally:
        #         await self.stop_processing_metrics()
        #         await self.push_frame(LLMFullResponseEndFrame())

        logger.debug(f"Processing frame in atoms agent: {frame}")

        if isinstance(frame, LLMMessagesFrame):
            human_input = self._get_user_input_from_frame_message(frame)
            await self.generate_response(human_input)
        else:
            await self.push_frame(frame, direction)

    def _get_user_input_from_frame_message(self, frame: LLMMessagesFrame):
        messages = frame.messages
        for message in messages:
            if "role" in message and message["role"] == "user" and "content" in message:
                return message["content"]
        return None

    async def generate_response(
        self,
        human_input,
    ):
        logger.debug(f"Generating response in atoms agent: {human_input}")
        try:
            if not self.initialized:
                await self.initialize()

            stream = await asyncio.to_thread(
                self.atoms_agent.get_response,
                prompt=human_input,
                stream=True,
                **self.llm_response_kwargs,
            )

            buffer = ""
            full_response = ""
            abbreviations = get_abbreviations()

            self.push_frame(LLMFullResponseStartFrame())

            # Process all chunks from the stream
            for chunk in stream:
                if chunk.get("end_call", False):
                    self._set_end_call()
                    cleaned_buffer = buffer.strip()
                    if cleaned_buffer:
                        full_response += " " + cleaned_buffer
                        self.push_frame(LLMTextFrame(text=cleaned_buffer))
                    break

                text = chunk.get("content", "")
                if text:
                    buffer += text

                    # Check for sentence-ending punctuation not in abbreviations
                    punctuation_indices = []
                    for i, char in enumerate(buffer):
                        if char in ".!?।":
                            # Check if it's not part of an abbreviation
                            is_abbreviation = False
                            for abbr in abbreviations:
                                if buffer.endswith(abbr, 0, i + 1):
                                    is_abbreviation = True
                                    break

                            if not is_abbreviation:
                                punctuation_indices.append(i)

                    # Yield complete segments if we have enough content
                    if punctuation_indices and len(buffer) >= 10:
                        last_index = punctuation_indices[-1] + 1
                        segment = buffer[:last_index].strip()
                        buffer = buffer[last_index:]

                        full_response += " " + segment
                        self.push_frame(LLMTextFrame(text=segment))

            cleaned_buffer = buffer.strip()
            if cleaned_buffer:
                full_response += " " + cleaned_buffer

            if cleaned_buffer:
                self.push_frame(LLMTextFrame(text=cleaned_buffer))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print("[ERROR IN AGENT] GENERATE_RESPONSE", e)
            traceback.print_exc()

            if self.atoms_agent.language_now == "hi":
                response = "माफ़ करें, अभी मेरे पास वह जानकारी उपलब्ध नहीं है। आपके समय के लिए धन्यवाद।"
            else:
                response = "Sorry, I do not have that information right now. Thanks for your time."

            self._set_end_call()
            self.push_frame(LLMTextFrame(text=response))

        finally:
            self.push_frame(LLMFullResponseEndFrame())

    def _set_end_call(self):
        self.end_call = True

    def __del__(self):
        asyncio.create_task(self.cleanup())


async def initialize_conversational_agent(
    *,
    agent_id: str,
    call_id: str,
    call_data: CallData,
    initialize_first_message: bool = True,
) -> AtomsAgent:
    """Initialize a conversational agent with the specified configuration.

    Args:
        agent_id: ID of the agent to initialize
        call_id: Call ID for logging
        call_data: Contains variables and other call-related information
        initialize_first_message: Whether to initialize first message
        save_msgs_path: Path to save messages
        **kwargs: Additional arguments passed to AtomsConversationalAgent

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
        conv_pathway_data, agent_config = await get_conv_pathway_graph(
            agent_id=agent_id, call_id=call_id
        )
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
        default_language = agent_config.get("default_language", "en")
        global_prompt = agent_config.get("global_prompt")
        global_kb_id = agent_config.get("global_knowledge_base_id")

        assert model_name in [model.value for model in AtomsLLMModels], (
            f"Unknown model name '{model_name}'"
        )

        flow_model_client = OpenAIClient(
            model_id="atoms-flow-navigation",
            api_key=os.getenv("ATOMS_INFER_API_KEY"),
            base_url=f"{os.getenv('FLOW_MODEL_ENDPOINT')}/v1",
            default_response_kwargs={"temperature": 0.0},
            keep_history=False,
            call_id=call_id,
        )

        response_model_client = OpenAIClient(
            model_id="atoms-responses",
            api_key=os.getenv("ATOMS_INFER_API_KEY"),
            base_url=f"{os.getenv('RESPONSE_MODEL_ENDPOINT')}/v1",
            default_response_kwargs={"temperature": 0.7},
            call_id=call_id,
        )
        override_response_system_prompt = None

        agent = AtomsConversationalAgent(
            conv_pathway=conv_pathway,
            response_model_client=response_model_client,
            flow_model_client=flow_model_client,
            language_switching=language_switching,
            default_language=default_language,
            initialize_first_message=initialize_first_message,
            initial_variables=initial_variables,
            agent_gender=agent_gender,
            global_prompt=global_prompt,
            global_kb_id=global_kb_id,
            call_id=call_id,
            override_response_system_prompt=override_response_system_prompt,
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            azure_api_key=os.getenv("AZURE_API_KEY"),
        )

        agent_config = AtomsAgentConfig(
            agent=agent,
            agent_id=agent_id,
            call_id=call_id,
            user_phone="+918815141271",
            conversation_id="",
            call_type=CallType.TELEPHONY_INBOUND,
            sample_rate=16000,
            bytes_per_sample=2,
            generate_responses=True,
            allow_agent_to_be_cut_off=True,
            initial_message="",
        )

        return AtomsAgent(agent_config=agent_config)

    except Exception as e:
        traceback.print_exc()

        raise Exception("Failed to initialize conversational agent")


async def get_conv_pathway_graph(agent_id, call_id) -> tuple[str, dict]:
    """
    Fetch conversation pathway graph along with config from Admin API.

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
                f"http://localhost:4001/api/v1/admin/get-agent-details",
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
                extra={"call_id": call_id},
            )
            return processed_workflow, agent_config

    except httpx.HTTPError as e:
        logger.error(
            f"HTTP error for agent ID {agent_id}: {str(e)}",
            extra={"call_id": call_id},
            exc_info=True,
        )
        raise Exception("Failed to fetch workflow graph")
    except Exception as e:
        logger.error(
            f"Error processing graph for agent ID {agent_id}: {str(e)}",
            extra={"call_id": call_id},
            exc_info=True,
        )
        raise Exception("Failed to process workflow graph")


def process_pathway_data(pathway_data: list):
    for node in pathway_data:
        if node["type"] == "webhook":
            if node["api_body"] and isinstance(node["api_body"], str):
                node["api_body"] = json.loads(node["api_body"])
    return pathway_data

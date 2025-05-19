import asyncio
import copy
import json
import os
import re
import traceback
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx
from loguru import logger
from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionUserMessageParam
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
    TranscriptionFrame,
    TranscriptionMessage,
    UserImageRawFrame,
    VisionImageRawFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)

from .agent import AtomsConversationalAgent
from .base_agent import BaseConversationalAgent
from .llm_client import AzureOpenAIClient, OpenAIClient
from .pathways import ConversationalPathway, Node, NodeType, Pathway
from .prompts import (
    FT_FLOW_MODEL_SYSTEM_PROMPT,
    GENERAL_RESPONSE_MODEL_SYSTEM_PROMPT,
    VARIABLE_EXTRACTION_PROMPT,
)
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


from datetime import datetime

import pytz

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .llm_client import BaseClient


class FlowGraphManager(FrameProcessor):
    """This is a frame processor that manages the flow graph of the agent.

    It is responsible for processing the frames and updating the flow graph.
    It will include the hopping model which will hop the graph if necessary.

    Args:
        conversation_pathway: The conversation pathway to manage.
    """

    def __init__(
        self,
        response_model_client: BaseClient,
        variable_extraction_client: BaseClient,
        conversation_pathway: ConversationalPathway,
        initial_variables: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.response_model_client: BaseClient = response_model_client
        self.variable_extraction_client: BaseClient = variable_extraction_client
        self.conv_pathway: ConversationalPathway = conversation_pathway
        self.variables = self._initialize_variables(initial_variables)
        self.current_node: Node = self._find_root()
        self._process_pre_call_api_nodes()
        self.conv_pathway.start_node = self.current_node

    def _initialize_variables(
        self, initial_variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Initialize variables with datetime information and any provided initial values.

        Args:
            initial_variables: Initial variable values to include

        Returns:
            Dictionary of initialized variables
        """
        now = datetime.now(pytz.timezone("Asia/Kolkata"))
        self.curr_date = now.strftime("%d %B %Y")
        self.curr_time = now.strftime("%I:%M %p")
        self.curr_day = now.strftime("%A")
        variables = {
            "current_date": self.curr_date,
            "current_day": self.curr_day,
            "current_time": self.curr_time,
        }
        if initial_variables:
            variables.update(initial_variables)
        return variables

    def _find_root(self) -> Node:
        """Find the true start node of the conversation pathway (node with no incoming edges).

        Returns:
            The start node
        Raises:
            AgentError: If no unique start node is found
        """
        # First, collect all nodes that are targets of pathways
        target_nodes = set()
        for node in self.conv_pathway.nodes.values():
            for pathway in node.pathways.values():
                target_nodes.add(pathway.target_node_id)

        # Then find nodes that don't appear as targets (no incoming edges)
        nodes_without_incoming = [
            node_id for node_id in self.conv_pathway.nodes.keys() if node_id not in target_nodes
        ]

        if not nodes_without_incoming:
            raise Exception(
                "No start node found in the conversation pathway. At least one node must have no incoming edges."
            )

        if len(nodes_without_incoming) > 1:
            raise Exception(
                f"Multiple potential start nodes found: {nodes_without_incoming}. Only one node should have no incoming edges."
            )

        return self.conv_pathway.nodes[nodes_without_incoming[0]]

    def _process_pre_call_api_nodes(self) -> None:
        """Process pre-call API nodes sequentially until hitting a non-pre-call node."""
        while self.current_node.type == NodeType.PRE_CALL_API:
            http_request = self.current_node.http_request
            if http_request:
                try:
                    response_data = self._make_api_request_from_node(self.current_node)
                except Exception as e:
                    logger.error(
                        f"Error in API request for node {self.current_node.id}: {str(e)}",
                    )

                # Process response data mappings
                if self.current_node.response_data and self.current_node.response_data.is_enabled:
                    self._extract_response_data(response_data, self.current_node.response_data)

            # Navigate to the next node - Pre-call nodes should only have one pathway
            if len(self.current_node.pathways) != 1:
                raise Exception(
                    f"PRE_CALL_API node {self.current_node.id} must have exactly one pathway"
                )

            # Get the first (and only) pathway
            next_node = next(iter(self.current_node.pathways.values())).target_node
            self.current_node = next_node

        # We've now reached a non-pre-call node. Verify it has is_start_node=True
        if not self.current_node.is_start_node:
            raise Exception(
                f"Non-PRE_CALL_API node {self.current_node.id} at end of pre-call sequence "
                f"must have is_start_node=True"
            )

    def _get_conditional_edges(self) -> List[Tuple[str, Pathway]]:
        """Get all condition edges from the current node."""
        conditional_edges = []
        for pathway_id, pathway in self.current_node.pathways.items():
            if pathway.is_conditional_edge and pathway.condition:
                conditional_edges.append((pathway_id, pathway))
        return conditional_edges

    def _hop_conditional_edges(
        self, conditional_edges: List[Tuple[str, Pathway]], context: OpenAILLMContext
    ) -> bool:
        for pathway_id, pathway in conditional_edges:
            if self._evaluate_conditional_edge(pathway.condition):
                self._hop(pathway_id, context=context)
                return True
        return False

    def _handle_hopping(self, context: OpenAILLMContext) -> bool:
        """Determine if a node transition is needed and execute if necessary.

        First checks conditional edges, then falls back to LLM-based decision.

        Args:
            context: OpenAILLMContext
        Returns:
            True if hopping occurred, False otherwise
        """
        # First, check if we have any conditional edges to evaluate
        conditional_edges = self._get_conditional_edges()
        # If we have conditional edges, evaluate them first
        if conditional_edges:
            is_conditional_edge_matched = self._hop_conditional_edges(
                conditional_edges, context=context
            )
            if is_conditional_edge_matched:
                return True

        # If no conditional edge matched (or none existed), continue with normal flow
        flow_history = self._build_flow_navigation_history(context=context)

        if len(flow_history) < 3:
            return False

        selected_pathway_id = self._cleanup_think_tokens(
            self.response_model_client.get_response(flow_history)
        )

        if selected_pathway_id == "null":
            return False

        try:
            self._hop(pathway_id=selected_pathway_id, context=context)
            return True
        except Exception:
            return False

    def _hop(self, pathway_id: str, context: OpenAILLMContext) -> None:
        """Transition to a new node via the specified pathway.

        Args:
            pathway_id: ID of the pathway to follow
            context: OpenAILLMContext

        Raises:
            Exception: If pathway_id not found in current node
        """
        if not pathway_id in self.current_node.pathways:
            raise Exception(
                f"Pathway '{pathway_id}' not found in current node '{self.current_node.name}'"
            )

        self._extract_variables(context=context)
        self.current_node = self.current_node.pathways[pathway_id].target_node

    def _extract_variables(self, context: OpenAILLMContext) -> Dict[str, Any]:
        """Extract variables from conversation context using LLM.

        Returns:
            Dictionary of extracted variables

        Raises:
            Exception: If extraction fails or required variables are missing
        """
        if not self.current_node.variables or not self.current_node.variables.is_enabled:
            return {}

        variable_schema = self.current_node.variables.data

        if not variable_schema:
            return {}

        # Format variable schema for prompt
        formatted_variable_schema = json.dumps(
            [var.model_dump() for var in variable_schema], indent=2, ensure_ascii=False
        )

        context = []
        for context_message in context.messages[2:]:
            if isinstance(context_message, ChatCompletionUserMessageParam):
                context.append(f"User: {context_message.content}")
            elif isinstance(context_message, ChatCompletionAssistantMessageParam):
                context.append(f"Assistant: {context_message.content}")

        # Create extraction prompt
        extraction_prompt = VARIABLE_EXTRACTION_PROMPT.format(
            current_date=self.curr_date,
            current_time=self.curr_time,
            current_day=self.curr_day,
            context="\n".join(context),
            variable_schema=formatted_variable_schema,
        )

        # Get extraction response from GPT-4o
        raw_extraction = self.variable_extraction_client.get_response(
            [{"role": "user", "content": extraction_prompt}],
        )

        try:
            # Clean and parse LLM response
            cleaned_json = self._clean_json(raw_extraction)
            extracted_variables = json.loads(cleaned_json)

            # Validate extracted variables
            if not isinstance(extracted_variables, dict):
                raise Exception("LLM response must be a JSON object")

            # Update and return
            self.variables.update(extracted_variables)
            return extracted_variables

        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse variable extraction response: {str(e)}")

    def _clean_json(self, text: str) -> str:
        """Clean JSON string from language model output."""
        return text.strip("```").strip("json")

    def _cleanup_think_tokens(self, text: str) -> str:
        return text.replace("<think>", "").replace("</think>", "").strip()

    def _evaluate_conditional_edge(self, condition: str) -> bool:
        """Evaluate conditional edge expressions against variables using Python's eval.

        Args:
            condition: Condition to evaluate, e.g. "{{variable_name}} == 'value'"

        Returns:
            Whether the condition is satisfied
        """
        try:
            # Extract the variable name
            match = re.search(r"{{(.+?)}}", condition)
            if not match:
                return False

            variable_name = match.group(1).strip()

            # Check if the variable exists
            if variable_name not in self.variables:
                return False

            # Get the actual value and format it for eval
            value = self.variables[variable_name]
            if isinstance(value, str):
                formatted_value = f"'{value}'"
            elif isinstance(value, bool):
                formatted_value = str(value)
            elif value is None:
                formatted_value = "None"
            else:
                formatted_value = str(value)

            # Replace the variable reference with its value
            python_condition = condition.replace(match.group(0), formatted_value)

            # Process special operators first
            if " contains " in python_condition:
                parts = python_condition.split(" contains ")
                if len(parts) == 2:
                    left, right = parts
                    python_condition = f"{right} in {left}"
            elif " not_contains " in python_condition:
                parts = python_condition.split(" not_contains ")
                if len(parts) == 2:
                    left, right = parts
                    python_condition = f"{right} not in {left}"
            elif " is null" in python_condition:
                python_condition = python_condition.replace(" is null", " == None")
            elif " is not null" in python_condition:
                python_condition = python_condition.replace(" is not null", " != None")
            elif " is true" in python_condition or " is True" in python_condition:
                python_condition = re.sub(r" is [tT]rue", " == True", python_condition)
            elif " is false" in python_condition or " is False" in python_condition:
                python_condition = re.sub(r" is [fF]alse", " == False", python_condition)
            else:
                # For equality operators, check if the right side needs quoting
                operators = ["==", "!=", " is not ", " is "]
                for op in operators:
                    if op in python_condition:
                        left, right = python_condition.split(op, 1)
                        right = right.strip()

                        # Handle boolean literals for operators
                        if right.lower() == "true":
                            right = "True"
                        elif right.lower() == "false":
                            right = "False"

                        # For 'is' and 'is not' operators, convert to '==' and '!=' when comparing with literals
                        if op == " is ":
                            op = " == "
                        elif op == " is not ":
                            op = " != "

                        if not (
                            right.startswith('"')
                            or right.startswith("'")
                            or right.replace(".", "", 1).isdigit()
                            or right in ["True", "False", "None"]
                        ):
                            right = f"'{right}'"

                        python_condition = f"{left.strip()} {op.strip()} {right}"
                        break

            # Evaluate using Python's eval
            result = eval(python_condition)
            return bool(result)

        except Exception as e:
            return False

    def _build_flow_navigation_history(self, context: OpenAILLMContext) -> List[Dict[str, Any]]:
        """Build message history for flow navigation decisions.

        Args:
            context: OpenAILLMContext

        Returns:
            List of messages formatted for the flow model
        """
        flow_navigation_history: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": FT_FLOW_MODEL_SYSTEM_PROMPT,
            }
        ]

        # Find the most recent node message
        current_node_index = None
        for idx, msg in enumerate(reversed(context.messages[1:])):
            if isinstance(msg, ChatCompletionUserMessageParam):
                try:
                    content = json.loads(msg.content)
                    if "id" in content and content["id"] == self.current_node.id:
                        current_node_index = len(context.messages) - 1 - idx
                        break
                except (json.JSONDecodeError, TypeError):
                    pass
        else:
            return []

        # Build current node representation with specific order
        node_data = {
            "name": self.current_node.name,
            "type": self.current_node.type.name,
            "action": self.current_node.action,
        }

        # Add loop condition if it exists
        if self.current_node.loop_condition and self.current_node.loop_condition.strip():
            node_data["loop_condition"] = self.current_node.loop_condition

        # Add pathways with non-empty descriptions
        node_data["pathways"] = []
        for pathway_id, pathway in self.current_node.pathways.items():
            if not pathway.is_conditional_edge:
                pathway_data = {
                    "id": pathway_id,
                    "condition": self._replace_variables(pathway.condition, self.variables),
                }
                if pathway.description and pathway.description.strip():
                    pathway_data["description"] = self._replace_variables(
                        pathway.description, self.variables
                    )
                node_data["pathways"].append(pathway_data)

        # Add current node to flow history
        flow_navigation_history.append(
            {"role": "user", "content": json.dumps(node_data, indent=2, ensure_ascii=False)}
        )

        # we have to build flow navigation history by appending only assistant and user messages
        while current_node_index < len(context.messages):
            # check if the current context is a assistant message
            if isinstance(
                context.messages[current_node_index], ChatCompletionAssistantMessageParam
            ):
                assistant_content: ChatCompletionAssistantMessageParam = context.messages[
                    current_node_index
                ]["content"]

                # now find the next user message
                while current_node_index < len(context.messages) and not isinstance(
                    context.messages[current_node_index], ChatCompletionUserMessageParam
                ):
                    current_node_index += 1

                if current_node_index < len(context.messages):
                    user_content: ChatCompletionUserMessageParam = context.messages[
                        current_node_index
                    ]["content"]

                    flow_navigation_history += [
                        {
                            "role": "assistant",
                            "content": "null",
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"assistant": assistant_content, "user": user_content},
                                indent=2,
                                ensure_ascii=False,
                            ),
                        },
                    ]

            current_node_index += 1

        return flow_navigation_history

    def _replace_variables(self, input_string: Any, variables: Dict[str, Any]) -> Any:
        """Replace variable placeholders in a string with their values.

        Args:
            input_string: String potentially containing variable placeholders
            variables: Dictionary of variable names and values

        Returns:
            String with variables replaced
        """
        pattern = r"\{\{(\w+)\}\}"

        def replace_func(match):
            variable_name = match.group(1) if match.group(1) else match.group(2)

            if variable_name in variables:
                return str(variables[variable_name])
            else:
                logger.error(
                    f"Variable replacement failed for '{variable_name}' in node {self.current_node.id}"
                )
                return match.group(0)

        if isinstance(input_string, str):
            return re.sub(pattern, replace_func, input_string)
        else:
            return input_string

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context: OpenAILLMContext = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = OpenAILLMContext.from_messages(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            context = OpenAILLMContext()
            context.add_image_frame_message(
                format=frame.format, size=frame.size, image=frame.image, text=frame.text
            )
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        else:
            await self.push_frame(frame, direction)

        if context:
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(context)
            except httpx.TimeoutException:
                await self._call_event_handler("on_completion_timeout")
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())


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

        variable_extraction_client = AzureOpenAIClient(
            model_id="gpt-4o",
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
        )

        return FlowGraphManager(
            response_model_client=response_model_client,
            variable_extraction_client=variable_extraction_client,
            conversation_pathway=conv_pathway,
            initial_variables=initial_variables,
        )

    except Exception as e:
        traceback.print_exc()

        raise Exception("Failed to initialize conversational agent")


async def get_conv_pathway_graph(agent_id, call_id) -> tuple[str, dict]:
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

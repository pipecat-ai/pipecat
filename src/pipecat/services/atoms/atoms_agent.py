import asyncio
import copy
import json
import os
import re
import traceback
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx
from loguru import logger
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, Field, field_validator

from pipecat.frames.frames import (
    Frame,
    LastTurnFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)

from .llm_client import AzureOpenAIClient, OpenAIClient
from .pathways import ConversationalPathway, Node, NodeType, Pathway
from .prompts import (
    FT_FLOW_MODEL_SYSTEM_PROMPT,
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


from datetime import datetime
from typing import AsyncGenerator, Text

import pytz

from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .llm_client import BaseClient


def replace_variables(input_string: Any, variables: Dict[str, Any]) -> Any:
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
                f"Variable replacement failed for '{variable_name} in text `{input_string}`'"
            )
            return match.group(0)

    if isinstance(input_string, str):
        return re.sub(pattern, replace_func, input_string)
    else:
        return input_string


class AtomsAgentContext(OpenAILLMContext):
    """This is the adapater for upgrading the openai context to the atoms agent context."""

    def __init__(
        self,
        messages: Optional[List[dict]] = None,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[dict] = None,
        *,
        system: Optional[str] = None,
    ):
        super().__init__(messages=messages, tools=tools, tool_choice=tool_choice)
        self.system = system

    @staticmethod
    def upgrade_to_atoms_agent(obj: OpenAILLMContext) -> "AtomsAgentContext":
        logger.debug(f"Upgrading to Atoms Agent: {obj}")
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, AtomsAgentContext):
            obj.__class__ = AtomsAgentContext
            obj._restructure_from_openai_messages()
        else:
            obj._restructure_from_openai_messages()
        return obj

    @classmethod
    def from_openai_context(cls, openai_context: OpenAILLMContext):
        self = cls(
            messages=openai_context.messages,
            tools=openai_context.tools,
            tool_choice=openai_context.tool_choice,
        )
        self.set_llm_adapter(openai_context.get_llm_adapter())
        self._restructure_from_openai_messages()
        return self

    def get_current_user_transcript(self):
        """Get the transcript from the messages.

        We expect that the last message in the context should be user message it will return empty string.
        """
        if len(self.messages) == 0:
            return ""

        last_message = self.messages[-1]
        if last_message["role"] == "user":
            try:
                content = json.loads(last_message["content"])
                transcript = content["transcript"]
                return transcript
            except Exception as e:
                logger.error(f"Error in getting last user transcript from context messages: {e}")
                return ""
        logger.error(
            f"Last message in the context is not a user message, last_message: {last_message}"
        )
        return ""

    def add_message(self, message: ChatCompletionMessageParam):
        if message["role"] == "user":
            # check if the previous role is "user" as well then we have to add assistant message with content "None"
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.append(
                    ChatCompletionAssistantMessageParam(role="assistant", content="")
                )
        if message["role"] == "assistant":
            # check if the previous role is "assistant" as well then we have to add user message with content "None"
            if self.messages and self.messages[-1]["role"] == "assistant":
                self.messages.append(ChatCompletionUserMessageParam(role="user", content=""))
        super().add_message(message=message)

    # convert a message in atoms agent format into one or more messages in OpenAI format
    def to_standard_message(self, obj):
        """Convert atoms agent message format to standard structured format.

        Handles text content and function calls for both user and assistant messages.

        Args:
            obj: Message in atoms agent format:
                {
                    "role": "user/assistant",
                    "content": [{"text": str} | {"toolUse": {...}} | {"toolResult": {...}}]
                }

        Returns:
            List of messages in standard format:
            [
                {
                    "role": "user/assistant/tool",
                    "content": [{"type": "text", "text": str}]
                }
            ]
        """
        if "role" in obj and obj["role"] == "user":
            try:
                json_content = json.loads(obj["content"])
                transcript = json_content["transcript"]
                return {"role": "user", "content": transcript}
            except json.JSONDecodeError:
                return obj
            except Exception as e:
                logger.error(f"Error parsing user message: {obj}")
                return obj
        else:
            return obj

    def get_user_context_delta(self) -> Optional[Dict[str, Any]]:
        """This function will return the delta from the previous user message and the current user message.

        To work it as exptected we have taken this into consideration that last message (current) we will get it at index [-1] and second last message (previous) we will get it at index [-3].
        why not [-2] because the last message is the user message and the second last message is the assistant message.
        """
        if len(self.messages) < 3:
            return None

        try:
            last_user_message = self.messages[-1]
            second_last_user_message = self.messages[-3]
            if last_user_message["role"] != "user" or second_last_user_message["role"] != "user":
                return None

            last_user_message_content = json.loads(last_user_message["content"])
            second_last_user_message_content = json.loads(second_last_user_message["content"])

            last_user_message_response_model_context_json = last_user_message_content[
                "response_model_context"
            ]
            second_last_user_message_response_model_context_json = second_last_user_message_content[
                "response_model_context"
            ]

            if (
                last_user_message_response_model_context_json is None
                or second_last_user_message_response_model_context_json is None
            ):
                return None

            last_user_message_response_model_context: Dict[str, Any] = json.loads(
                last_user_message_response_model_context_json
            )
            second_last_user_message_response_model_context: Dict[str, Any] = json.loads(
                second_last_user_message_response_model_context_json
            )

            if (
                last_user_message_response_model_context is None
                or not isinstance(last_user_message_response_model_context, dict)
                or second_last_user_message_response_model_context is None
                or not isinstance(second_last_user_message_response_model_context, dict)
            ):
                return None

            if (
                last_user_message_response_model_context["id"]
                != second_last_user_message_response_model_context["id"]
            ):
                return None

            transcript = last_user_message_content["transcript"]

            delta = {"user_response": transcript}

            # Compare other fields and include only changed ones
            for key, value in last_user_message_response_model_context.items():
                if key != "user_response" and (
                    key not in second_last_user_message_response_model_context
                    or second_last_user_message_response_model_context[key] != value
                ):
                    delta[key] = value

            return delta
        except Exception as e:
            logger.error(f"Error getting user content delta: {e}")
            return None

    def _update_last_user_context(self, key: str, value: Any):
        if self.messages and self.messages[-1]["role"] == "user":
            try:
                json_content = json.loads(self.messages[-1]["content"])
                json_content[key] = value
                self.messages[-1]["content"] = json.dumps(json_content)
            except Exception as e:
                logger.error(
                    f"Error updating last user context, description: updating the last user message the user content should be in the json format"
                )

    def _validate_messages(self, message):
        """validate messages to ensure the roles alternate and the content is in the correct format."""
        pass

    def _restructure_from_atoms_agent_messages(self):
        """restructure the open ai context from the atoms agent context."""
        messages = []

        for message in self.messages:
            if message["role"] == "user":
                try:
                    json_content = json.loads(message["content"])
                    transcript = json_content["transcript"]
                    messages.append(ChatCompletionUserMessageParam(role="user", content=transcript))
                except Exception:
                    logger.debug(f"Error parsing user message: {message}")
                    messages.append(
                        ChatCompletionUserMessageParam(role="user", content=message["content"])
                    )
            else:
                messages.append(message)

        self.messages.clear()
        self.messages.extend(messages)

    def _restructure_from_openai_messages(self):
        """This function will restructure the openai user context messages which are in the default string format and convert them to the atoms agent context messages."""
        messages = []

        for message in self.messages:
            if message["role"] == "user":
                try:
                    json.loads(message["content"])
                    # if the content is json than it is already in the atoms agent format no need to convert it
                    messages.append(message)
                except json.JSONDecodeError:
                    # if json conversion fails than it is a user message and we need to convert it to the atoms agent format
                    messages.append(
                        ChatCompletionUserMessageParam(
                            role="user", content=json.dumps({"transcript": message["content"]})
                        )
                    )
                except Exception as e:
                    logger.error(f"Error parsing user message: {message}")
                    messages.append(message)
            else:
                messages.append(message)

        self.messages.clear()
        self.messages.extend(messages)

    def get_response_model_context(self):
        """Get the response model context from the messages."""
        try:
            messages = []
            for message in self.messages:
                if message["role"] == "user":
                    try:
                        content = json.loads(message["content"])
                        response_model_context = content["response_model_context"]
                        delta = content.get("delta", None)

                        if delta:
                            messages.append(
                                ChatCompletionUserMessageParam(role="user", content=delta)
                            )
                        else:
                            messages.append(
                                ChatCompletionUserMessageParam(
                                    role="user", content=response_model_context
                                )
                            )
                    except Exception as e:
                        logger.error(
                            f"Error in generating user message for response model context {message}"
                        )
                elif message["role"] == "assistant":
                    messages.append(message)

            return messages
        except Exception as e:
            logger.error(f"Error in getting response model context: {e}")
            return []

    def get_flow_navigation_model_context(
        self, current_node: Node, variables: Optional[Dict[str, Any]]
    ):
        """Format the messages for the flow navigation."""
        flow_navigation_history: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": FT_FLOW_MODEL_SYSTEM_PROMPT,
            }
        ]

        # Find the most recent node message
        current_node_index = None
        for idx, msg in enumerate(reversed(self.messages[1:])):
            if msg["role"] == "user":
                try:
                    content = json.loads(msg["content"])
                    if "response_model_context" in content:
                        repsonse_model_context = json.loads(content["response_model_context"])
                        if (
                            "id" in repsonse_model_context
                            and repsonse_model_context["id"] == current_node.id
                        ):
                            current_node_index = len(self.messages) - 1 - idx
                            break
                except (json.JSONDecodeError, TypeError):
                    pass
        else:
            return []

        # Build current node representation with specific order
        node_data = {
            "name": current_node.name,
            "type": current_node.type.name,
            "action": current_node.action,
        }

        # Add loop condition if it exists
        if current_node.loop_condition and current_node.loop_condition.strip():
            node_data["loop_condition"] = current_node.loop_condition

        # Add pathways with non-empty descriptions
        node_data["pathways"] = []
        for pathway_id, pathway in current_node.pathways.items():
            if not pathway.is_conditional_edge:
                pathway_data = {
                    "id": pathway_id,
                    "condition": replace_variables(pathway.condition, variables),
                }
                if pathway.description and pathway.description.strip():
                    pathway_data["description"] = replace_variables(pathway.description, variables)
                node_data["pathways"].append(pathway_data)

        # Add current node to flow history
        flow_navigation_history.append(
            {"role": "user", "content": json.dumps(node_data, indent=2, ensure_ascii=False)}
        )

        # we have to build flow navigation history by appending only assistant and user messages
        while current_node_index < len(self.messages):
            # check if the current context is a assistant message
            if self.messages[current_node_index]["role"] == "assistant":
                assistant_content: ChatCompletionAssistantMessageParam = self.messages[
                    current_node_index
                ]["content"]

                # now find the next user message
                while (
                    current_node_index < len(self.messages)
                    and self.messages[current_node_index]["role"] != "user"
                ):
                    current_node_index += 1

                if current_node_index < len(self.messages):
                    user_content: ChatCompletionUserMessageParam = self.messages[
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

    def _get_variable_extraction_prompt(self):
        """Get the prompt for the variable extraction."""
        messages = []
        for message in self.messages:
            if message["role"] == "user":
                messages.append(f"User: {message['content']}")
            else:
                messages.append(f"Assistant: {message['content']}")
        return "\n".join(messages)


class BackgroundTaskManager:
    def __init__(self):
        self.tasks: List[asyncio.Task] = []

    def run_task(self, task: Callable, *args, **kwargs):
        task: asyncio.Task = asyncio.create_task(task(*args, **kwargs))
        self.tasks.append(task)
        task.add_done_callback(lambda _: self.tasks.remove(task))
        return task

    async def wait(self):
        await asyncio.gather(*self.tasks)


class FlowGraphManager(FrameProcessor):
    """This is a frame processor that manages the flow graph of the agent.

    It is responsible for processing the frames and updating the flow graph.
    It will include the hopping model which will hop the graph if necessary.

    Args:
        conversation_pathway: The conversation pathway to manage.
    """

    def __init__(
        self,
        flow_model_client: BaseClient,
        variable_extraction_client: BaseClient,
        response_model_client: BaseClient,
        conversation_pathway: ConversationalPathway,
        background_task_manager: Optional[BackgroundTaskManager] = BackgroundTaskManager(),
        initial_variables: Optional[Dict[str, Any]] = None,
        agent_persona: Optional[str] = None,
    ):
        super().__init__()
        self.flow_model_client: BaseClient = flow_model_client
        self.variable_extraction_client: BaseClient = variable_extraction_client
        self.response_model_client: BaseClient = response_model_client
        self.conv_pathway: ConversationalPathway = conversation_pathway
        self.variables = self._initialize_variables(initial_variables)
        self.current_node: Node = self._find_root()
        self._process_pre_call_api_nodes()
        self.conv_pathway.start_node = self.current_node
        self._end_call_tag = "<end_call>"
        self.agent_persona = agent_persona
        self._background_task_manager = background_task_manager

    def _get_initial_user_message(self) -> str:
        """Get the initial user message for the conversation."""
        return self._get_current_state_as_json()

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
                    logger.debug(f"response data from pre-call api node: {response_data}")
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

    async def _hop_conditional_edges(
        self, conditional_edges: List[Tuple[str, Pathway]], context: AtomsAgentContext
    ) -> bool:
        for pathway_id, pathway in conditional_edges:
            logger.debug(f"evaluating conditional edge: {pathway.condition}")
            if self._evaluate_conditional_edge(pathway.condition):
                await self._hop(pathway_id, context=context)
                return True
        return False

    async def _handle_hopping(self, context: AtomsAgentContext) -> bool:
        """Determine if a node transition is needed and execute if necessary.

        First checks conditional edges, then falls back to LLM-based decision.

        Args:
            context: AtomsAgentContext
        Returns:
            True if hopping occurred, False otherwise
        """
        # First, check if we have any conditional edges to evaluate
        conditional_edges = self._get_conditional_edges()
        logger.debug(f"conditional edges: {conditional_edges}")
        # If we have conditional edges, evaluate them first
        if conditional_edges:
            is_conditional_edge_matched = await self._hop_conditional_edges(
                conditional_edges, context=context
            )
            if is_conditional_edge_matched:
                return True

        # If no conditional edge matched (or none existed), continue with normal flow
        flow_history = context.get_flow_navigation_model_context(
            current_node=self.current_node, variables=self.variables
        )
        logger.debug(f"flow navigation context: {flow_history}")

        if len(flow_history) < 3:
            return False

        selected_pathway_id = self._cleanup_think_tokens(
            await self.flow_model_client.get_response(flow_history)
        )
        logger.debug(f"selected pathway id: {selected_pathway_id}")
        if selected_pathway_id == "null":
            return False

        try:
            await self._hop(pathway_id=selected_pathway_id, context=context)
            return True
        except Exception:
            return False

    async def _hop(self, pathway_id: str, context: AtomsAgentContext) -> None:
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

        await self._extract_variables(context=context)
        self.previous_node = self.current_node
        self.current_node = self.current_node.pathways[pathway_id].target_node

    async def _extract_variables(self, context: AtomsAgentContext) -> Dict[str, Any]:
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
            if context_message["role"] == "user":
                context.append(f"User: {context_message.content}")
            elif context_message["role"] == "assistant":
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
        raw_extraction = await self.variable_extraction_client.get_response(
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

    async def get_response(self, context: AtomsAgentContext) -> AsyncGenerator[str, None]:
        """Get the response from the response model client."""
        # validate the messages to ensure the roles alternate and the content is in the correct format else openai will throw an error
        await self._handle_hopping(context=context)

        # # after hopping we have to update the node data in the context
        # context._update_last_user_context("node_data", self.current_node.model_dump_json())

        # format the messages for the response model
        context._update_last_user_context(
            "response_model_context",
            self._get_current_state_as_json(
                user_response=context.get_current_user_transcript(),
            ),
        )

        # after hopping we have to update the delta in the context
        # we will only get delta after updating the current reponse model context because the delta is calculated based on the current response model context and the previous user message
        delta = context.get_user_context_delta()
        if delta:
            delta = json.dumps(delta, indent=2, ensure_ascii=False)
            context._update_last_user_context("delta", delta)

        if self.current_node.type == NodeType.END_CALL:
            logger.debug("end call node detected")
            await self.push_frame(LastTurnFrame(conversation_id="123"))

        if self.current_node.static_text:
            return self._handle_static_response(context=context)
        else:
            return self._handle_dynamic_response(context=context)

    async def _process_context(self, context: AtomsAgentContext) -> None:
        """Process the context and update the flow model client."""
        response = await self.get_response(context=context)
        async for chunk in response:
            logger.debug(f"chunk: {chunk}")
            await self.push_frame(LLMTextFrame(text=chunk))

    def _get_transcript_from_context(self, context: AtomsAgentContext) -> str:
        """Get the transcript from the context."""
        for idx in range(len(context.messages) - 1, -1, -1):
            if context.messages[idx]["role"] == "user":
                return context.messages[idx]["content"]
        return ""

    def _get_current_state_as_json(
        self,
        user_response: Optional[str] = None,
        custom_instructions: Optional[list] = None,
        add_variables: bool = False,
        add_agent_persona: bool = False,
    ) -> str:
        """Convert current state and user response to JSON string.

        Args:
            user_response: User's input text
            custom_instructions: Additional instructions for the LLM
            add_variables: Whether to include variables in context
            add_agent_persona: Whether to include agent persona in context

        Returns:
            JSON string representing the current state
        """
        current_state = {
            "id": self.current_node.id,
            "name": self.current_node.name,
            "type": self.current_node.type.name,
            "action": replace_variables(self.current_node.action, self.variables),
            "loop_condition": replace_variables(self.current_node.loop_condition, self.variables),
            "user_response": user_response,
        }

        if self.current_node.knowledge_base and self.current_node.knowledge_base.strip():
            current_state["knowledge_base"] = self.current_node.knowledge_base
        if add_variables:
            current_state["variables"] = {
                "current_date": self.curr_date,
                "current_day": self.curr_day,
                "current_time": self.curr_time,
            }

        if add_agent_persona and self.agent_persona:
            current_state["agent_persona"] = self.agent_persona
        if custom_instructions:
            current_state["custom_instructions"] = custom_instructions
        if current_state.get("loop_condition") is None or (
            isinstance(current_state["loop_condition"], str)
            and current_state["loop_condition"].strip() == ""
        ):
            current_state.pop("loop_condition")

        # Return full state if no delta or node changed
        return json.dumps(current_state, indent=2, ensure_ascii=False)

    # async def _handle_response(
    #     self,
    #     context: OpenAILLMContext,
    #     add_variables: bool = False,
    #     add_agent_persona: bool = False,
    # ) -> None:
    #     """Handle the response from the response model client."""
    #     self.current_node.action = self._replace_variables(self.current_node.action, self.variables)
    #     transcript = self._get_transcript_from_context(context)

    #     if (
    #         self.current_node.loop_condition
    #         and isinstance(self.current_node.loop_condition, str)
    #         and self.current_node.loop_condition.strip()
    #     ):
    #         self.current_node.loop_condition = self._replace_variables(
    #             self.current_node.loop_condition, self.variables
    #         )

    #     # Do language switching if turned on
    #     if self.language_switching and transcript:
    #         self.language_now = self._detect_language(transcript)

    #     custom_instructions = custom_instructions or []
    #     custom_instructions.append(self._get_language_switch_inst())

    #     # Get current state of conversation
    #     current_state = self._format_current_state_as_json(
    #         user_response=transcript,
    #         custom_instructions=custom_instructions,
    #         add_variables=add_variables,
    #         add_agent_persona=add_agent_persona,
    #     )

    #     # Use static text or not
    #     use_static_text = self.current_node.static_text

    async def _handle_static_response(
        self, context: AtomsAgentContext
    ) -> AsyncGenerator[str, None]:
        """Handle the static response from the response model client."""
        logger.debug(f"handling static response, text: {self.current_node.action}")
        # await self.push_frame(LLMTextFrame(text=self.current_node.action))
        yield self.current_node.action
        # await self.push_frame(LLMTextFrame(text=self.current_node.action))

    async def _punctuation_based_response_generator(self, stream: AsyncGenerator[str, None]):
        buffer = ""
        async for chunk in stream:
            buffer += chunk
            punctuation_indices = []
            for i, char in enumerate(buffer):
                if char in ".!?।":
                    # Check if it's not part of an abbreviation
                    is_abbreviation = False
                    for abbr in get_abbreviations():
                        if buffer.endswith(abbr, 0, i + 1):
                            is_abbreviation = True
                            break

                    # Check if followed by whitespace or end of buffer
                    if not is_abbreviation and (i + 1 >= len(buffer) or buffer[i + 1].isspace()):
                        punctuation_indices.append(i)

            # Yield complete segments if we have enough content
            if punctuation_indices and len(buffer) >= 10:
                last_index = punctuation_indices[-1] + 1
                segment = buffer[:last_index].strip()
                buffer = buffer[last_index:]
                await self.push_frame(LLMTextFrame(text=segment.strip()))

        if buffer and buffer.strip():
            await self.push_frame(LLMTextFrame(text=buffer.strip()))

    async def _handle_dynamic_response(
        self, context: AtomsAgentContext
    ) -> AsyncGenerator[str, None]:
        """Handle the dynamic response from the response model client."""
        try:
            response_model_context = context.get_response_model_context()
            logger.debug(
                f"getting response from response model client, messages: {response_model_context}"
            )
            async for chunk in await self.response_model_client.get_response(
                response_model_context, stream=True, stop=[self._end_call_tag]
            ):
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        # await self.push_frame(LLMTextFrame(text=content))
                        yield content
                    if (
                        chunk.choices[0].finish_reason
                        and hasattr(chunk.choices[0], "stop_reason")
                        and chunk.choices[0].stop_reason == self._end_call_tag
                    ):
                        logger.debug("last turn chunk detected")
                        await self.push_frame(LastTurnFrame(conversation_id="123"))
        except Exception as e:
            logger.error(f"Error handling dynamic response: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, OpenAILLMContextFrame):
            context = AtomsAgentContext.upgrade_to_atoms_agent(frame.context)
        elif isinstance(frame, LLMMessagesFrame):
            context = AtomsAgentContext.from_messages(frame.messages)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        else:
            await self.push_frame(frame, direction)

        if context:
            try:
                await self.push_frame(LLMFullResponseStartFrame())
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
) -> FlowGraphManager:
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
        )

        response_model_client = OpenAIClient(
            model_id="atoms-responses",
            api_key=os.getenv("ATOMS_INFER_API_KEY"),
            base_url=f"{os.getenv('RESPONSE_MODEL_ENDPOINT')}/v1",
            default_response_kwargs={"temperature": 0.7},
        )

        variable_extraction_client = AzureOpenAIClient(
            model_id="gpt-4o",
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
        )

        return FlowGraphManager(
            response_model_client=response_model_client,
            flow_model_client=flow_model_client,
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

"""This file contains the atoms agent.

It is a frame processor that manages the flow graph of the agent.
It will include the hopping model which will hop the graph if necessary.
It will inclide the variable extraction model which will extract the variables from the user response.
It will include the response model which will generate the response for the user.

"""

import json
import os
import re
import traceback
from datetime import datetime
from enum import Enum
from inspect import isasyncgen, iscoroutinefunction, isgenerator
from typing import Any, AsyncGenerator, Callable, Dict, Generator, List, Optional, Tuple, Union

import aiohttp
import aiohttp.client_exceptions
import httpx
import pytz
from jsonpath_ng import parse
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from pipecat.frames.frames import (
    Frame,
    LastTurnFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    SetTransferCallDataFrame,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContextFrame
from pipecat.processors.filters.custom_mute_filter import TransportInputFilter
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.documentdb.rag import DocumentDBVectorStore
from pipecat.utils.tracing.service_decorators import track_latency

from .context import AtomsAgentContext
from .llm_client import AzureOpenAIClient, BaseClient, OpenAIClient
from .pathways import ConversationalPathway, Node, NodeType, Pathway
from .prompts import (
    VARIABLE_EXTRACTION_PROMPT,
)
from .utils import (
    get_abbreviations,
    get_language_switch_inst,
    replace_variables,
    replace_variables_recursive,
)


class FlowGraphManager(FrameProcessor):
    """This is a frame processor that manages the flow graph of the agent.

    It is responsible for processing the frames and updating the flow graph.
    It will include the hopping model which will hop the graph if necessary.

    Args:
        conversation_pathway: The conversation pathway to manage.
    """

    class APICallNodeHandler:
        """This class is responsible for handling the API call node."""

        def __init__(self, flow_graph_manager: "FlowGraphManager"):
            self.flow_graph_manager: "FlowGraphManager" = flow_graph_manager

        def _extract_variables_from_context(self, node: Node, context: AtomsAgentContext) -> None:
            """Extract data from API response using JSON paths.

            Args:
                node: The node containing the HTTP request configuration
                context: The context of the agent
            """
            response_data = context.get_last_user_context()["api_node_response"]
            self._extract_variables(node, response_data)

        def _extract_variables(
            self,
            node: Node,
            response_data: str,
        ) -> None:
            """Extract data from API response using JSON paths.

            Args:
                node: The node containing the HTTP request configuration
                response_data: Response data from API call in string format
            """
            config = node.response_data
            if (
                not config.is_enabled
                or not config.data
                or not response_data
                or not isinstance(response_data, dict)
            ):
                return

            extracted = {}
            for mapping in config.data:
                try:
                    # Parse and find value using JSON path
                    jsonpath_expr = parse(mapping.json_path)
                    matches = jsonpath_expr.find(response_data)

                    if matches:
                        # Extract the first matching value
                        value = matches[0].value
                        extracted[mapping.variable_name] = value
                        logger.debug(f"extracted variable: {mapping.variable_name}, value: {value}")
                    else:
                        logger.error(f"No match found for JSON path: {mapping.json_path}")

                except Exception as e:
                    logger.error(f"Error extracting data with path '{mapping.json_path}': {str(e)}")

            # Update variables with extracted data
            if extracted:
                self.flow_graph_manager.variables.update(extracted)
                logger.info(f"Updated variables with response data: {extracted}")

        @track_latency(
            service_name="agent",
            metric_name="api_call_node_latency",
            logger=logger,
        )
        async def _make_api_request_from_node(
            self, node: Node, variables: Dict[str, Any]
        ) -> Union[Dict[str, Any], str]:
            """Make an API request based on node configuration.

            Args:
                node: The node containing the HTTP request configuration
                variables: The variables to use for the API request

            Returns:
                API response data

            Raises:
                AgentError: If the request fails or is misconfigured
            """
            # Extract request parameters
            http_request = node.http_request
            if not http_request:
                raise Exception(f"Node {node.id} has no HTTP request configuration")

            # Process headers with variable substitution
            headers = {}
            if http_request.headers and http_request.headers.is_enabled:
                for key, value in http_request.headers.data.items():
                    headers[key] = replace_variables(value, variables)

            # Process authorization if enabled
            if http_request.authorization and http_request.authorization.is_enabled:
                auth_data = http_request.authorization.data
                if auth_data and auth_data.token:
                    # TODO: we are currently using bearer token for all the api calls
                    # we need to add support for other types of authorization
                    headers["Authorization"] = f"Bearer {auth_data.token}"

            # Process body with variable substitution
            body = None
            if http_request.body and http_request.body.is_enabled and http_request.body.data:
                try:
                    # Check if body is already a dict or try to parse from string
                    if isinstance(http_request.body.data, dict):
                        body_dict = http_request.body.data
                    else:
                        body_dict = json.loads(http_request.body.data)
                    body = replace_variables_recursive(body_dict, variables)
                except json.JSONDecodeError:
                    # If not JSON, treat as string with variable substitution
                    body = replace_variables(http_request.body.data, variables)
                except Exception as e:
                    logger.error(f"Error processing body: {str(e)}")
                    raise Exception(f"Error processing body: {str(e)}")

            # Make the actual request
            result = await self._make_api_request(
                api_request_type=http_request.method.value,
                api_link=replace_variables(http_request.url, variables),
                api_headers=headers,
                api_body=body,
                api_timeout_sec=http_request.timeout,
            )

            return result

        async def process(
            self, context: AtomsAgentContext, node: Node, variables: Dict[str, Any]
        ) -> bool:
            """Handle API node by making request and determining next node based on pathways.

            Args:
                context: The context of the agent
                node: The API call node to process

            Returns:
                True if the API request was successful, False otherwise
            """
            # Make API request
            try:
                response_data = await self._make_api_request_from_node(node, variables)
                context._update_last_user_context(
                    "api_node_response", json.dumps(response_data, indent=2, ensure_ascii=False)
                )
                await self.flow_graph_manager._process_context(context=context)
                return True
            except Exception as e:
                logger.error(f"API request failed: {str(e)}")
                return False

        async def _make_api_request(
            self,
            api_request_type: str,
            api_link: str,
            api_headers: Optional[dict] = None,
            api_body: Optional[Union[dict, str]] = None,
            api_timeout_sec: int = 30,
        ) -> Union[Dict[str, Any], str]:
            """Make an API request with the specified parameters.

            Args:
                api_request_type: HTTP method (GET, POST, etc.)
                api_link: URL for the API request
                api_headers: Headers to include in the request
                api_body: Body content for POST/PUT requests
                api_timeout_sec: Timeout in seconds

            Returns:
                Response data as dictionary or string

            Raises:
                AgentError: If the request fails
            """
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        api_request_type,
                        api_link,
                        data=api_body,
                        headers=api_headers,
                        timeout=api_timeout_sec,
                    ) as response:
                        response.raise_for_status()
                        try:
                            response.raise_for_status()
                            return await response.json()
                        except aiohttp.client_exceptions.ContentTypeError:
                            return await response.text()
                        except Exception as e:
                            logger.error(f"Error parsing response: {str(e)}")
                            return ""
            except Exception as e:
                logger.error(f"Error making API request: {str(e)}")
                return ""

    class AgentInputParams(BaseModel):
        """This class is responsible for passing the input parameters to the agent."""

        current_language: str = "en"
        agent_persona: Optional[Union[str, Dict[str, Any]]] = None
        end_call_tag: str = "<end_call>"
        initial_variables: Optional[Dict[str, Any]] = None
        is_language_switching_enabled: bool = False
        custom_instructions: Optional[List[str]] = []
        global_kb_id: Optional[str] = None

        @field_validator("global_kb_id")
        def validate_global_kb_id(cls, v: Optional[str]) -> Optional[str]:
            if v is not None:
                return v.strip()
            return v

        @field_validator("custom_instructions")
        def validate_custom_instructions(cls, v: Optional[List[str]]) -> List[str]:
            if v is None:
                return []
            return [instruction.strip() for instruction in v if instruction.strip()]

        @field_validator("current_language")
        def validate_current_language(cls, v: str) -> str:
            return v.strip().lower()

    def __init__(
        self,
        flow_model_client: BaseClient,
        variable_extraction_client: BaseClient,
        response_model_client: BaseClient,
        conversation_pathway: ConversationalPathway,
        transport_input_filter: TransportInputFilter,
        agent_input_params: AgentInputParams,
        vector_datastore: DocumentDBVectorStore,
    ):
        super().__init__()
        self.flow_model_client: BaseClient = flow_model_client
        self.variable_extraction_client: BaseClient = variable_extraction_client
        self.response_model_client: BaseClient = response_model_client
        self.conv_pathway: ConversationalPathway = conversation_pathway
        self.current_node: Node = self._find_root()
        self.variables = self._initialize_variables(agent_input_params.initial_variables)
        self._end_call_tag = agent_input_params.end_call_tag
        self.agent_persona = agent_input_params.agent_persona
        self.api_node_handler = self.APICallNodeHandler(self)
        self._event_handlers: Dict[str, List[Callable[..., None]]] = {}
        self.transport_input_filter: TransportInputFilter = transport_input_filter
        self.current_language: str = agent_input_params.current_language
        self._is_language_switching_enabled: bool = agent_input_params.is_language_switching_enabled
        self.custom_instructions: List[str] = agent_input_params.custom_instructions
        self.vector_datastore: DocumentDBVectorStore = vector_datastore
        self.global_kb_id: Optional[str] = agent_input_params.global_kb_id
        self._register_node_event_handlers()

    async def start(self) -> None:
        """Start the flow graph manager."""
        await self._process_pre_call_api_nodes()
        self.conv_pathway.start_node = self.current_node

    @track_latency(
        service_name="agent",
        metric_name="llm_knowledge_base_latency",
        logger=logger,
    )
    async def _handle_llm_knowledge_base(self, context: AtomsAgentContext) -> None:
        """Process knowledge base if configured in current node.

        Args:
            context: Atoms agent context
        """
        if self.current_node.use_global_knowledge_base:
            logger.debug(f"Using global knowledge base: {self.global_kb_id}")
            if self.global_kb_id is None or self.global_kb_id.strip() == "":
                return

            user_transcript = context.get_current_user_transcript()
            if user_transcript and len(user_transcript.split(" ")) > 2:
                chunks = await self.vector_datastore.retrieve(
                    knowledge_base_id=self.global_kb_id, query=user_transcript, limit=4
                )
                knowledge_base = "\n\n".join(chunks)
                self.current_node.knowledge_base = knowledge_base
                logger.debug(f"Knowledge Base Retrieval: {knowledge_base}")

    def _register_node_event_handlers(self) -> None:
        """Register event handlers for the node types."""
        self._register_event_handler("on_api_call_node_started")
        self._register_event_handler("on_api_call_node_ended")

    def _register_event_handler(self, event_name: str) -> None:
        """Register an event handler for the given event name."""
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []

    def event_handler(self, event_name):
        """Decorator for registering event handlers."""

        def decorator(func):
            if event_name not in self._event_handlers:
                self._register_event_handler(event_name)
            self._event_handlers[event_name].append(func)
            return func

        return decorator

    async def _call_event_handler(self, event_name: str, *args, **kwargs):
        """Call the event handler for the given event name."""
        if event_name not in self._event_handlers:
            return
        for handler in self._event_handlers[event_name]:
            if iscoroutinefunction(handler):
                await handler(*args, **kwargs)
            else:
                handler(*args, **kwargs)

    def _detect_language(self, text: str) -> str:
        """Detect language (English or Hindi) based on character analysis.

        Args:
            text: Text to analyze for language detection
            default_language: Default language to return if detection fails or words are less than 3

        Returns:
            Language code ('en' or 'hi')
        """
        HINDI_RANGE = (0x0900, 0x097F)

        try:
            # Split into words and check minimum word requirement
            words = text.strip().split()
            if len(words) < 3:
                return self.current_language

            # Remove whitespace for character counting
            text_no_space = "".join(text.split())
            total_chars = len(text_no_space)

            if total_chars == 0:
                return self.current_language

            # Count only Hindi characters
            hindi_chars = 0
            for char in text_no_space:
                char_code = ord(char)
                if HINDI_RANGE[0] <= char_code <= HINDI_RANGE[1]:
                    hindi_chars += 1

            # Return Hindi if more than 50% characters are Hindi
            return "hi" if (hindi_chars / total_chars) > 0.5 else "en"

        except Exception as e:
            logger.error(
                f"Error processing text for language detection: {e}",
                extra={"call_id": self.call_id},
            )
            return self.current_language

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

    async def _process_pre_call_api_nodes(self) -> None:
        """Process pre-call API nodes sequentially until hitting a non-pre-call node."""
        while self.current_node.type == NodeType.PRE_CALL_API:
            http_request = self.current_node.http_request
            if http_request:
                try:
                    # response_data = self._make_api_request_from_node(self.current_node)
                    response_data = await self.api_node_handler._make_api_request_from_node(
                        self.current_node, self.variables
                    )
                    logger.debug(f"response data from pre-call api node: {response_data}")
                except Exception as e:
                    logger.error(
                        f"Error in API request for node {self.current_node.id}: {str(e)}",
                    )

                # Process response data mappings
                if self.current_node.response_data and self.current_node.response_data.is_enabled:
                    self.api_node_handler._extract_variables(
                        self.current_node, json.dumps(response_data)
                    )

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

    @track_latency(
        service_name="agent",
        metric_name="hopping_latency",
        logger=logger,
    )
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
        if conditional_edges:
            for pathway_id, pathway in conditional_edges:
                if self._evaluate_conditional_edge(pathway.condition):
                    await self._hop(pathway_id)
                    return True

        flow_history = None

        if self.current_node.type == NodeType.API_CALL:
            flow_history = context.get_api_node_flow_navigation_model_context(self.current_node)
        else:
            flow_history = context.get_flow_navigation_model_context(
                current_node=self.current_node, variables=self.variables
            )

        # If no conditional edge matched (or none existed), continue with normal flow
        flow_history = context.get_flow_navigation_model_context(
            current_node=self.current_node, variables=self.variables
        )

        if len(flow_history) < 3:
            return False

        selected_pathway_id = self._cleanup_think_tokens(
            await self.flow_model_client.get_response(flow_history)
        )
        if selected_pathway_id == "null" or not isinstance(selected_pathway_id, str):
            return False

        try:
            logger.debug(
                f"hopping to {selected_pathway_id} for node_type: {self.current_node.type} node_name: {self.current_node.name}"
            )
            self._hop(pathway_id=selected_pathway_id)
            return True
        except Exception as e:
            logger.error(f"Error hopping: {str(e)}")
            return False

    def _hop(self, pathway_id: str) -> None:
        """Transition to a new node via the specified pathway.

        Args:
            pathway_id: ID of the pathway to follow

        Raises:
            Exception: If pathway_id not found in current node
        """
        if not pathway_id in self.current_node.pathways:
            raise Exception(
                f"Pathway '{pathway_id}' not found in current node '{self.current_node.name}'"
            )

        self.previous_node = self.current_node
        self.current_node = self.current_node.pathways[pathway_id].target_node

    @track_latency(
        service_name="agent",
        metric_name="variable_extraction_latency",
        logger=logger,
    )
    async def _extract_variables(self, context: AtomsAgentContext) -> bool:
        """Extract variables from conversation context using LLM.

        Returns:
            Dictionary of extracted variables

        Raises:
            audio_in_filter=KrispFilter(
                model_path=os.getenv("KRISP_MODEL_PATH"),
                suppression_level=90,
            ),
            Exception: If extraction fails or required variables are missing
        """
        variable_schema = self.current_node.variables.data

        # Format variable schema for prompt
        formatted_variable_schema = json.dumps(
            [var.model_dump() for var in variable_schema], indent=2, ensure_ascii=False
        )

        variable_extraction_messages = context._get_variable_extraction_messages()

        # Create extraction prompt
        extraction_prompt = VARIABLE_EXTRACTION_PROMPT.format(
            current_date=self.curr_date,
            current_time=self.curr_time,
            current_day=self.curr_day,
            context="\n".join(variable_extraction_messages),
            variable_schema=formatted_variable_schema,
        )

        logger.debug(f"variable extraction extraction prompt: {extraction_prompt}")

        # Get extraction response from GPT-4o
        raw_extraction = await self.variable_extraction_client.get_response(
            [{"role": "user", "content": extraction_prompt}],
        )

        logger.debug(f"variable extraction raw extraction: {raw_extraction}")

        try:
            # Clean and parse LLM response
            cleaned_json = self._clean_json(raw_extraction)
            extracted_variables = json.loads(cleaned_json)

            logger.debug(f"variable extraction cleaned json: {cleaned_json}")

            # Validate extracted variables
            if not isinstance(extracted_variables, dict):
                raise Exception("LLM response must be a JSON object")

            # Update and return
            self.variables.update(extracted_variables)
            return True

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse variable extraction response: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Failed to extract variables: {str(e)}")
            return False

    def _clean_json(self, text: str) -> str:
        """Clean JSON string from language model output."""
        return text.strip("```").strip("json")

    def _cleanup_think_tokens(self, text: str) -> str:
        return text.replace("<think>", "").replace("</think>", "").strip().strip("\"'")

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

    def _handle_stt_mute(self):
        match self.current_node.type:
            case NodeType.PRE_CALL_API:
                self.transport_input_filter.mute()
            case NodeType.POST_CALL_API:
                self.transport_input_filter.mute()
            case NodeType.DEFAULT:
                self.transport_input_filter.unmute()
            case NodeType.API_CALL:
                self.transport_input_filter.mute()
            case NodeType.END_CALL:
                self.transport_input_filter.mute()
            case NodeType.TRANSFER_CALL:
                self.transport_input_filter.mute()

    def _get_custom_instructions(self) -> str:
        """Get the custom instructions for the response model."""
        return [*self.custom_instructions, get_language_switch_inst(self.current_language)]

    async def _handle_variables_extraction(self, context: AtomsAgentContext) -> None:
        """Handle variables extraction for the current node."""
        match self.current_node.type:
            case NodeType.API_CALL | NodeType.PRE_CALL_API | NodeType.POST_CALL_API:
                if self.current_node.response_data and self.current_node.response_data.is_enabled:
                    self.api_node_handler._extract_variables_from_context(
                        self.current_node, context=context
                    )
            case NodeType.DEFAULT:
                if self.current_node.variables and self.current_node.variables.is_enabled:
                    await self._extract_variables(context=context)

    def _handle_language_switching(self, context: AtomsAgentContext) -> None:
        """Handle language switching for the current node."""
        if self._is_language_switching_enabled:
            current_user_transcript = context.get_current_user_transcript()
            self.current_language = self._detect_language(current_user_transcript)

    def _update_user_context(self, context: AtomsAgentContext) -> None:
        current_user_transcript = context.get_current_user_transcript()
        context._update_last_user_context(
            "response_model_context",
            self._get_current_state_as_json(
                user_response=current_user_transcript,
                custom_instructions=self._get_custom_instructions(),
            ),
        )
        delta = context.get_user_context_delta()
        if delta:
            delta = json.dumps(delta, indent=2, ensure_ascii=False)
            context._update_last_user_context("delta", delta)

    async def get_response(self, context: AtomsAgentContext):
        """Get the response from the response model client."""
        # Before hopping we need to extract variables from the current node and user response
        await self._handle_llm_knowledge_base(context=context)
        self._handle_language_switching(context=context)
        await self._handle_variables_extraction(context=context)
        hopped = await self._handle_hopping(context=context)
        if self.current_node.type == NodeType.API_CALL:
            if not hopped:
                yield "Something went wrong, please try again later."
                await self.push_frame(LastTurnFrame(conversation_id="123"))
                logger.debug(f"hopping failed for api call node")
                return
        self._handle_stt_mute()
        self._update_user_context(context=context)
        if self.current_node.type == NodeType.DEFAULT:
            if self.current_node.static_text:
                for chunk in self._handle_static_response(context=context):
                    yield chunk
            else:
                async for chunk in self._handle_dynamic_response(context=context):
                    yield chunk
        elif self.current_node.type == NodeType.END_CALL:
            if self.current_node.static_text:
                for chunk in self._handle_static_response(context=context):
                    yield chunk
                await self.push_frame(LastTurnFrame(conversation_id="123"))
            else:
                async for chunk in self._handle_dynamic_response(context=context):
                    yield chunk
                await self.push_frame(LastTurnFrame(conversation_id="123"))
        elif self.current_node.type == NodeType.TRANSFER_CALL:
            if self.current_node.static_text:
                for chunk in self._handle_static_response(context=context):
                    yield chunk
                await self.push_frame(
                    SetTransferCallDataFrame(
                        transfer_call_number=self.current_node.transfer_number,
                        conversation_id="123",
                    )
                )
        elif self.current_node.type == NodeType.API_CALL:
            if self.current_node.static_text:
                for chunk in self._handle_static_response(context=context):
                    yield chunk
            else:
                async for chunk in self._handle_dynamic_response(context=context):
                    yield chunk
            # Wait for the response to complete before processing API call
            await self.api_node_handler.process(
                context=context, node=self.current_node, variables=self.variables
            )
        else:
            raise Exception(f"Unknown node type: {self.current_node.type}")

        self.transport_input_filter.unmute()

    async def _process_context(self, context: AtomsAgentContext) -> None:
        """Process the context and update the flow model client."""
        response = self.get_response(context=context)
        if isgenerator(response):
            for chunk in response:
                await self.push_frame(LLMTextFrame(text=chunk))
        elif isasyncgen(response):
            async for chunk in response:
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
        add_default_variables: bool = False,
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
            "agent_persona": self.agent_persona,
        }

        if self.current_node.knowledge_base and self.current_node.knowledge_base.strip():
            current_state["knowledge_base"] = self.current_node.knowledge_base
        if add_default_variables:
            current_state["variables"] = {
                "current_date": self.curr_date,
                "current_day": self.curr_day,
                "current_time": self.curr_time,
            }

        if custom_instructions:
            current_state["custom_instructions"] = custom_instructions
        if current_state.get("loop_condition") is None or (
            isinstance(current_state["loop_condition"], str)
            and current_state["loop_condition"].strip() == ""
        ):
            current_state.pop("loop_condition")

        # Return full state if no delta or node changed
        return json.dumps(current_state, indent=2, ensure_ascii=False)

    def _handle_static_response(self, context: AtomsAgentContext) -> Generator[str, None, None]:
        """Handle the static response from the response model client."""
        yield self.current_node.action

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

                # TODO: This is a not a good way to but the same event handlers can be called multiple times for the same node
                # why did we did this? -> because FlowGraphManager can be interrupted and it end might not be called
                # we need the better way to handle this
                await self._call_event_handler("on_api_call_node_ended")


async def initialize_conversational_agent(
    *, agent_id: str, call_id: str, call_data: CallData, transport_input_filter: Any
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
            transport_input_filter=transport_input_filter,
            agent_input_params=FlowGraphManager.AgentInputParams(
                initial_variables=initial_variables,
                agent_persona=agent_persona,
                current_language=agent_language,
                is_language_switching_enabled=language_switching,
            ),
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
                f"{os.getenv('ATOMS_API_ENDPOINT')}/api/v1/admin/get-agent-details",
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

            agent_gender = agent_config.get("synthesizer_args", {}).get("gender", "female")

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

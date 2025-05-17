import concurrent.futures
import json
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Union

import pytz
import requests
from jsonpath_ng import parse
from loguru import logger

from .llm_client import AzureOpenAIClient, BaseClient
from .pathways import ConversationalPathway, Node, NodeType, ResponseDataConfig
from .prompts import VARIABLE_EXTRACTION_PROMPT


class EventType(str, Enum):
    """Event types for tracking agent operations."""

    USER_INPUT = "user_input"
    AGENT_RESPONSE = "agent_response"
    NODE_HOPPING = "node_hopping"
    KNOWLEDGE_BASE = "knowledge_base"
    VARIABLE_EXTRACTION = "variable_extraction"
    API_REQUEST_INPUT = "api_request_input"
    API_REQUEST_OUTPUT = "api_request_output"
    ERROR = "error"


class BaseConversationalAgent(ABC):
    """
    Abstract base class for conversational agents.
    Provides common functionality for different agent implementations.
    """

    def __init__(
        self,
        conv_pathway: ConversationalPathway,
        llm_client: BaseClient,
        language_switching: bool = False,
        default_language: Literal["en", "hi"] = "en",
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        initial_variables: Optional[Dict[str, Any]] = None,
        agent_gender: Optional[Literal["male", "female"]] = None,
        knowledge_base_callback: Optional[Callable] = None,
        global_prompt: Optional[str] = None,
        global_kb_id: Optional[str] = None,
        call_id: Optional[str] = None,
    ) -> None:
        """
        Initialize the base conversational agent.

        Args:
            conv_pathway: Conversational pathway defining nodes and transitions
            llm_client: Language model client for generating responses
            language_switching: Enable language detection and switching
            default_language: Default language for responses
            initial_variables: Initial variables for the conversation
            agent_gender: Gender of the agent for persona configuration
            knowledge_base_callback: Function to retrieve knowledge base content
            global_kb_id: ID of the global knowledge base
            call_id: Unique identifier for the current conversation
        """
        # Core components
        self.conv_pathway = conv_pathway
        self.client = llm_client
        self.language_switching = language_switching
        self.language_now = default_language
        self.agent_gender = agent_gender
        self.knowledge_base_callback = knowledge_base_callback
        self.global_prompt = global_prompt
        self.global_kb_id = global_kb_id
        self.call_id = call_id

        # Internal variables
        self._last_state = None
        self._intermediate_response_callback = None
        self._transfer_call_callback = None
        self._is_post_call_complete = False

        # Events system
        self.event_queue = deque()
        self.event_history = []

        # Initialize variables
        self.variables = self._initialize_variables(initial_variables)
        self.agent_persona = {"gender": agent_gender} if agent_gender else None

        # Initialize variable extraction client
        self.variable_extraction_client = AzureOpenAIClient(
            model_id="gpt-4o",
            api_version="2024-12-01-preview",
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            keep_history=False,
            call_id=call_id,
        )

        # Find the true start node (node with no incoming connections)
        true_start_node = self._find_true_start_node()
        self._log(f"Found true start node: {true_start_node.id} - {true_start_node.name}")

        # Process Pre-Call API nodes if the true start node is a PRE_CALL_API node
        if true_start_node.type == NodeType.PRE_CALL_API:
            self._log("Starting with PRE_CALL_API nodes...")
            # Start from the true start node and process pre-call sequence
            self.current_node: Node = true_start_node
            self._process_pre_call_api_nodes()
        else:
            # If not a pre-call node, verify it has is_start_node=True
            if not true_start_node.is_start_node:
                raise AgentError(
                    f"Node {true_start_node.id} has no incoming edges but is_start_node is not True"
                )
            self.current_node: Node = true_start_node

        # At this point, current_node is the actual start node for the agent
        self.conv_pathway.start_node = self.current_node

    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def get_response(
        self,
        prompt: Optional[str],
        custom_instructions: Optional[list] = None,
        add_variables: bool = False,
        add_agent_persona: bool = False,
        **kwargs,
    ) -> Union[Dict[str, Any], str, Iterator[Dict[str, Any]]]:
        """
        Get a response from the language model client.

        Args:
            prompt: User input text
            custom_instructions: Additional instructions for the LLM
            add_variables: Whether to include variables in context
            add_agent_persona: Whether to include agent persona in context
            **kwargs: Additional arguments for the LLM client

        Returns:
            Response from the language model
        """
        pass

    @abstractmethod
    def _get_response(
        self, prompt: str, use_static_text: bool, **kwargs
    ) -> Union[str, Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Internal method to get response from language model.

        Args:
            prompt: Formatted prompt for the language model
            use_static_text: Whether to use static text from the node
            **kwargs: Additional arguments

        Returns:
            Response from the language model
        """
        pass

    @abstractmethod
    def _hop(self, pathway_id: str) -> None:
        """
        Transition to a new node via the specified pathway.

        Args:
            pathway_id: ID of the pathway to follow
        """
        pass

    @abstractmethod
    def _handle_hopping(self, response: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        Handle node hopping logic.

        Args:
            response: Response data containing potential pathway information

        Returns:
            Tuple of (updated response, whether hopping occurred)
        """
        pass

    @abstractmethod
    def _initialize_conversation(self, **kwargs) -> None:
        """
        Initialize the conversation with the language model.

        Args:
            **kwargs: Implementation-specific arguments
        """
        pass

    @abstractmethod
    def _let_llm_decide_pathway(
        self, response_data: Union[Dict[str, Any], str], pathways: List[Tuple[str, Any]]
    ) -> str:
        """
        Let the language model decide which pathway to take.

        Args:
            response_data: API response data
            pathways: List of pathway options

        Returns:
            Selected pathway ID
        """
        pass

    # Intermediate callback for API Call Node
    def set_intermediate_response_callback(self, callback: Callable) -> None:
        """
        Set callback function to be called when intermediate responses are generated.

        Args:
            callback: Function to call with intermediate responses
        """
        self._intermediate_response_callback = callback

    def set_transfer_call_callback(self, callback: Callable):
        """
        Set the callback function to be called when the call is transferred.
        """
        self._transfer_call_callback = callback

    # Event management methods
    def add_event(self, event_type: EventType, details: Dict[str, Any]) -> None:
        """
        Add an event to the queue and history.

        Args:
            event_type: Type of the event
            details: Event-specific details
        """
        event = {
            "type": event_type.name,
            "timestamp": datetime.now().isoformat(),
            "call_id": self.call_id,
            "current_node": {
                "id": self.current_node.id,
                "name": self.current_node.name,
                "type": self.current_node.type.value,
                "action": self.current_node.action,
                "loop_condition": self.current_node.loop_condition,
            },
            "variables": self.variables,
            "details": details,
        }
        self.event_queue.append(event)
        self.event_history.append(event)
        self._log(f"Event added: {event_type.value}")

    def get_event(self) -> Optional[Dict[str, Any]]:
        """
        Get the next event from the queue.

        Returns:
            The next event or None if queue is empty
        """
        return self.event_queue.popleft() if self.event_queue else None

    def get_event_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete event history.

        Returns:
            List of all events recorded
        """
        return self.event_history

    # Common methods for all agent implementations
    def _initialize_variables(
        self, initial_variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Initialize variables with datetime information and any provided initial values.

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

    def _find_true_start_node(self) -> Node:
        """
        Find the true start node of the conversation pathway (node with no incoming edges).

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
            raise AgentError(
                "No start node found in the conversation pathway. At least one node must have no incoming edges."
            )

        if len(nodes_without_incoming) > 1:
            raise AgentError(
                f"Multiple potential start nodes found: {nodes_without_incoming}. Only one node should have no incoming edges."
            )

        return self.conv_pathway.nodes[nodes_without_incoming[0]]

    def _process_knowledge_base(self, prompt: Optional[str]) -> None:
        """
        Process knowledge base if configured in current node.

        Args:
            prompt: User input to use for knowledge retrieval
        """
        if self.current_node.use_global_knowledge_base:
            if self.knowledge_base_callback is None:
                raise AgentError("Knowledge Base Callback is required for knowledge base usage")

            if self.global_kb_id is None or self.global_kb_id.strip() == "":
                self._log("Global Knowledge Base ID is not set. Skipping Knowledge Base Usage.")
                return

            if prompt and len(prompt.split(" ")) > 2:
                chunks = self.knowledge_base_callback(
                    knowledge_base_id=self.global_kb_id, query=prompt, limit=4, call_id=self.call_id
                )
                self.current_node.knowledge_base = "\n\n".join(chunks)

                self._log(f"Knowledge Base Retrieval: {self.current_node.knowledge_base}")

                # Add knowledge base event
                self.add_event(
                    EventType.KNOWLEDGE_BASE,
                    {
                        "kb_id": self.global_kb_id,
                        "query": prompt,
                        "retrieved_content": self.current_node.knowledge_base,
                    },
                )

    def _get_language_switch_inst(self) -> str:
        """
        Get language switching instruction based on current language.

        Returns:
            Instruction for language switching
        """
        if self.language_now not in {"en", "hi"}:
            raise ValueError(f"Unsupported language: {self.language_now}")

        base_template = "Respond in {}"

        if self.language_now == "en":
            return base_template.format("English")

        elif self.language_now == "hi":
            return base_template.format("Hindi")

    def _detect_language(self, text: str) -> str:
        """
        Detect language (English or Hindi) based on character analysis.

        Args:
            text: Text to analyze for language detection

        Returns:
            Language code ('en' or 'hi')
        """
        HINDI_RANGE = (0x0900, 0x097F)

        try:
            # Split into words and check minimum word requirement
            words = text.strip().split()
            if len(words) < 3:
                return self.language_now

            # Remove whitespace for character counting
            text_no_space = "".join(text.split())
            total_chars = len(text_no_space)

            if total_chars == 0:
                return "en"

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
            return "en"

    def _replace_variables(self, input_string: Any, variables: Dict[str, Any]) -> Any:
        """
        Replace variable placeholders in a string with their values.

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
                    f"Variable replacement failed for '{variable_name}' in node {self.current_node.id}",
                    extra={
                        "call_id": self.call_id,
                        "category": "user",
                        "error_message": "Unexpected variable usage in workflow",
                    },
                )
                return match.group(0)

        if isinstance(input_string, str):
            return re.sub(pattern, replace_func, input_string)
        else:
            return input_string

    def _replace_variables_recursive(self, data: Any, variables: dict) -> Any:
        """
        Replace variables in nested data structures (dicts, lists, strings).

        Args:
            data: Object potentially containing variable placeholders
            variables: Dictionary of variable names and values

        Returns:
            Object with variables replaced
        """
        if isinstance(data, str):
            return self._replace_variables(data, variables)
        elif isinstance(data, dict):
            return {k: self._replace_variables_recursive(v, variables) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._replace_variables_recursive(item, variables) for item in data]
        else:
            return data

    def _process_pre_call_api_nodes(self) -> None:
        """
        Process pre-call API nodes sequentially until hitting a non-pre-call node.
        """
        while self.current_node.type == NodeType.PRE_CALL_API:
            self._log(
                f"Processing PRE_CALL_API node: {self.current_node.id} - {self.current_node.name}"
            )

            # Make API request
            http_request = self.current_node.http_request
            if http_request:
                try:
                    response_data = self._make_api_request_from_node(self.current_node)
                except Exception as e:
                    logger.error(
                        f"Error in API request for node {self.current_node.id}: {str(e)}",
                        extra={
                            "call_id": self.call_id,
                            "category": "user",
                            "error_message": "PRE_CALL_API node failure",
                        },
                    )

                # Process response data mappings
                if self.current_node.response_data and self.current_node.response_data.is_enabled:
                    self._extract_response_data(response_data, self.current_node.response_data)

            # Navigate to the next node - Pre-call nodes should only have one pathway
            if len(self.current_node.pathways) != 1:
                raise AgentError(
                    f"PRE_CALL_API node {self.current_node.id} must have exactly one pathway"
                )

            # Get the first (and only) pathway
            next_node = next(iter(self.current_node.pathways.values())).target_node
            self.current_node = next_node

        # We've now reached a non-pre-call node. Verify it has is_start_node=True
        if not self.current_node.is_start_node:
            raise AgentError(
                f"Non-PRE_CALL_API node {self.current_node.id} at end of pre-call sequence "
                f"must have is_start_node=True"
            )

        self._log(
            f"Finished pre-call sequence. Starting with node: {self.current_node.id} - {self.current_node.name}"
        )

    def _process_post_call_api_nodes_async(self) -> threading.Thread:
        """
        Process post-call API nodes in a separate thread without blocking.

        Returns:
            Thread handling the post-call processing
        """
        thread = threading.Thread(
            target=self._process_post_call_api_nodes_thread, name="post_call_api_thread"
        )
        thread.daemon = True
        thread.start()
        return thread

    def _process_post_call_api_nodes_thread(self) -> None:
        """Thread-safe version of _process_post_call_api_nodes to run in a separate thread."""
        try:
            if self.current_node.type != NodeType.END_CALL:
                return

            self._log(
                f"Reached END_CALL node: {self.current_node.id}. Starting POST_CALL_API processing in background thread."
            )

            # DFS to find and process all post-call API paths
            visited = set()

            def dfs(node: Node):
                if node.id in visited:
                    return

                visited.add(node.id)

                if node.type == NodeType.POST_CALL_API:
                    self._log(f"Processing POST_CALL_API node: {node.id} - {node.name}")

                    # Make API request
                    http_request = node.http_request
                    if http_request:
                        try:
                            response_data = self._make_api_request_from_node(node)

                            # Process response data mappings
                            if node.response_data and node.response_data.is_enabled:
                                self._extract_response_data(response_data, node.response_data)
                        except Exception as e:
                            logger.error(
                                f"Error in API request for node {node.id}: {str(e)}",
                                extra={
                                    "call_id": self.call_id,
                                    "category": "user",
                                    "error_message": "POST_CALL_API node failure",
                                },
                            )

                    # Continue DFS on all pathways
                    for pathway in node.pathways.values():
                        dfs(pathway.target_node)

            # To handle multiple pathways in parallel, we can use ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for pathway in self.current_node.pathways.values():
                    futures.append(executor.submit(dfs, pathway.target_node))
                concurrent.futures.wait(futures)

            self._log("Finished processing all POST_CALL_API nodes in background thread")
        except Exception as e:
            self._log(f"Error in background post-call API processing: {str(e)}")

    def _make_api_request_from_node(self, node: Node) -> Union[Dict[str, Any], str]:
        """
        Make an API request based on node configuration.

        Args:
            node: The node containing the HTTP request configuration

        Returns:
            API response data

        Raises:
            AgentError: If the request fails or is misconfigured
        """
        # Extract request parameters
        http_request = node.http_request
        if not http_request:
            raise AgentError(f"Node {node.id} has no HTTP request configuration")

        # Process headers with variable substitution
        headers = {}
        if http_request.headers and http_request.headers.is_enabled:
            for key, value in http_request.headers.data.items():
                headers[key] = self._replace_variables(value, self.variables)

        # Process authorization if enabled
        if http_request.authorization and http_request.authorization.is_enabled:
            auth_data = http_request.authorization.data
            if auth_data and auth_data.token:
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
                body = self._replace_variables_recursive(body_dict, self.variables)
            except json.JSONDecodeError:
                # If not JSON, treat as string with variable substitution
                body = self._replace_variables(http_request.body.data, self.variables)

        # Log the API request
        self._log_api_request(
            node, http_request.method.value, http_request.url, headers, body, http_request.timeout
        )

        # Add API request event
        self.add_event(
            EventType.API_REQUEST_INPUT,
            {
                "request_type": http_request.method.value,
                "url": http_request.url,
                "headers": headers,
                "body": body,
                "timeout": http_request.timeout,
            },
        )

        # Make the actual request
        result = self._make_api_request(
            api_request_type=http_request.method.value,
            api_link=self._replace_variables(http_request.url, self.variables),
            api_headers=headers,
            api_body=body,
            api_timeout_sec=http_request.timeout,
        )

        # Add API response event
        self.add_event(
            EventType.API_REQUEST_OUTPUT,
            {
                "response_data": result,
                "is_json": isinstance(result, dict),
            },
        )

        return result

    def _log_api_request(
        self, node: Node, method: str, url: str, headers: Dict[str, str], body: Any, timeout: int
    ) -> None:
        """
        Log API request details.

        Args:
            node: Node making the request
            method: HTTP method
            url: Request URL
            headers: Request headers
            body: Request body
            timeout: Request timeout
        """
        self._log(
            f"API Request on node {node.id} - {method} {url}\n"
            f"Headers: {headers}\n"
            f"Body: {body}\n"
            f"Timeout: {timeout}s"
        )

    def _make_api_request(
        self,
        api_request_type: str,
        api_link: str,
        api_headers: Optional[dict] = None,
        api_body: Optional[Union[dict, str]] = None,
        api_timeout_sec: int = 30,
    ) -> Union[Dict[str, Any], str]:
        """
        Make an API request with the specified parameters.

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
            # Map the request type to the appropriate requests method
            request_method = {
                "GET": requests.get,
                "POST": requests.post,
                "PUT": requests.put,
                "DELETE": requests.delete,
                "PATCH": requests.patch,
            }.get(api_request_type.upper())

            if not request_method:
                raise AgentError(f"Invalid API request type: {api_request_type}")

            # Make the request with appropriate parameters based on method type
            if api_request_type.upper() in ["GET", "DELETE"]:
                response = request_method(
                    url=api_link,
                    headers=api_headers,
                    params=api_body if isinstance(api_body, dict) else None,
                    timeout=api_timeout_sec,
                )
            else:
                if isinstance(api_body, dict):
                    response = request_method(
                        url=api_link,
                        headers=api_headers,
                        json=api_body,
                        timeout=api_timeout_sec,
                    )
                else:
                    response = request_method(
                        url=api_link,
                        headers=api_headers,
                        data=api_body,
                        timeout=api_timeout_sec,
                    )

            # Raise an exception if the response was unsuccessful
            response.raise_for_status()

            # Return the JSON response or text
            try:
                return response.json()
            except requests.exceptions.JSONDecodeError:
                if response.text.strip():
                    return response.text
                else:
                    return ""

        except requests.exceptions.HTTPError as e:
            # This captures the errors raised by raise_for_status()
            self._log(f"API Request failed with status code {e.response.status_code}: {str(e)}")
            self.add_event(
                EventType.ERROR,
                {
                    "error_message": f"API request failed with status code {e.response.status_code}: {str(e)}"
                },
            )
            raise AgentError(
                f"API Request failed with status code {e.response.status_code}: {str(e)}"
            )
        except requests.exceptions.RequestException as e:
            # This captures other request-related errors (connection, timeout, etc.)
            self._log(f"API Request failed: {str(e)}")
            self.add_event(
                EventType.ERROR,
                {"error_message": f"API request failed: {str(e)}"},
            )
            raise AgentError(f"API Request failed: {str(e)}")

    def _handle_api_call_node(self, node: Node) -> Dict[str, Any]:
        """
        Handle API node by making request and determining next node based on pathways.

        Args:
            node: The API call node to process

        Returns:
            Response data

        Raises:
            AgentError: If node processing fails
        """
        # Extract variables if configured
        if node.variables and node.variables.is_enabled:
            extracted_vars = self._extract_variables()
            self._log(f"Extracted variables: {extracted_vars}")

        # Make API request
        try:
            response_data = self._make_api_request_from_node(node)
            api_success = True
        except AgentError as e:
            logger.error(
                f"API request failed: {str(e)}",
                extra={
                    "call_id": self.call_id,
                    "category": "user",
                    "error_message": "API node failure",
                },
            )
            response_data = {"error": str(e)}
            api_success = False

        # Extract data from response if configured
        if api_success and node.response_data and node.response_data.is_enabled:
            self._extract_response_data(response_data, node.response_data)

        # Process pathways based on the response
        pathway_id_or_response = self._process_api_node_pathways(node, response_data, api_success)

        if isinstance(pathway_id_or_response, dict):
            return pathway_id_or_response
        elif isinstance(pathway_id_or_response, str):
            self._hop(pathway_id_or_response)
            return self.get_response(
                f"api output = {json.dumps(response_data, indent=2, ensure_ascii=False)}"
            )
        else:
            self.add_event(
                EventType.ERROR,
                {
                    "error_message": "API request completed but no branch was determined",
                    "response_data": response_data,
                },
            )

            raise AgentError(
                "API request completed but no pathway was determined. Response: "
                f"{json.dumps(response_data, indent=2, ensure_ascii=False)}. Output: {pathway_id_or_response}"
            )

    def _process_api_node_pathways(
        self, node: Node, response_data: Union[Dict[str, Any], str], api_success: bool
    ) -> Optional[str]:
        """
        Process API node pathways based on variables and return the selected pathway ID.

        Args:
            node: The API call node
            response_data: Response data from the API request
            api_success: Whether the API request was successful

        Returns:
            Selected pathway ID or None if no pathway was selected
        """
        # Categorize pathways
        normal_edges = []
        conditional_edges = []
        fallback_edge = None

        for pathway_id, pathway in node.pathways.items():
            if pathway.is_fallback_edge:
                fallback_edge = (pathway_id, pathway)
            elif pathway.is_conditional_edge:
                conditional_edges.append((pathway_id, pathway))
            else:
                normal_edges.append((pathway_id, pathway))

        # Handle API failure case first - use fallback edge if available
        if not api_success:
            if fallback_edge:
                self._log(f"API request failed, using fallback edge: {fallback_edge[0]}")
                return fallback_edge[0]
            else:
                raise AgentError(
                    f"API request failed and no fallback edge defined for node {node.id}"
                )

        # Check conditional edges if API was successful
        # Try to evaluate conditional edges using variables stored in self.variables
        for pathway_id, pathway in conditional_edges:
            condition = pathway.condition
            if self._evaluate_conditional_edge(condition):
                self._log(f"Conditional edge matched: {pathway_id} with condition {condition}")
                return pathway_id

        # If no conditional edge matched, let LLM decide from normal edges
        if normal_edges:
            pathways_for_llm = [(pathway_id, pathway) for pathway_id, pathway in normal_edges]

            # Add fallback as an option if no conditional edges matched
            if fallback_edge:
                pathways_for_llm.append(fallback_edge)

            # Let LLM decide which pathway to take
            return self._let_llm_decide_pathway(response_data, pathways_for_llm)
        elif fallback_edge:
            # Use fallback if no normal edges and no conditional edge matched
            self._log(f"No conditional edge matched, using fallback edge: {fallback_edge[0]}")
            return fallback_edge[0]

        # If we got here, there are no valid pathways
        self.add_event(
            EventType.ERROR,
            {
                "error_message": f"No valid branch found for API node '{node.name}'",
                "response_data": response_data,
            },
        )
        raise AgentError(f"No valid pathways found for API node {node.id}")

    def _evaluate_conditional_edge(self, condition: str) -> bool:
        """
        Evaluate conditional edge expressions against variables using Python's eval.

        Args:
            condition: Condition to evaluate, e.g. "{{variable_name}} == 'value'"

        Returns:
            Whether the condition is satisfied
        """
        try:
            # Extract the variable name
            match = re.search(r"{{(.+?)}}", condition)
            if not match:
                self._log(f"No variable found in condition: {condition}")
                return False

            variable_name = match.group(1).strip()

            # Check if the variable exists
            if variable_name not in self.variables:
                self.add_event(
                    EventType.ERROR,
                    {
                        "error_message": f"Variable '{variable_name}' not found in variables in condition '{condition}'",
                    },
                )

                self._log(
                    f"Variable '{variable_name}' not found in variables when evaluating condition."
                )
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

            # Log the final condition for debugging
            self._log(f"Evaluating condition: {python_condition}")

            # Evaluate using Python's eval
            result = eval(python_condition)
            return bool(result)

        except Exception as e:
            self._log(f"Error evaluating condition '{condition}': {str(e)}")
            return False

    def _extract_response_data(
        self, response_data: Union[Dict[str, Any], str], config: ResponseDataConfig
    ) -> None:
        """
        Extract data from API response using JSON paths.

        Args:
            response_data: Response data from API call
            config: Configuration for response data extraction
        """
        if not config.is_enabled or not config.data:
            return

        # Skip if response is not a dict
        if not isinstance(response_data, dict):
            self.add_event(
                EventType.ERROR,
                {
                    "error_message": "Unable to extract data from API response as it is not in JSON format. Please ensure API returns JSON data.",
                    "response_data": response_data,
                },
            )

            raise AgentError("Cannot extract data from non-JSON response")

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
                    self._log(f"Extracted {mapping.variable_name}: {value}")
                else:
                    self._log(f"No match found for JSON path: {mapping.json_path}")

            except Exception as e:
                self._log(f"Error extracting data with path '{mapping.json_path}': {str(e)}")

        # Update variables with extracted data
        if extracted:
            self.variables.update(extracted)
            self._log(f"Updated variables with response data: {extracted}")

    def _extract_variables(self, latest_user_response: str = None) -> Dict[str, Any]:
        """
        Extract variables from conversation context using LLM.

        Returns:
            Dictionary of extracted variables

        Raises:
            AgentError: If extraction fails or required variables are missing
        """
        start_time = time.time()

        # Get variable schema
        if not self.current_node.variables or not self.current_node.variables.is_enabled:
            return {}

        variable_schema = self.current_node.variables.data

        if not variable_schema:
            return {}

        # Format variable schema for prompt
        formatted_variable_schema = json.dumps(
            [var.model_dump() for var in variable_schema], indent=2, ensure_ascii=False
        )

        # Collect conversation context
        context = []

        for msg in self.client.messages[2:]:
            if msg["role"] == "user":
                try:
                    content = msg["content"]
                    if isinstance(content, str) and content.startswith("{"):
                        user_content = json.loads(content)
                        user_response = user_content.get("user_response") or user_content.get(
                            "delta", {}
                        ).get("user_response")
                        if user_response:
                            context.append(f"User: {user_response}")
                    else:
                        context.append(f"User: {content}")
                except Exception:
                    context.append(f"User: {content}")
            elif msg["role"] == "assistant":
                try:
                    content = msg["content"]
                    if isinstance(content, str) and content.startswith("{"):
                        assistant_content = json.loads(content)
                        response = assistant_content.get("response", content)
                        context.append(f"Assistant: {response}")
                    else:
                        context.append(f"Assistant: {content}")
                except Exception:
                    context.append(f"Assistant: {content}")

        if latest_user_response:
            context.append(f"User: {latest_user_response}")

        # Create extraction prompt
        extraction_prompt = VARIABLE_EXTRACTION_PROMPT.format(
            current_date=self.curr_date,
            current_time=self.curr_time,
            current_day=self.curr_day,
            context="\n".join(context),
            variable_schema=formatted_variable_schema,
        )
        self._log(f"Extraction Prompt: {extraction_prompt}")

        # Get extraction response from GPT-4o
        raw_extraction = self.variable_extraction_client.get_response(
            [{"role": "user", "content": extraction_prompt}],
        )

        latency_ms = (time.time() - start_time) * 1000
        self._log(f"Variable Extraction: {raw_extraction}")
        self._log(f"Variable Extraction Latency: {latency_ms:.2f} ms")

        try:
            # Clean and parse LLM response
            cleaned_json = self._clean_json(raw_extraction)
            extracted_variables = json.loads(cleaned_json)

            # Validate extracted variables
            if not isinstance(extracted_variables, dict):
                raise AgentError("LLM response must be a JSON object")

            # Add variable extraction event
            self.add_event(
                EventType.VARIABLE_EXTRACTION,
                {
                    "variable_schema": [var.model_dump() for var in variable_schema],
                    "extracted_variables": extracted_variables,
                    "latency_ms": latency_ms,
                },
            )

            # Update and return
            self.variables.update(extracted_variables)
            return extracted_variables

        except json.JSONDecodeError as e:
            self._log(f"Failed to parse variable extraction response: {str(e)}")
            raise AgentError(f"Failed to parse variable extraction response: {str(e)}")

    def _clean_json(self, text: str) -> str:
        """
        Clean JSON string from language model output.

        Args:
            text: Text containing JSON

        Returns:
            Cleaned JSON string
        """
        return text.strip("```").strip("json")

    def _safe_json_loads(self, text: str) -> Dict:
        """
        Safely parse JSON string.

        Args:
            text: JSON string to parse

        Returns:
            Parsed JSON object

        Raises:
            json.JSONDecodeError: If parsing fails
        """
        return json.loads(text, strict=False)

    def _log(self, message: str) -> None:
        """
        Log a message.

        Args:
            message: Message to log
        """
        logger.info(message)


class AgentError(Exception):
    """Custom exception for errors during Agent execution"""

    pass

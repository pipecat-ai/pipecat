import copy
import json
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Union

from .base_agent import AgentError, BaseConversationalAgent, EventType
from .llm_client import BaseClient
from .pathways import ConversationalPathway, NodeType
from .prompts import FT_FLOW_MODEL_SYSTEM_PROMPT, FT_RESPONSE_MODEL_SYSTEM_PROMPT


class AtomsConversationalAgent(BaseConversationalAgent):
    """
    Conversational agent that navigates through a predefined pathway,
    generating responses based on user input and executing API calls.

    This agent supports streaming responses and uses separate models
    for response generation and flow navigation.
    """

    def __init__(
        self,
        conv_pathway: ConversationalPathway,
        response_model_client: BaseClient,
        flow_model_client: BaseClient,
        language_switching: bool = False,
        default_language: Literal["en", "hi"] = "en",
        initialize_first_message: bool = True,
        initial_variables: Optional[Dict[str, Any]] = None,
        agent_gender: Literal["male", "female"] = "male",
        knowledge_base_callback: Optional[Callable] = None,
        global_prompt: Optional[str] = None,
        global_kb_id: Optional[str] = None,
        call_id: Optional[str] = None,
        override_response_system_prompt: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
    ):
        """
        Initialize the Atoms conversational agent.

        Args:
            conv_pathway: The conversational pathway defining nodes and transitions
            response_model_client: LLM client for generating responses
            flow_model_client: LLM client for navigation decisions
            language_switching: Enable language detection and switching
            default_language: Default language for responses
            initialize_first_message: Whether to generate an initial message
            initial_variables: Initial variables for the conversation
            agent_gender: Gender of the agent for persona configuration
            knowledge_base_callback: Function to retrieve knowledge base content
            global_kb_id: ID of the global knowledge base
            call_id: Unique identifier for the current conversation

        Raises:
            ValueError: If flow model client has keep_history enabled
        """
        if flow_model_client.keep_history:
            raise ValueError("Flow Navigation Model's keep_history should be False")

        # Initialize with response model client
        super().__init__(
            conv_pathway=conv_pathway,
            llm_client=response_model_client,
            language_switching=language_switching,
            default_language=default_language,
            initial_variables=initial_variables,
            agent_gender=agent_gender,
            knowledge_base_callback=knowledge_base_callback,
            global_prompt=global_prompt,
            global_kb_id=global_kb_id,
            call_id=call_id,
            azure_api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
        )

        # Atoms-specific attributes
        self.response_model_client = response_model_client
        self.flow_model_client = flow_model_client
        self.response_model_messages = self.response_model_client.messages
        self.override_response_system_prompt = override_response_system_prompt

        # End call configuration in client
        self._end_call_tag = "<end_call>"

        stop_sequences = self.response_model_client.default_response_kwargs.get("stop", [])

        if not isinstance(stop_sequences, list):
            stop_sequences = [stop_sequences] if stop_sequences else []

        if self._end_call_tag not in stop_sequences:
            stop_sequences.append(self._end_call_tag)

        self.response_model_client.default_response_kwargs["stop"] = stop_sequences

        # Initialize conversation
        self._initialize_conversation(initialize_first_message)

    def get_response(
        self,
        prompt: Optional[str],
        stream: bool = True,
        custom_instructions: Optional[list] = None,
        add_variables: bool = False,
        add_agent_persona: bool = False,
        skip_hop: bool = False,
        **client_kwargs,
    ) -> Union[Dict[str, Any], str, Iterator[Dict[str, Any]]]:
        """
        Generate a response based on user input.

        Args:
            prompt: User input text
            stream: Whether to stream the response
            custom_instructions: Additional instructions for the LLM
            add_variables: Whether to include variables in context
            add_agent_persona: Whether to include agent persona in context
            skip_hop: Whether to skip node transition logic
            **client_kwargs: Additional arguments for the LLM client

        Returns:
            Response text or stream iterator
        """
        self._log(f"Human Input: {prompt}")

        # Add user input event
        if prompt:
            self.add_event(
                EventType.USER_INPUT,
                {
                    "user_input": prompt,
                    "language": self.language_now,
                },
            )

        if isinstance(prompt, str):
            prompt = prompt.strip()

        if not skip_hop:
            self._handle_hopping(user_response=prompt)

        # Get agent response
        is_api_call = self.current_node.type == NodeType.API_CALL
        is_transfer_call = self.current_node.type == NodeType.TRANSFER_CALL

        response = self._get_response(
            prompt=prompt,
            stream=stream,
            custom_instructions=custom_instructions,
            add_variables=add_variables,
            add_agent_persona=add_agent_persona,
            **client_kwargs,
        )

        if is_api_call:
            if response and self._intermediate_response_callback:
                # self._intermediate_response_callback(response["response"]) # TODO: use streaming here
                pass

            return self._handle_api_call_node(self.current_node)

        elif is_transfer_call:
            if self._transfer_call_callback and self.current_node.transfer_number:
                self._transfer_call_callback(self.current_node.transfer_number)
            else:
                self._log(
                    "No transfer call callback set or transfer number set in the transfer call node"
                )
                raise AgentError(
                    "No transfer call callback set or transfer number set in the transfer call node"
                )

        # Handle End Call and trigger Post-Call API nodes if needed
        elif self.current_node.type == NodeType.END_CALL:
            if not self._is_post_call_complete:
                self._is_post_call_complete = True
                self._process_post_call_api_nodes_async()

        return response

    def _initialize_conversation(
        self,
        initialize_first_message: bool = True,
    ) -> None:
        """
        Set up initial conversation state and generate first message if requested.

        Args:
            initialize_first_message: Whether to generate initial message
        """
        if self.override_response_system_prompt:
            system_prompt = self.override_response_system_prompt
        else:
            system_prompt = FT_RESPONSE_MODEL_SYSTEM_PROMPT

        if self.global_prompt and self.global_prompt.strip():
            system_prompt += f"\n\nSpecial Instructions:\n\n{self.global_prompt}"

        self.response_model_client.add_message("system", system_prompt)

        if initialize_first_message:
            self.initial_message = self.get_response(
                prompt=None,
                stream=False,
                add_variables=True,
                add_agent_persona=True,
                skip_hop=True,
            )["response"]

            self._log(f"Initial message: {self.initial_message}")

    def _get_response(
        self,
        prompt: Optional[str],
        stream: bool = False,
        custom_instructions: Optional[List] = None,
        add_variables: bool = False,
        add_agent_persona: bool = False,
        **client_kwargs,
    ) -> Union[Dict[str, Any], str, Iterator[Dict[str, Any]]]:
        """
        Generate a response based on the current conversation state.

        Args:
            prompt: User input text
            stream: Whether to stream the response
            custom_instructions: Additional instructions for the LLM
            add_variables: Whether to include variables in context
            add_agent_persona: Whether to include agent persona in context
            **client_kwargs: Additional arguments for the LLM client

        Returns:
            Response text or stream iterator
        """
        # Process knowledge base
        # self._process_knowledge_base(prompt)

        # Apply variable substitution
        self.current_node.action = self._replace_variables(self.current_node.action, self.variables)

        if (
            self.current_node.loop_condition
            and isinstance(self.current_node.loop_condition, str)
            and self.current_node.loop_condition.strip()
        ):
            self.current_node.loop_condition = self._replace_variables(
                self.current_node.loop_condition, self.variables
            )

        # Do language switching if turned on
        if self.language_switching and prompt and not prompt.startswith("api output"):
            self.language_now = self._detect_language(prompt)

        custom_instructions = custom_instructions or []
        custom_instructions.append(self._get_language_switch_inst())

        # Get current state of conversation
        current_state = self._format_current_state_as_json(
            user_response=prompt,
            custom_instructions=custom_instructions,
            add_variables=add_variables,
            add_agent_persona=add_agent_persona,
        )
        self._log(f"Current state: {current_state}")

        # Use static text or not
        use_static_text = self.current_node.static_text

        # Delegate to appropriate method based on stream parameter
        if stream:
            return self._get_response_stream(
                current_state=current_state,
                use_static_text=use_static_text,
                **client_kwargs,
            )
        else:
            return self._get_response_non_stream(
                current_state=current_state,
                use_static_text=use_static_text,
                **client_kwargs,
            )

    def _get_response_non_stream(
        self,
        current_state: str,
        use_static_text: bool,
        **client_kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a non-streaming response.

        Args:
            current_state: Formatted current state as JSON
            use_static_text: Whether to use static text
            **client_kwargs: Additional arguments for the LLM client

        Returns:
            Dictionary with response and end_call flag
        """
        end_call = True if self.current_node.type == NodeType.END_CALL else False
        if use_static_text:
            response: str = self.current_node.action
            response = response.strip()

            if self.current_node.type == NodeType.END_CALL:
                end_call = True

            self.response_model_client.add_message("user", current_state)
            self.response_model_client.add_message("assistant", response)

            # Add agent response event
            self.add_event(
                EventType.AGENT_RESPONSE,
                {
                    "response": response,
                    "is_static": True,
                    "is_stream": False,
                    "end_call": end_call,
                },
            )

            return {"response": response, "end_call": end_call}
        else:
            response = self.response_model_client.get_response(
                prompt=current_state, stream=False, **client_kwargs
            )
            if self._end_call_tag in response:
                end_call = True
                response = response.strip().strip(self._end_call_tag).strip()

            # Add agent response event
            self.add_event(
                EventType.AGENT_RESPONSE,
                {
                    "response": response,
                    "is_static": False,
                    "is_stream": False,
                    "end_call": end_call,
                },
            )

            return {"response": response, "end_call": end_call}

    def _get_response_stream(
        self,
        current_state: str,
        use_static_text: bool,
        **client_kwargs,
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate a streaming response.

        Args:
            current_state: Formatted current state as JSON
            use_static_text: Whether to use static text
            **client_kwargs: Additional arguments for the LLM client

        Returns:
            Iterator yielding response chunks
        """
        end_call = True if self.current_node.type == NodeType.END_CALL else False

        if use_static_text:
            response: str = self.current_node.action
            response = response.strip()

            if self.current_node.type == NodeType.END_CALL:
                end_call = True

            self.response_model_client.add_message("user", current_state)
            self.response_model_client.add_message("assistant", response)

            # Add agent response event
            self.add_event(
                EventType.AGENT_RESPONSE,
                {
                    "response": response,
                    "is_static": True,
                    "is_stream": True,
                    "end_call": end_call,
                },
            )

            # Yield static text as a single chunk
            yield {"content": response, "accumulated_text": response}
            if end_call:
                yield {
                    "content": "",
                    "accumulated_text": response,
                    "end_call": end_call,
                }
        else:
            # Get the OpenAI stream
            stream_response = self.response_model_client.get_response(
                prompt=current_state, stream=True, **client_kwargs
            )
            accumulated_text = ""

            # Initialize assistant message in history
            self.response_model_client.add_message("assistant", "")

            # Process and yield chunks as they arrive
            for chunk in stream_response:
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        accumulated_text += content

                        # Update the assistant's message in history with each chunk
                        self.response_model_client.update_last_assistant_message(accumulated_text)

                        yield {"content": content, "accumulated_text": accumulated_text}
                    if (
                        chunk.choices[0].finish_reason
                        and hasattr(chunk.choices[0], "stop_reason")
                        and chunk.choices[0].stop_reason == self._end_call_tag
                    ):
                        end_call = True

            # Final cleanup - ensure the message is properly formatted
            accumulated_text = accumulated_text.strip()
            self.response_model_client.update_last_assistant_message(accumulated_text)

            # Add agent response event after stream completes
            self.add_event(
                EventType.AGENT_RESPONSE,
                {
                    "response": accumulated_text,
                    "is_static": False,
                    "is_stream": True,
                    "end_call": end_call,
                },
            )

            if end_call:
                yield {
                    "content": "",
                    "accumulated_text": accumulated_text,
                    "end_call": end_call,
                }

    def _format_current_state_as_json(
        self,
        user_response: Optional[str],
        custom_instructions: Optional[list] = None,
        add_variables: bool = False,
        add_agent_persona: bool = False,
    ) -> str:
        """
        Convert current state and user response to JSON string.

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
            "action": self.current_node.action,
            "loop_condition": self.current_node.loop_condition,
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

        # If we have a previous state
        if self._last_state is not None and self._last_state.get("id") == current_state["id"]:
            # Calculate delta
            delta = {"user_response": user_response}  # Always include user_response

            # Compare other fields and include only changed ones
            for key, value in current_state.items():
                if key != "user_response" and (
                    key not in self._last_state or self._last_state[key] != value
                ):
                    delta[key] = value

            # Store current state for next comparison
            self._last_state = current_state

            # Return delta
            return json.dumps({"delta": delta}, indent=2, ensure_ascii=False)

        # Store current state for next comparison
        self._last_state = current_state

        # Return full state if no delta or node changed
        return json.dumps(current_state, indent=2, ensure_ascii=False)

    def _hop(self, pathway_id: str, latest_user_response: str = None) -> None:
        """
        Transition to a new node via the specified pathway.

        Args:
            pathway_id: ID of the pathway to follow

        Raises:
            AgentError: If pathway_id not found in current node
        """
        if pathway_id in self.current_node.pathways:
            # Extract variables if leaving a DEFAULT node
            if self.current_node.type == NodeType.DEFAULT:
                self._extract_variables(latest_user_response=latest_user_response)

            prev_node = self.current_node
            self.current_node = self.current_node.pathways[pathway_id].target_node
            self._log(f"Agent is hopping from '{prev_node.name}' to '{self.current_node.name}'.")

            # Add detailed hopping event
            self.add_event(
                EventType.NODE_HOPPING,
                {
                    "selected_pathway_id": pathway_id,
                    "from_node_id": prev_node.id,
                    "from_node_name": prev_node.name,
                    "to_node_id": self.current_node.id,
                    "to_node_name": self.current_node.name,
                },
            )
        else:
            self._log(
                f"Critical Issue: Pathway '{pathway_id}' not found in current node '{self.current_node.name}'"
            )
            raise AgentError(
                f"Pathway '{pathway_id}' not found in current node '{self.current_node.name}'"
            )

    def _handle_hopping(self, user_response: str) -> bool:
        """
        Determine if a node transition is needed and execute if necessary.
        First checks conditional edges, then falls back to LLM-based decision.

        Args:
            user_response: The user's input text

        Returns:
            True if hopping occurred, False otherwise
        """
        # First, check if we have any conditional edges to evaluate
        conditional_edges = []
        for pathway_id, pathway in self.current_node.pathways.items():
            if pathway.is_conditional_edge and pathway.condition:
                conditional_edges.append((pathway_id, pathway))

        # If we have conditional edges, evaluate them first
        if conditional_edges:
            for pathway_id, pathway in conditional_edges:
                if self._evaluate_conditional_edge(pathway.condition):
                    self._log(
                        f"Conditional edge matched: {pathway_id} with condition {pathway.condition}"
                    )
                    self._hop(pathway_id, latest_user_response=user_response)
                    return True

        # If no conditional edge matched (or none existed), continue with normal flow
        flow_history = self._build_flow_navigation_history(user_response)

        if len(flow_history) < 3:
            self._log("Hopping not required yet")
            return False

        selected_pathway_id = self.flow_model_client.get_response(flow_history)
        self._log(f"Selected Pathway ID: {selected_pathway_id}")

        # "null" indicates no hop is needed
        if selected_pathway_id == "null":
            return False

        try:
            self._hop(selected_pathway_id, latest_user_response=user_response)
            return True
        except AgentError:
            return False

    def _build_flow_navigation_history(self, latest_user_response: str) -> List[Dict[str, Any]]:
        """
        Build message history for flow navigation decisions.

        Args:
            latest_user_response: Most recent user input

        Returns:
            List of messages formatted for the flow model
        """
        # Create a copy of messages to avoid modifying the original
        messages = copy.deepcopy(self.response_model_messages)

        # Initialize flow model input with system prompt
        flow_history = [{"role": "system", "content": FT_FLOW_MODEL_SYSTEM_PROMPT}]

        # Find the most recent node message
        current_node_index = None
        for idx, msg in enumerate(reversed(messages[1:])):
            if msg["role"] == "user":
                try:
                    content = json.loads(msg["content"])
                    if "id" in content and content["id"] == self.current_node.id:
                        current_node_index = len(messages) - 1 - idx
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
        flow_history.append(
            {"role": "user", "content": json.dumps(node_data, indent=2, ensure_ascii=False)}
        )

        self._log(f"Flow history for current node: {flow_history}")

        # Process subsequent conversation pairs
        for idx in range(current_node_index + 1, len(messages), 2):
            # Add placeholder assistant response
            flow_history.append({"role": "assistant", "content": "null"})

            # Extract user and assistant responses
            assistant_response = messages[idx]["content"]

            try:
                if idx + 1 < len(messages):
                    user_msg_content = json.loads(messages[idx + 1]["content"])
                    user_response = user_msg_content.get("user_response") or user_msg_content.get(
                        "delta", {}
                    ).get("user_response")
                else:
                    user_response = latest_user_response
            except (json.JSONDecodeError, TypeError):
                user_response = latest_user_response

            # Ensure both responses exist
            if not user_response or not assistant_response:
                break

            # Add conversation pair to flow history
            flow_history.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {"assistant": assistant_response, "user": user_response},
                        indent=2,
                        ensure_ascii=False,
                    ),
                }
            )

        return flow_history

    def _let_llm_decide_pathway(
        self, response_data: Union[Dict[str, Any], str], pathways: List[Tuple[str, Any]]
    ) -> str:
        """
        Let the language model decide which pathway to take based on API response.

        Args:
            response_data: API response data
            pathways: List of pathway options

        Returns:
            Selected pathway ID
        """
        # Format response data for LLM
        formatted_response = (
            f"```\n{json.dumps(response_data, indent=2, ensure_ascii=False)}\n```"
            if isinstance(response_data, dict)
            else f"```\n{response_data}\n```"
        )

        # Create context for flow model
        pathway_options = []
        for pathway_id, pathway in pathways:
            option = {"id": pathway_id, "condition": pathway.condition}
            if pathway.description:
                option["description"] = pathway.description
            pathway_options.append(option)

        prompt = [
            {
                "role": "system",
                "content": FT_FLOW_MODEL_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"API Response:\n{formatted_response}\n\nAvailable Pathways:\n{json.dumps(pathway_options, ensure_ascii=False, indent=2)}\n\nAnalyze the API response and select the most appropriate pathway. Return only the pathway ID. Do not output null - classify the closest pathway that matches the API output even if there isn't a perfect match. A pathway ID is mandatory.",
            },
        ]

        # Get flow model decision
        pathway_id = self.flow_model_client.get_response(prompt)

        # Clean up response to ensure it's just the ID
        pathway_id = pathway_id.strip().strip("\"'")
        self._log(f"Selected Pathway ID in _let_llm_decide_pathway: {pathway_id}")

        return pathway_id

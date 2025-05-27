"""This file contains the context for the atoms agent.

AtomsAgentContext extends OpenAILLMContext and it store the user context in the json format.
format looks something like this:

{
    "transcript": "user_transcript",
    "response_model_context": {
        "id": "node_id",
        "user_response": "user_response"
    }
    "delta": {
        "user_response": "user_response",
    },
    "api_node_response": "api_node_response",
}
"""

import json
from typing import Any, Dict, List, Optional

from loguru import logger
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)

from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
)

from .pathways import Node
from .prompts import (
    FT_FLOW_MODEL_SYSTEM_PROMPT,
)
from .utils import (
    replace_variables,
)


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
        """This function will upgrade the openai context to the atoms agent context."""
        logger.debug(f"Upgrading to Atoms Agent: {obj}")
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, AtomsAgentContext):
            obj.__class__ = AtomsAgentContext
            obj._restructure_from_openai_messages()
        else:
            obj._restructure_from_openai_messages()
        return obj

    @classmethod
    def from_openai_context(cls, openai_context: OpenAILLMContext):
        """This function will create the atoms agent context from the openai context."""
        self = cls(
            messages=openai_context.messages,
            tools=openai_context.tools,
            tool_choice=openai_context.tool_choice,
        )
        self.set_llm_adapter(openai_context.get_llm_adapter())
        self._restructure_from_openai_messages()
        return self

    def get_last_user_context(self) -> Dict[str, Any]:
        """Get the last user context from the messages."""
        if len(self.messages) == 0:
            return ""

        last_message = self.messages[-1]
        if last_message["role"] == "user":
            content = last_message["content"]
            try:
                return json.loads(content)
            except Exception as e:
                raise Exception(f"Error in getting last user context from context messages: {e}")
        else:
            raise Exception(
                f"Last message in the context is not a user message, last_message: {last_message}"
            )

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

    @classmethod
    def upgrade_user_message_to_atoms_agent_message(
        cls, content: ChatCompletionUserMessageParam
    ) -> ChatCompletionUserMessageParam:
        """Update the user message to the atoms agent message."""
        return ChatCompletionUserMessageParam(
            role="user",
            content=json.dumps({"transcript": content["content"]}),
        )

    def add_message(self, message: ChatCompletionMessageParam):
        """Add a message to the context."""
        if message["role"] == "user":
            # if the previous role is "user" aggregate it with the current user message
            # now are using user content in the form of json like this
            # {
            #     "transcript": "user_transcript",
            #     "response_model_context": {
            #         "id": "node_id",
            #         "user_response": "user_response"
            #     }
            # }
            # so we need to check if the previous message is also a user message and if it is then we need to aggregate the content
            if self.messages and self.messages[-1]["role"] == "user":
                previous_user_message_content = json.loads(self.messages[-1]["content"])
                previous_user_message_content["transcript"] = (
                    previous_user_message_content["transcript"] + message["content"]
                )
                self.messages[-1]["content"] = json.dumps(previous_user_message_content)
            else:
                self.messages.append(self.upgrade_user_message_to_atoms_agent_message(message))
        elif message["role"] == "assistant":
            # if the previous role is "assistant" aggregate it with the current assistant message
            if self.messages and self.messages[-1]["role"] == "assistant":
                self.messages[-1]["content"] = self.messages[-1]["content"] + message["content"]
            else:
                self.messages.append(message)
        else:
            self.messages.append(message)

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

            # check if the current user message content already has delta then do not need to generate it again
            # this is the case for api call node where we have already extracted variables and updated the delta
            if "delta" in last_user_message_content and last_user_message_content["delta"]:
                return json.loads(last_user_message_content["delta"])

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

    def get_openai_restructure_messages(self):
        """Get the messages in the openai format."""
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

        return messages

    def _restructure_from_atoms_agent_messages(self):
        """restructure the open ai context from the atoms agent context."""
        messages = self.get_openai_restructure_messages()

        self.messages.clear()
        self.messages.extend(messages)

    def _restructure_from_openai_messages(self):
        """This function will restructure the openai user context messages which are in the default string format and convert them to the atoms agent context messages."""
        messages = []

        # for message in self.messages:
        #     if message["role"] == "user":
        #         try:
        #             json.loads(message["content"])
        #             # if the content is json than it is already in the atoms agent format no need to convert it
        #             messages.append(message)
        #         except json.JSONDecodeError:
        #             # if json conversion fails than it is a user message and we need to convert it to the atoms agent format
        #             messages.append(
        #                 ChatCompletionUserMessageParam(
        #                     role="user", content=json.dumps({"transcript": message["content"]})
        #                 )
        #             )
        #         except Exception as e:
        #             logger.error(f"Error parsing user message: {message}")
        #             messages.append(message)
        #     else:
        #         messages.append(message)

        # self.messages.clear()
        # self.messages.extend(messages)

        # NOTE: We have commented out the above code because we are using the atomsAgentContext when creating pipeline and we are converting the messages to the atoms agent context format when adding messages to the context
        pass

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
                                ChatCompletionUserMessageParam(
                                    role="user",
                                    content=json.dumps(
                                        {
                                            "delta": json.loads(delta),
                                        },
                                        indent=2,
                                        ensure_ascii=False,
                                    ),
                                )
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

    def get_api_node_flow_navigation_model_context(self, current_node: Node):
        """Get the flow navigation model context for the API node."""
        last_user_context = self.get_last_user_context()
        api_response = last_user_context["api_node_response"]
        prompt = [
            {
                "role": "system",
                "content": FT_FLOW_MODEL_SYSTEM_PROMPT,
            },
        ]

        if not api_response:
            return prompt

        formatted_response = f"```\n{api_response}\n```"

        # Create context for flow model
        pathway_options = []
        for pathway_id, pathway in current_node.pathways.items():
            if not pathway.is_conditional_edge:
                option = {"id": pathway_id, "condition": pathway.condition}
                if pathway.description:
                    option["description"] = pathway.description
            pathway_options.append(option)

        prompt.append(
            {
                "role": "user",
                "content": f"API Response:\n{formatted_response}\n\nAvailable Pathways:\n{json.dumps(pathway_options, ensure_ascii=False, indent=2)}\n\nAnalyze the API response and select the most appropriate pathway. Return only the pathway ID. Do not output null - classify the closest pathway that matches the API output even if there isn't a perfect match. A pathway ID is mandatory.",
            },
        )

        return prompt

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
        for idx, msg in enumerate(self.messages[1:]):
            if msg["role"] == "user":
                try:
                    content = json.loads(msg["content"])
                    if "response_model_context" in content:
                        repsonse_model_context = json.loads(content["response_model_context"])
                        if (
                            "id" in repsonse_model_context
                            and repsonse_model_context["id"] == current_node.id
                        ):
                            current_node_index = idx + 1
                            break
                except (json.JSONDecodeError, TypeError):
                    pass
        else:
            return []

        # Build current node representation with specific order
        node_data = {
            "name": current_node.name,
            "type": current_node.type.name,
            "action": replace_variables(current_node.action, variables),
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

                    user_content_deserialized = json.loads(user_content)
                    transcript = user_content_deserialized["transcript"]

                    flow_navigation_history += [
                        {
                            "role": "assistant",
                            "content": "null",
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {"assistant": assistant_content, "user": transcript},
                                indent=2,
                                ensure_ascii=False,
                            ),
                        },
                    ]

            current_node_index += 1

        return flow_navigation_history

    def _get_variable_extraction_messages(self):
        """Get the messages for the variable extraction."""
        messages = []
        for message in self.messages[2:]:
            if message["role"] == "user":
                messages.append(f"User: {message['content']}")
            else:
                messages.append(f"Assistant: {message['content']}")
        return messages

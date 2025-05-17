import time
from typing import Any, Callable, Dict, List, Optional, Union

import openai


class BaseClient:
    """Base class for all client classes."""

    def __init__(
        self,
        model_id: str,
        keep_history: bool = True,
        api_key: Optional[str] = None,
        default_response_kwargs: Optional[Dict[str, Any]] = None,
        prepare_messages_callback: Optional[Callable[[List], List]] = None,
        call_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the LLM client.

        Args:
            model_id (str): The identifier for the language model to use.
            keep_history (bool, optional): Whether to maintain conversation history. Defaults to True.
            api_key (str, optional): API key for authentication. Defaults to None.
            default_response_kwargs (Dict[str, Any], optional): Default parameters to pass to model response generation. Defaults to None.
            prepare_messages_callback (Callable[[List], List], optional): Function to preprocess messages before sending to model. Defaults to None.
            **kwargs: Additional keyword arguments passed to the client initialization.
        """
        self.model_id = model_id
        self.keep_history = keep_history
        self.default_response_kwargs = default_response_kwargs or {}
        self.messages = []
        self.call_id = call_id
        self._initialize_client(api_key, **kwargs)

    def _initialize_client(self, api_key: str, **kwargs):
        """Initialize the client. This method should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize_client")

    def _prepare_messages_for_api(self, messages: List[Dict[str, Any]]):
        """Prepares messages for API request by formatting them into the required structure.

        Args:
            messages: List of message dictionaries containing 'role' and 'content' keys.

        Returns:
            List of formatted message dictionaries with only 'role' and 'content' keys.
            If prepare_messages_callback is set, returns the result of that callback instead.
        """
        return messages

    def _make_api_call(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Make the API call to the model. This method should be implemented by subclasses.

        Args:
            messages (List[Dict[str, Any]]): The messages to send to the model.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            str: The response from the model.
        """
        raise NotImplementedError("Subclasses must implement _make_api_call")

    def get_response(self, prompt: Union[str, List[Dict[str, Any]]], **kwargs):
        """Get a response from the client.

        Args:
            prompt (str): The prompt to send to the client.
            **kwargs: Additional keyword arguments to pass to the client.

        Returns:
            str: The response from the client.
        """
        if isinstance(prompt, list) and self.keep_history:
            raise ValueError("keep_history must be False when passing a list of messages")

        if self.keep_history:
            self.add_message(role="user", content=prompt)
            messages = self._prepare_messages_for_api(self.messages)
        else:
            if isinstance(prompt, list):
                messages = self._prepare_messages_for_api(prompt[:])
            else:
                messages = [{"role": "user", "content": prompt}]

        output = self._make_api_call(messages, **kwargs)

        if self.keep_history and isinstance(output, str):
            self.add_message(role="assistant", content=output)

        return output

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history.

        Args:
            role (str): The role of the message (e.g., "user" or "assistant").
            content (str): The content of the message.

        Returns:
            None
        """
        self.messages.append({"role": role, "content": content})

    def clear_history(self):
        """Clear the conversation history.

        Returns:
            None
        """
        self.messages = []

    def update_message_content(self, index: int, content: str):
        """Update the content of a message in the conversation history.

        Args:
            index (int): The index of the message to update.
            content (str): The new content for the message.

        Returns:
            None
        """
        self.messages[index]["content"] = content

    def update_last_assistant_message(self, new_content: str):
        """Update the content of the last assistant message in the history."""
        if self.messages and self.messages[-1]["role"] == "assistant":
            self.messages[-1]["content"] = new_content
        else:
            # If last message is not assistant, we do not update or add new message
            pass


class AzureOpenAIClient(BaseClient):
    """A client for interacting with the Azure OpenAI API."""

    def __init__(
        self,
        model_id: str,
        api_version: str,
        azure_endpoint: str,
        api_key: str,
        keep_history: bool = True,
        default_response_kwargs: Optional[Dict[str, Any]] = None,
        prepare_messages_callback: Optional[Callable[[List], List]] = None,
        call_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initialize the OpenAIClient with a specified model and history settings."""
        super().__init__(
            model_id=model_id,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            keep_history=keep_history,
            api_key=api_key,
            default_response_kwargs=default_response_kwargs,
            prepare_messages_callback=prepare_messages_callback,
            call_id=call_id,
            **kwargs,
        )

    def _initialize_client(
        self,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
        **kwargs,
    ) -> None:
        """Initialize the OpenAI client with the API key."""
        self.client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            **kwargs,
        )

    def _make_api_call(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> str:
        """Get a response from the OpenAI model."""
        request_kwargs = self.default_response_kwargs.copy()
        request_kwargs.update(kwargs)

        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            **request_kwargs,
        )
        if request_kwargs.get("stream", False):
            return completion
        return completion.choices[0].message.content


class OpenAIClient(BaseClient):
    """
    A client for interacting with the OpenAI API.
    """

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        keep_history: bool = True,
        api_key: Optional[str] = None,
        default_response_kwargs: Optional[Dict[str, Any]] = None,
        prepare_messages_callback: Optional[Callable[[List], List]] = None,
        call_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the OpenAIClient with a specified model and history settings.
        """
        super().__init__(
            model_id=model_id,
            base_url=base_url,
            keep_history=keep_history,
            api_key=api_key,
            default_response_kwargs=default_response_kwargs,
            prepare_messages_callback=prepare_messages_callback,
            call_id=call_id,
            **kwargs,
        )

    def _initialize_client(self, api_key: str, base_url: Optional[str] = None, **kwargs) -> None:
        """
        Initialize the OpenAI client with the API key.
        """
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, **kwargs)

    def _make_api_call(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> str:
        """
        Make the API call to the OpenAI model.
        """
        request_kwargs = self.default_response_kwargs.copy()
        request_kwargs.update(kwargs)

        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            **request_kwargs,
        )
        if request_kwargs.get("stream", False):
            return completion
        return completion.choices[0].message.content

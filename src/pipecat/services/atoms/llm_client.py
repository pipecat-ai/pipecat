import time
from typing import Any, Callable, Dict, List, Optional, Union

import openai


class BaseClient:
    """Base class for all client classes."""

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        default_response_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize the LLM client.

        Args:
            model_id (str): The identifier for the language model to use.
            api_key (str, optional): API key for authentication. Defaults to None.
            default_response_kwargs (Dict[str, Any], optional): Default parameters to pass to model response generation. Defaults to None.
            prepare_messages_callback (Callable[[List], List], optional): Function to preprocess messages before sending to model. Defaults to None.
            **kwargs: Additional keyword arguments passed to the client initialization.
        """
        self.model_id = model_id
        self.default_response_kwargs = default_response_kwargs or {}
        self._initialize_client(api_key, **kwargs)

    def _initialize_client(self, api_key: str, **kwargs):
        """Initialize the client. This method should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _initialize_client")

    async def _make_api_call(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Make the API call to the model. This method should be implemented by subclasses.

        Args:
            messages (List[Dict[str, Any]]): The messages to send to the model.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            str: The response from the model.
        """
        raise NotImplementedError("Subclasses must implement _make_api_call")

    async def get_response(self, messages: List[Dict[str, Any]], **kwargs):
        """Get a response from the client.

        Args:
            messages (List[Dict[str, Any]]): The messages to send to the client.
            **kwargs: Additional keyword arguments to pass to the client.

        Returns:
            str: The response from the client.
        """
        output = await self._make_api_call(messages=messages, **kwargs)
        return output


class AzureOpenAIClient(BaseClient):
    """A client for interacting with the Azure OpenAI API."""

    def __init__(
        self,
        model_id: str,
        api_version: str,
        azure_endpoint: str,
        api_key: str,
        default_response_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize the OpenAIClient with a specified model and history settings."""
        super().__init__(
            model_id=model_id,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=api_key,
            default_response_kwargs=default_response_kwargs,
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
        self.client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            **kwargs,
        )

    async def _make_api_call(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> str:
        """Get a response from the OpenAI model."""
        request_kwargs = self.default_response_kwargs.copy()
        request_kwargs.update(kwargs)

        completion = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            **request_kwargs,
        )
        if request_kwargs.get("stream", False):
            return completion
        return completion.choices[0].message.content


class OpenAIClient(BaseClient):
    """A client for interacting with the OpenAI API."""

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_response_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Initialize the OpenAIClient with a specified model and history settings."""
        super().__init__(
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            default_response_kwargs=default_response_kwargs,
            **kwargs,
        )

    def _initialize_client(self, api_key: str, base_url: Optional[str] = None, **kwargs) -> None:
        """Initialize the OpenAI client with the API key."""
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url, **kwargs)

    async def _make_api_call(
        self,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> str:
        """Make the API call to the OpenAI model."""
        request_kwargs = self.default_response_kwargs.copy()
        request_kwargs.update(kwargs)

        completion = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            **request_kwargs,
        )
        if request_kwargs.get("stream", False):
            return completion
        return completion.choices[0].message.content

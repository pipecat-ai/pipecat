#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat services for a Deepgram Voice Agent API ``Settings`` message.

A Voice Agent client chooses its providers and models in ``Settings``
(``agent.listen``, ``agent.think``, ``agent.speak``). The functions here build
the matching Pipecat services, and the function handlers for server-side
(``endpoint``) functions.
"""

import asyncio
import ipaddress
import json
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import aiohttp
from loguru import logger

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    Frame,
    FunctionCallFromLLM,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

DEFAULT_LISTEN_MODEL = "nova-3"
DEFAULT_SPEAK_MODEL = "aura-2-thalia-en"
ENDPOINT_FUNCTION_TIMEOUT_SECS = 10.0


class SettingsError(Exception):
    """A ``Settings`` message this server can't honor.

    Parameters:
        code: The Voice Agent error code to report.
        description: A human-readable description.
    """

    def __init__(self, description: str, code: str = "INVALID_SETTINGS"):
        """Initialize the error.

        Args:
            description: A human-readable description.
            code: The Voice Agent error code to report.
        """
        super().__init__(description)
        self.code = code
        self.description = description


@dataclass
class ProviderCredentials:
    """Credentials for the providers a session may use.

    Parameters:
        deepgram_api_key: Deepgram API key, for listen and speak.
        openai_api_key: OpenAI API key, for ``think.provider.type == "open_ai"``.
        openai_base_url: OpenAI API base URL, e.g. for region-scoped keys.
        anthropic_api_key: Anthropic API key, for ``"anthropic"``.
    """

    deepgram_api_key: str
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None


def _provider(stage: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return (stage or {}).get("provider") or {}


def create_stt(
    listen: Mapping[str, Any] | None, *, credentials: ProviderCredentials, tag: str | None = None
) -> STTService:
    """Create the STT service for ``agent.listen``.

    Flux models (``flux-*``) get :class:`DeepgramFluxSTTService`, which does its
    own turn detection; other Deepgram models get :class:`DeepgramSTTService`.

    Args:
        listen: The ``agent.listen`` settings.
        credentials: Provider credentials.
        tag: Usage tag for the Deepgram requests.

    Returns:
        The STT service.

    Raises:
        SettingsError: If the provider isn't supported.
    """
    provider = _provider(listen)
    provider_type = provider.get("type", "deepgram")
    if provider_type != "deepgram":
        raise SettingsError(f"Unsupported listen provider: {provider_type}")
    model = provider.get("model") or DEFAULT_LISTEN_MODEL
    keyterms = provider.get("keyterms") or None
    tags = [tag] if tag else None

    if model.startswith("flux"):
        from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService

        return DeepgramFluxSTTService(
            api_key=credentials.deepgram_api_key,
            tag=tags,
            settings=DeepgramFluxSTTService.Settings(
                model=model,
                eot_threshold=provider.get("eot_threshold"),
                eager_eot_threshold=provider.get("eager_eot_threshold"),
                eot_timeout_ms=provider.get("eot_timeout_ms"),
                keyterm=keyterms or [],
            ),
        )

    from pipecat.services.deepgram.stt import DeepgramSTTService

    return DeepgramSTTService(
        api_key=credentials.deepgram_api_key,
        tag=tags,
        settings=DeepgramSTTService.Settings(
            model=model, smart_format=True, interim_results=True, keyterm=keyterms
        ),
    )


def uses_external_turn_detection(stt: STTService) -> bool:
    """Whether the STT service detects turns itself (Flux does).

    Args:
        stt: The STT service.

    Returns:
        Whether it does.
    """
    from pipecat.services.deepgram.flux.stt_base import DeepgramFluxSTTBase

    return isinstance(stt, DeepgramFluxSTTBase)


async def create_llm(
    think: Mapping[str, Any] | None,
    *,
    credentials: ProviderCredentials,
    http_session: aiohttp.ClientSession,
    system_instruction: str | None = None,
) -> LLMService[Any]:
    """Create the LLM service for ``agent.think``.

    Args:
        think: The ``agent.think`` settings.
        system_instruction: The system prompt.
        credentials: Provider credentials.
        http_session: HTTP session, used to check that the model exists.

    Returns:
        The LLM service.

    Raises:
        SettingsError: If the provider isn't supported, has no credentials, or
            the model doesn't exist.
    """
    provider = _provider(think)
    provider_type = provider.get("type")
    model = provider.get("model")

    if provider_type == "mock":
        return DeepgramMockLLMService(reply_chunks=provider.get("reply_chunks") or [])

    if provider_type == "open_ai":
        if not credentials.openai_api_key:
            raise SettingsError("The open_ai think provider is not configured on this server.")
        await _check_openai_model(model, credentials, http_session)
        from pipecat.services.openai.llm import OpenAILLMService

        return OpenAILLMService(
            api_key=credentials.openai_api_key,
            base_url=credentials.openai_base_url,
            settings=OpenAILLMService.Settings(
                model=model,
                system_instruction=system_instruction,
                temperature=provider.get("temperature"),
            ),
        )

    if provider_type == "anthropic":
        if not credentials.anthropic_api_key:
            raise SettingsError("The anthropic think provider is not configured on this server.")
        from pipecat.services.anthropic.llm import AnthropicLLMService

        return AnthropicLLMService(
            api_key=credentials.anthropic_api_key,
            settings=AnthropicLLMService.Settings(
                model=model,
                system_instruction=system_instruction,
                temperature=provider.get("temperature"),
            ),
        )

    raise SettingsError(f"Unsupported think provider: {provider_type}")


async def _check_openai_model(
    model: str | None, credentials: ProviderCredentials, http_session: aiohttp.ClientSession
):
    if not model:
        raise SettingsError("agent.think.provider.model is required.")
    base_url = (credentials.openai_base_url or "https://api.openai.com/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {credentials.openai_api_key}"}
    try:
        async with http_session.get(
            f"{base_url}/models/{model}",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as response:
            if response.status == 404:
                raise SettingsError(f"Unknown think model: {model}")
    except (TimeoutError, aiohttp.ClientError) as e:
        # Don't fail the session because the check itself failed.
        logger.warning(f"Could not check OpenAI model {model}: {e}")


def create_tts(
    speak: Any,
    *,
    credentials: ProviderCredentials,
    http_session: aiohttp.ClientSession,
    sample_rate: int,
) -> TTSService:
    """Create the TTS service for ``agent.speak``.

    ``agent.speak`` may list fallback providers; the first is used.

    Args:
        speak: The ``agent.speak`` settings, an object or a list of them.
        credentials: Provider credentials.
        http_session: HTTP session for the TTS requests.
        sample_rate: Sample rate to synthesize at.

    Returns:
        The TTS service.

    Raises:
        SettingsError: If the provider isn't supported.
    """
    if isinstance(speak, list):
        speak = speak[0] if speak else None
    provider = _provider(speak)
    provider_type = provider.get("type", "deepgram")
    if provider_type != "deepgram":
        raise SettingsError(f"Unsupported speak provider: {provider_type}")

    from pipecat.services.deepgram.tts import DeepgramHttpTTSService

    # One streamed HTTP request per utterance: Voice Agent sessions idle for long
    # stretches, and an idle TTS WebSocket gets closed.
    return DeepgramHttpTTSService(
        api_key=credentials.deepgram_api_key,
        aiohttp_session=http_session,
        sample_rate=sample_rate,
        settings=DeepgramHttpTTSService.Settings(
            voice=provider.get("model") or DEFAULT_SPEAK_MODEL
        ),
    )


def function_schemas(functions: list[Mapping[str, Any]]) -> ToolsSchema:
    """Convert ``agent.think.functions`` into a tools schema.

    Args:
        functions: The ``agent.think.functions`` list.

    Returns:
        A tools schema with a function schema for each function.
    """
    schemas = []
    for function in functions:
        parameters = function.get("parameters") or {}
        schemas.append(
            FunctionSchema(
                name=function["name"],
                description=function.get("description", ""),
                properties=parameters.get("properties", {}),
                required=parameters.get("required", []),
            )
        )
    return ToolsSchema(standard_tools=schemas)


async def _check_public_endpoint(url: str, *, allow_nonpublic: bool):
    parsed = urlparse(url)
    if allow_nonpublic:
        return
    if parsed.scheme != "https":
        raise ValueError(f"Function endpoints must use https: {url}")
    host = parsed.hostname or ""
    infos = await asyncio.get_running_loop().getaddrinfo(host, parsed.port or 443)
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise ValueError(f"Function endpoint is not publicly routable: {url}")


async def check_endpoint_functions(
    functions: list[Mapping[str, Any]], *, allow_nonpublic: bool = False
):
    """Check that the server-side functions' endpoints are allowed.

    Endpoints must be ``https`` URLs that resolve only to public addresses,
    unless ``allow_nonpublic`` is set, so a client can't make the server
    request its own network.

    Args:
        functions: The ``agent.think.functions`` list.
        allow_nonpublic: Whether to allow any URL.

    Raises:
        SettingsError: If an endpoint isn't allowed.
    """
    for function in functions:
        endpoint = function.get("endpoint")
        if not endpoint:
            continue
        try:
            await _check_public_endpoint(endpoint.get("url", ""), allow_nonpublic=allow_nonpublic)
        except (ValueError, OSError) as e:
            raise SettingsError(f"Invalid endpoint for function {function['name']}: {e}")


def endpoint_function_handler(
    endpoint: Mapping[str, Any],
    *,
    http_session: aiohttp.ClientSession,
    send: Callable[[dict[str, Any]], Awaitable[None]],
    allow_nonpublic: bool = False,
    timeout_secs: float = ENDPOINT_FUNCTION_TIMEOUT_SECS,
):
    """Create a handler for a server-side (``endpoint``) function.

    The handler announces the call to the client (``FunctionCallRequest`` with
    ``client_side: false``), requests the endpoint with the arguments as the
    JSON body, and reports the response body as the result, which it also
    sends to the client as a ``FunctionCallResponse``. The endpoint's address
    is checked again before each request.

    Args:
        endpoint: The function's ``endpoint`` (``url``, ``method``, ``headers``).
        http_session: HTTP session for the requests.
        send: Sends a message to the client.
        allow_nonpublic: Whether to allow non-public endpoints.
        timeout_secs: Timeout for the request.

    Returns:
        The function handler.
    """
    url = endpoint["url"]
    method = (endpoint.get("method") or "post").upper()
    headers = dict(endpoint.get("headers") or {})

    async def handler(params: FunctionCallParams):
        await send(
            {
                "type": "FunctionCallRequest",
                "functions": [
                    {
                        "id": params.tool_call_id,
                        "name": params.function_name,
                        "arguments": json.dumps(dict(params.arguments)),
                        "client_side": False,
                    }
                ],
            }
        )
        try:
            await _check_public_endpoint(url, allow_nonpublic=allow_nonpublic)
            async with http_session.request(
                method,
                url,
                headers=headers,
                json=dict(params.arguments) or None,
                timeout=aiohttp.ClientTimeout(total=timeout_secs),
            ) as response:
                body = await response.text()
                result = body if response.ok else f"Error {response.status}: {body}"
        except Exception as e:
            result = f"The function call failed: {e}"
        await send(
            {
                "type": "FunctionCallResponse",
                "id": params.tool_call_id,
                "name": params.function_name,
                "content": result,
            }
        )
        await params.result_callback(result)

    return handler


class DeepgramMockLLMService(LLMService):
    """Scripted LLM matching the Voice Agent API's ``mock`` think provider.

    Every inference replays ``reply_chunks`` in order: each waits ``delay_ms``,
    then contributes its ``content`` as text or its ``function_call``. An
    inference that answers a function result skips the function calls, so a
    scripted call happens once per turn.
    """

    def __init__(self, *, reply_chunks: list[Mapping[str, Any]], **kwargs):
        """Initialize the service.

        Args:
            reply_chunks: The chunks to reply with.
            **kwargs: Additional arguments passed to :class:`LLMService`.
        """
        settings = LLMSettings(
            model="mock",
            system_instruction=None,
            temperature=None,
            max_tokens=None,
            top_p=None,
            top_k=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=None,
            user_turn_completion_config=None,
        )
        super().__init__(settings=settings, **kwargs)
        self._reply_chunks = reply_chunks

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Reply to each context frame; pass everything else on.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            await self._reply(frame.context)
        else:
            await self.push_frame(frame, direction)

    async def _reply(self, context: LLMContext):
        messages = context.messages
        last = messages[-1] if messages else None
        answering_function_result = isinstance(last, dict) and last.get("role") == "tool"
        function_calls = []
        await self.push_frame(LLMFullResponseStartFrame())
        for chunk in self._reply_chunks:
            await asyncio.sleep(chunk.get("delay_ms", 0) / 1000)
            function_call = chunk.get("function_call")
            if function_call:
                if answering_function_result:
                    continue
                arguments = function_call.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments or "{}")
                function_calls.append(
                    FunctionCallFromLLM(
                        function_name=function_call["name"],
                        tool_call_id=f"call_{uuid.uuid4().hex}",
                        arguments=arguments,
                        context=context,
                    )
                )
            elif chunk.get("content"):
                await self.push_frame(LLMTextFrame(chunk["content"]))
        await self.push_frame(LLMFullResponseEndFrame())
        if function_calls:
            await self.run_function_calls(function_calls)


__all__ = [
    "DeepgramMockLLMService",
    "ProviderCredentials",
    "SettingsError",
    "check_endpoint_functions",
    "create_llm",
    "create_stt",
    "create_tts",
    "endpoint_function_handler",
    "function_schemas",
    "uses_external_turn_detection",
]

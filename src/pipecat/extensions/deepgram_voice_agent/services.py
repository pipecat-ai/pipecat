#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Pipecat services for a Deepgram Voice Agent API ``Settings`` message.

A Voice Agent client chooses its providers and models in ``Settings``
(``agent.listen``, ``agent.think``, ``agent.speak``). The functions here build
the think and speak services, and the function handlers for server-side
(``endpoint``) functions; :mod:`.listen` builds the listen services.
"""

import asyncio
import ipaddress
import json
import re
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
from pipecat.services.tts_service import TTSService
from pipecat.utils.text.base_text_filter import BaseTextFilter

DEFAULT_SPEAK_MODEL = "aura-2-thalia-en"
ENDPOINT_FUNCTION_TIMEOUT_SECS = 10.0
_STAGE_DIRECTION = re.compile(r"\*[^*\n]+\*")


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
        cartesia_api_key: Cartesia API key, for a ``"cartesia"`` speak provider
            whose endpoint carries none.
        elevenlabs_api_key: ElevenLabs API key, likewise for ``"eleven_labs"``.
        google_api_key: Google API key, for ``think.provider.type == "google"``.
    """

    deepgram_api_key: str
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None
    cartesia_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    google_api_key: str | None = None


def _provider(stage: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return (stage or {}).get("provider") or {}


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

    if provider_type == "google":
        if not credentials.google_api_key:
            raise SettingsError("The google think provider is not configured on this server.")
        from pipecat.services.google.llm import GoogleLLMService

        return GoogleLLMService(
            api_key=credentials.google_api_key,
            settings=GoogleLLMService.Settings(
                model=model,
                system_instruction=system_instruction,
                temperature=provider.get("temperature"),
            ),
        )

    raise SettingsError(f"Unsupported think provider: {provider_type}")


async def validate_think_update(
    think: Mapping[str, Any] | None,
    *,
    credentials: ProviderCredentials,
    http_session: aiohttp.ClientSession,
):
    """Check an ``UpdateThink`` before applying it.

    Args:
        think: The requested ``think`` settings.
        credentials: Provider credentials.
        http_session: HTTP session, used to check that the model exists.

    Raises:
        SettingsError: If the provider isn't supported or configured, or the
            model doesn't exist.
    """
    provider = _provider(think)
    provider_type = provider.get("type")
    if provider_type is None or provider_type == "mock":
        return
    if provider_type == "open_ai":
        if not credentials.openai_api_key:
            raise SettingsError("The open_ai think provider is not configured on this server.")
        await _check_openai_model(provider.get("model"), credentials, http_session)
    elif provider_type == "anthropic":
        if not credentials.anthropic_api_key:
            raise SettingsError("The anthropic think provider is not configured on this server.")
    elif provider_type == "google":
        if not credentials.google_api_key:
            raise SettingsError("The google think provider is not configured on this server.")
    else:
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


def _first_speak(speak: Any) -> Mapping[str, Any]:
    if isinstance(speak, list):
        speak = speak[0] if speak else None
    return speak or {}


def _endpoint(speak: Mapping[str, Any]) -> tuple[str | None, dict[str, str]]:
    """The speak endpoint's URL and headers, checking the URL is one we can use."""
    endpoint = speak.get("endpoint") or {}
    url = endpoint.get("url")
    headers = {str(k).lower(): str(v) for k, v in (endpoint.get("headers") or {}).items()}
    if url is not None:
        scheme = urlparse(url).scheme
        if scheme not in ("https", "wss"):
            raise SettingsError(f"Speak endpoints must use https or wss: {url}")
    return url, headers


def speak_provider_type(speak: Any) -> str:
    """The provider type of ``agent.speak`` (or an ``UpdateSpeak``).

    Args:
        speak: The speak settings, an object or a list of them.

    Returns:
        The provider type, ``deepgram`` when none is given.
    """
    return _provider(_first_speak(speak)).get("type", "deepgram")


def create_tts(
    speak: Any,
    *,
    credentials: ProviderCredentials,
    http_session: aiohttp.ClientSession,
    sample_rate: int,
) -> TTSService:
    """Create the TTS service for ``agent.speak``.

    ``agent.speak`` may list fallback providers; the first is used. Supported
    providers are ``deepgram``, ``cartesia`` and ``eleven_labs``. A
    third-party provider's API key comes from its ``endpoint`` headers, as in
    the Voice Agent API, or else from the server's credentials.

    Args:
        speak: The ``agent.speak`` settings, an object or a list of them.
        credentials: Provider credentials.
        http_session: HTTP session for the TTS requests.
        sample_rate: Sample rate to synthesize at.

    Returns:
        The TTS service.

    Raises:
        SettingsError: If the settings aren't supported.
    """
    speak = _first_speak(speak)
    provider_type = speak_provider_type(speak)
    filters: list[BaseTextFilter] = [StageDirectionFilter()]
    if speak.get("additional_delay_ms_connect"):
        filters.insert(0, ConnectDelayFilter(speak["additional_delay_ms_connect"] / 1000))

    if provider_type == "deepgram":
        # Flux or Aura settings, by voice.
        settings: Any = speak_settings(speak, default_voice=DEFAULT_SPEAK_MODEL)
        if is_flux_voice(settings.voice):
            from pipecat.services.deepgram.flux.tts import DeepgramFluxTTSService

            return DeepgramFluxTTSService(
                api_key=credentials.deepgram_api_key,
                sample_rate=sample_rate,
                settings=settings,
                text_filters=filters,
            )

        from pipecat.services.deepgram.tts import DeepgramHttpTTSService

        # One streamed HTTP request per utterance: Voice Agent sessions idle for
        # long stretches, and an idle TTS WebSocket gets closed.
        return DeepgramHttpTTSService(
            api_key=credentials.deepgram_api_key,
            aiohttp_session=http_session,
            sample_rate=sample_rate,
            settings=settings,
            text_filters=filters,
        )

    if provider_type == "cartesia":
        return _create_cartesia_tts(speak, credentials, http_session, sample_rate, filters)

    if provider_type == "eleven_labs":
        return _create_eleven_labs_tts(speak, credentials, sample_rate, filters)

    raise SettingsError(f"Unsupported speak provider: {provider_type}")


def _cartesia_extra(provider: Mapping[str, Any]) -> dict[str, Any]:
    """Cartesia request fields for ``speed`` and ``volume``.

    sonic-3 and later take both in ``generation_config``, as a multiplier
    (presets map onto it); older models take ``speed`` at the top level, as a
    preset or -1 to 1.
    """
    model = provider.get("model_id") or provider.get("model") or ""
    legacy = model.startswith(("sonic-2", "sonic-english", "sonic-multilingual", "sonic-turbo"))
    extra: dict[str, Any] = {}
    generation_config: dict[str, Any] = {}
    speed = provider.get("speed")
    if speed is not None:
        if legacy and _valid_legacy_cartesia_speed(speed):
            extra["speed"] = speed
        else:
            # generation_config takes a multiplier; presets map onto its range.
            # Cartesia ignores an invalid top-level speed but rejects an invalid
            # generation_config one, so an invalid value fails the request.
            generation_config["speed"] = _CARTESIA_SPEED_PRESETS.get(speed, speed)
    if provider.get("volume") is not None:
        generation_config["volume"] = provider["volume"]
    if generation_config:
        extra["generation_config"] = generation_config
    return extra


_CARTESIA_SPEED_PRESETS = {"slowest": 0.6, "slow": 0.8, "normal": 1.0, "fast": 1.2, "fastest": 1.5}


def _valid_legacy_cartesia_speed(speed: Any) -> bool:
    """Whether a speed is valid on the legacy scale: a preset, or -1 to 1."""
    if isinstance(speed, str) and speed in _CARTESIA_SPEED_PRESETS:
        return True
    try:
        return -1.0 <= float(speed) <= 1.0
    except (TypeError, ValueError):
        return False


def _create_cartesia_tts(speak, credentials, http_session, sample_rate, filters) -> TTSService:
    from pipecat.services.cartesia.tts import CartesiaHttpTTSService, CartesiaTTSService

    provider = _provider(speak)
    url, headers = _endpoint(speak)
    api_key = headers.get("x-api-key") or credentials.cartesia_api_key
    if not api_key:
        raise SettingsError("The cartesia speak provider needs an X-API-Key endpoint header.")
    voice = (provider.get("voice") or {}).get("id")
    model = provider.get("model_id") or provider.get("model")
    language = provider.get("language")
    if url and urlparse(url).scheme == "wss":
        if provider.get("speed") is not None:
            raise SettingsError("Cartesia speed is only supported on its batch (https) endpoint.")
        return CartesiaTTSService(
            api_key=api_key,
            url=url,
            sample_rate=sample_rate,
            settings=CartesiaTTSService.Settings(model=model, voice=voice, language=language),
            text_filters=filters,
        )
    base_url = "https://api.cartesia.ai"
    if url:
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
    return CartesiaHttpTTSService(
        api_key=api_key,
        base_url=base_url,
        aiohttp_session=http_session,
        sample_rate=sample_rate,
        settings=CartesiaHttpTTSService.Settings(
            model=model, voice=voice, language=language, extra=_cartesia_extra(provider)
        ),
        text_filters=filters,
    )


def eleven_labs_voice(url: str | None) -> str:
    """The voice in an ElevenLabs ``multi-stream-input`` endpoint URL.

    Args:
        url: The endpoint URL.

    Returns:
        The voice ID.

    Raises:
        SettingsError: If the URL isn't a ``multi-stream-input`` endpoint.
    """
    path = [part for part in urlparse(url or "").path.split("/") if part]
    if len(path) < 2 or path[-1] != "multi-stream-input":
        raise SettingsError(
            "ElevenLabs speak endpoints must end with /multi-stream-input, which supports barge-in."
        )
    return path[-2]


def _create_eleven_labs_tts(speak, credentials, sample_rate, filters) -> TTSService:
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

    provider = _provider(speak)
    url, headers = _endpoint(speak)
    api_key = headers.get("xi-api-key") or credentials.elevenlabs_api_key
    if not api_key:
        raise SettingsError("The eleven_labs speak provider needs an xi-api-key endpoint header.")
    voice = eleven_labs_voice(url)
    parsed = urlparse(url or "")
    fields: dict[str, Any] = {"model": provider.get("model_id"), "voice": voice}
    if provider.get("language_code"):
        fields["language"] = str(provider["language_code"]).split("-")[0]
    if provider.get("speed") is not None:
        fields["speed"] = float(provider["speed"])
    return ElevenLabsTTSService(
        api_key=api_key,
        url=f"{parsed.scheme}://{parsed.netloc}",
        sample_rate=sample_rate,
        settings=ElevenLabsTTSService.Settings(**fields),
        text_filters=filters,
    )


def speak_settings(speak: Any, *, default_voice: str | None = None):
    """Check Deepgram speak settings and build the TTS settings for them.

    Used for ``agent.speak`` with the ``deepgram`` provider, and for an
    ``UpdateSpeak`` to it. A custom Deepgram ``endpoint`` isn't supported.

    Args:
        speak: The speak settings, an object or a list of them.
        default_voice: Voice to use when none is given; None leaves it unset.

    Returns:
        The TTS settings (a delta when ``default_voice`` is None).

    Raises:
        SettingsError: If the settings aren't supported.
    """
    from pipecat.services.deepgram.tts import DeepgramHttpTTSService

    speak = _first_speak(speak)
    url, _ = _endpoint(speak)
    if url:
        raise SettingsError("Custom Deepgram speak endpoints aren't supported.")
    provider = _provider(speak)
    provider_type = provider.get("type", "deepgram")
    if provider_type != "deepgram":
        raise SettingsError(f"Unsupported speak provider: {provider_type}")
    voice = provider.get("model") or default_voice
    expressivity = provider.get("expressivity")
    if expressivity is not None:
        if not is_flux_voice(voice):
            raise SettingsError(
                f"`expressivity` is only supported on Deepgram Flux voices, not {voice}."
            )
        if not isinstance(expressivity, int) or not -2 <= expressivity <= 2:
            raise SettingsError(f"`expressivity` must be an integer from -2 to 2: {expressivity}")
    if is_flux_voice(voice):
        from pipecat.services.deepgram.flux.tts import DeepgramFluxTTSService

        flux_fields: dict[str, Any] = {"voice": voice}
        if expressivity is not None:
            flux_fields["expressivity"] = expressivity
        return DeepgramFluxTTSService.Settings(**flux_fields)
    fields: dict[str, Any] = {}
    if voice:
        fields["voice"] = voice
    if provider.get("speed") is not None:
        fields["speed"] = float(provider["speed"])
    return DeepgramHttpTTSService.Settings(**fields)


def is_flux_voice(voice: str | None) -> bool:
    """Whether a Deepgram speak voice is a Flux one (streamed over ``/v2/speak``).

    Args:
        voice: The voice (``agent.speak.provider.model``).

    Returns:
        Whether it is.
    """
    return bool(voice) and str(voice).startswith("flux-")


def speak_update(speak: Any, *, current_type: str):
    """Build the TTS settings delta for an ``UpdateSpeak``.

    The update must keep the session's provider; it may change the voice or
    model, and speed where the provider has it.

    Args:
        speak: The ``speak`` settings from the ``UpdateSpeak``.
        current_type: The session's speak provider type.

    Returns:
        The TTS settings delta.

    Raises:
        SettingsError: If the update isn't supported.
    """
    speak = _first_speak(speak)
    provider_type = speak_provider_type(speak)
    if provider_type != current_type:
        raise SettingsError(
            f"UpdateSpeak can't change the speak provider from {current_type} to {provider_type}."
        )
    if provider_type == "deepgram":
        return speak_settings(speak)
    provider = _provider(speak)
    url, _ = _endpoint(speak)
    if provider_type == "cartesia":
        from pipecat.services.cartesia.tts import CartesiaHttpTTSService

        fields: dict[str, Any] = {}
        if (provider.get("voice") or {}).get("id"):
            fields["voice"] = provider["voice"]["id"]
        if provider.get("model_id"):
            fields["model"] = provider["model_id"]
        if _cartesia_extra(provider):
            fields["extra"] = _cartesia_extra(provider)
        return CartesiaHttpTTSService.Settings(**fields)
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

    fields = {}
    if url:
        fields["voice"] = eleven_labs_voice(url)
    if provider.get("model_id"):
        fields["model"] = provider["model_id"]
    return ElevenLabsTTSService.Settings(**fields)


class ConnectDelayFilter(BaseTextFilter):
    """Waits before each synthesis request, for ``additional_delay_ms_connect``.

    The Voice Agent API lets a client add this delay to the speak connection,
    to exercise its handling of a slow TTS provider.
    """

    def __init__(self, delay_secs: float):
        """Initialize the filter.

        Args:
            delay_secs: How long to wait.
        """
        self._delay_secs = delay_secs

    async def filter(self, text: str) -> str:
        """Wait, then pass the text on unchanged.

        Args:
            text: The text to speak.

        Returns:
            The same text.
        """
        await asyncio.sleep(self._delay_secs)
        return text


class StageDirectionFilter(BaseTextFilter):
    """Removes stage directions (``*beep*``, ``*laughs*``) from text before it's spoken.

    The Voice Agent API doesn't speak text between asterisks, and a reply that
    is only a stage direction isn't spoken at all.
    """

    async def filter(self, text: str) -> str:
        """Remove stage directions.

        Args:
            text: The text to speak.

        Returns:
            The text without stage directions.
        """
        return " ".join(_STAGE_DIRECTION.sub(" ", text).split())


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
    "create_tts",
    "endpoint_function_handler",
    "speak_settings",
    "speak_update",
    "speak_provider_type",
    "eleven_labs_voice",
    "validate_think_update",
    "StageDirectionFilter",
    "function_schemas",
]

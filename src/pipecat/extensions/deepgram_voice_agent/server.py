#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""A Deepgram Voice Agent API server built on Pipecat.

:class:`DeepgramVoiceAgentServer` serves the Voice Agent API
(``/v1/agent/converse``) on a FastAPI WebSocket, so a client written for
Deepgram's Voice Agent runs against a Pipecat pipeline unchanged. Each
connection gets its own pipeline, built from the client's ``Settings``::

    server = DeepgramVoiceAgentServer(ProviderCredentials(deepgram_api_key=...))
    app = FastAPI()
    server.add_routes(app)

Besides the converse WebSocket, :meth:`DeepgramVoiceAgentServer.add_routes`
serves the API's settings endpoints (validation and the think model
catalogue), latency metrics on ``/metrics``, and CORS preflight responses.

Client functions (``agent.think.functions`` without an ``endpoint``) run on
the client through
:meth:`~pipecat.services.llm_service.LLMService.register_client_function`.
Functions with an ``endpoint`` are requested by the server.
"""

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import aiohttp
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger

from pipecat.extensions.deepgram_voice_agent.listen import DeepgramListenUpdater, create_listen
from pipecat.extensions.deepgram_voice_agent.metrics import VoiceAgentMetrics
from pipecat.extensions.deepgram_voice_agent.processor import (
    CLIENT_MESSAGE_TYPES,
    NON_SETTINGS_MESSAGE_BEFORE_SETTINGS,
    UNPARSABLE_CLIENT_MESSAGE,
    DeepgramVoiceAgentObserver,
    DeepgramVoiceAgentProcessor,
    DeepgramVoiceAgentSpeechInjector,
    error_message,
)
from pipecat.extensions.deepgram_voice_agent.serializer import DeepgramVoiceAgentSerializer
from pipecat.extensions.deepgram_voice_agent.services import (
    ProviderCredentials,
    SettingsError,
    check_endpoint_functions,
    create_llm,
    create_tts,
    endpoint_function_handler,
    function_schemas,
    speak_provider_type,
    validate_think_update,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.turns.user_turn_strategies import StrategyMode, UserTurnStrategiesParams
from pipecat.workers.runner import WorkerRunner

PIPELINE_IN_SAMPLE_RATE = 16000
PIPELINE_OUT_SAMPLE_RATE = 24000
# The think models the settings endpoints list, by provider.
DEFAULT_THINK_MODELS: dict[str, list[str]] = {
    "open_ai": ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini"],
    "anthropic": ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1"],
    "google": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
}
PROVIDER_NAMES = {"open_ai": "OpenAI", "anthropic": "Anthropic", "google": "Google"}
# Headers a browser client may send, including those of Deepgram's SDKs.
CORS_ALLOWED_HEADERS = (
    "authorization,content-type,x-fern-language,x-fern-sdk-name,x-fern-sdk-version,"
    "x-deepgram-session-id"
)

# Default client sample rates by encoding, as in the Voice Agent API.
DEFAULT_SAMPLE_RATES = {"linear16": 24000, "mulaw": 8000, "alaw": 8000}


class DeepgramVoiceAgentServer:
    """Serves the Deepgram Voice Agent API with Pipecat pipelines.

    Supports the ``deepgram`` listen provider; the ``deepgram``, ``cartesia``
    and ``eleven_labs`` speak providers; the ``open_ai``, ``anthropic``,
    ``google`` and ``mock`` think providers; linear16, mulaw and alaw audio,
    optionally in a wav container; client and server-side functions, and the client messages
    handled by :class:`DeepgramVoiceAgentProcessor`.
    """

    def __init__(
        self,
        credentials: ProviderCredentials,
        *,
        allow_nonpublic_endpoints: bool = False,
        function_call_timeout_secs: float = 30.0,
        think_models: Mapping[str, Sequence[str]] | None = None,
        metrics: VoiceAgentMetrics | None = None,
    ):
        """Initialize the server.

        Args:
            credentials: Credentials for the providers sessions may use.
            allow_nonpublic_endpoints: Whether server-side functions may call
                non-public or non-https endpoints. Leave off unless every client
                is trusted: a client chooses the URLs the server requests.
            function_call_timeout_secs: How long a function call may take.
            think_models: The think models to list per provider, for the settings
                endpoints. Defaults to :data:`DEFAULT_THINK_MODELS`; only providers
                with credentials are listed.
            metrics: Where sessions record their latencies, served on ``/metrics``.
        """
        self._credentials = credentials
        self._allow_nonpublic_endpoints = allow_nonpublic_endpoints
        self._function_call_timeout_secs = function_call_timeout_secs
        self._think_models = dict(think_models or DEFAULT_THINK_MODELS)
        self.metrics = metrics or VoiceAgentMetrics()

    def add_routes(self, app: FastAPI, *, path: str = "/v1/agent/converse"):
        """Serve the Voice Agent API on a FastAPI app.

        Args:
            app: The app.
            path: Path of the converse WebSocket.
        """
        app.add_api_websocket_route(path, self.handle_websocket)
        routes = {
            "/v1/agent/settings/validate": (self._validate_settings, "POST"),
            "/v1/agent/settings/agent/validate": (self._validate_agent_settings, "POST"),
            "/v1/agent/settings/think/providers": (self._think_providers, "GET"),
            "/v1/agent/settings/think/providers/{provider_id}": (self._think_provider, "GET"),
            "/v1/agent/settings/think/providers/{provider_id}/models": (
                self._think_provider_models,
                "GET",
            ),
            "/v1/agent/settings/think/models": (self._think_models_list, "GET"),
            "/v1/agent/internal/settings/think/models": (self._think_models_internal, "GET"),
            "/metrics": (self._metrics, "GET"),
        }
        for route, (endpoint, method) in routes.items():
            app.add_api_route(route, endpoint, methods=[method])
        for route in [path, *routes]:
            app.add_api_route(route, _cors_preflight, methods=["OPTIONS"])

    async def validate(self, settings: Mapping[str, Any]):
        """Check a ``Settings`` message as a session would, without starting one.

        Args:
            settings: The ``Settings`` message.

        Raises:
            SettingsError: If the settings aren't valid or supported.
        """
        async with aiohttp.ClientSession() as http_session:
            await self._create_worker(None, settings, http_session, tag=None)

    async def _validate_settings(self, request: Request) -> Response:
        return await self._validation_response(await request.json())

    async def _validate_agent_settings(self, request: Request) -> Response:
        body = await request.json()
        return await self._validation_response({"type": "Settings", "agent": body.get("agent")})

    async def _validation_response(self, settings: Any) -> Response:
        if not isinstance(settings, dict):
            return _json({"error": "INVALID_SETTINGS", "description": "Expected JSON."}, 400)
        try:
            await self.validate(settings)
        except SettingsError as e:
            return _json({"error": e.code, "description": e.description}, 400)
        return _json({"valid": True})

    def _configured_think_providers(self) -> dict[str, Sequence[str]]:
        keys = {
            "open_ai": self._credentials.openai_api_key,
            "anthropic": self._credentials.anthropic_api_key,
            "google": self._credentials.google_api_key,
        }
        return {p: models for p, models in self._think_models.items() if keys.get(p)}

    async def _think_providers(self) -> Response:
        providers = self._configured_think_providers()
        return _json(
            {"providers": [{"id": p, "name": PROVIDER_NAMES.get(p, p)} for p in providers]}
        )

    async def _think_provider(self, provider_id: str) -> Response:
        if provider_id not in self._configured_think_providers():
            return _json({"error": "NOT_FOUND", "description": f"No provider {provider_id}."}, 404)
        return _json({"id": provider_id, "name": PROVIDER_NAMES.get(provider_id, provider_id)})

    async def _think_provider_models(self, provider_id: str) -> Response:
        models = self._configured_think_providers().get(provider_id)
        if models is None:
            return _json({"error": "NOT_FOUND", "description": f"No provider {provider_id}."}, 404)
        return _json({"models": [{"id": m, "name": m} for m in models]})

    async def _think_models_list(self) -> Response:
        return _json(
            {
                "models": [
                    {"id": m, "name": m, "provider": p}
                    for p, models in self._configured_think_providers().items()
                    for m in models
                ]
            }
        )

    async def _think_models_internal(self) -> Response:
        return _json(
            {
                "providers": {
                    p: {"name": PROVIDER_NAMES.get(p, p), "models": list(models)}
                    for p, models in self._configured_think_providers().items()
                }
            }
        )

    async def _metrics(self) -> Response:
        return PlainTextResponse(self.metrics.render(), media_type="text/plain; version=0.0.4")

    async def handle_websocket(self, websocket: WebSocket):
        """Run one Voice Agent session on an incoming WebSocket.

        Args:
            websocket: The client's WebSocket.
        """
        await websocket.accept()
        try:
            await websocket.send_text(
                json.dumps({"type": "Welcome", "request_id": str(uuid.uuid4())})
            )
            settings = await self._receive_settings(websocket)
            if settings is None:
                return
            tag = websocket.query_params.get("tag")
            async with aiohttp.ClientSession() as http_session:
                try:
                    worker = await self._create_worker(websocket, settings, http_session, tag)
                    assert worker is not None
                except SettingsError as e:
                    await websocket.send_text(json.dumps(error_message(e.code, e.description)))
                    return
                await websocket.send_text(json.dumps({"type": "SettingsApplied"}))
                logger.info(f"Starting Voice Agent session (tag={tag})")
                runner = WorkerRunner(handle_sigint=False)
                await runner.add_workers(worker)
                await runner.run()
                logger.info(f"Voice Agent session ended (tag={tag})")
        except WebSocketDisconnect:
            pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    async def _receive_settings(self, websocket: WebSocket) -> dict[str, Any] | None:
        """Wait for ``Settings``; report any other message before it as an error.

        ``KeepAlive`` is allowed before ``Settings``.
        """
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return None
            text = message.get("text")
            if text is None:
                await self._reject(
                    websocket,
                    NON_SETTINGS_MESSAGE_BEFORE_SETTINGS,
                    "The first message must be Settings, not audio.",
                )
                return None
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None
            if not isinstance(data, dict) or data.get("type") not in CLIENT_MESSAGE_TYPES:
                await self._reject(
                    websocket,
                    UNPARSABLE_CLIENT_MESSAGE,
                    f"Could not parse client message: {text!r}",
                )
                return None
            if data["type"] == "Settings":
                return data
            if data["type"] != "KeepAlive":
                await self._reject(
                    websocket,
                    NON_SETTINGS_MESSAGE_BEFORE_SETTINGS,
                    f"The first message must be Settings, not {data['type']}.",
                )
                return None

    async def _reject(self, websocket: WebSocket, code: str, description: str):
        await websocket.send_text(json.dumps(error_message(code, description)))

    async def _create_worker(
        self,
        websocket: WebSocket | None,
        settings: Mapping[str, Any],
        http_session: aiohttp.ClientSession,
        tag: str | None,
    ) -> PipelineWorker | None:
        audio = settings.get("audio") or {}
        audio_in = audio.get("input") or {}
        audio_out = audio.get("output") or {}
        in_encoding = audio_in.get("encoding", "linear16")
        out_encoding = audio_out.get("encoding", "linear16")
        try:
            serializer = DeepgramVoiceAgentSerializer(
                client_in_sample_rate=int(
                    audio_in.get("sample_rate") or DEFAULT_SAMPLE_RATES[in_encoding]
                ),
                client_out_sample_rate=int(
                    audio_out.get("sample_rate") or DEFAULT_SAMPLE_RATES[out_encoding]
                ),
                pipeline_in_sample_rate=PIPELINE_IN_SAMPLE_RATE,
                input_encoding=in_encoding,
                output_encoding=out_encoding,
                output_container=audio_out.get("container") or "none",
            )
        except (KeyError, ValueError) as e:
            raise SettingsError(f"Invalid audio settings: {e}")

        agent = settings.get("agent") or {}
        think = agent.get("think") or {}
        functions = think.get("functions") or []
        prompt = think.get("prompt") or ""
        await check_endpoint_functions(functions, allow_nonpublic=self._allow_nonpublic_endpoints)

        try:
            listen = create_listen(
                agent.get("listen"),
                deepgram_api_key=self._credentials.deepgram_api_key,
                tag=tag,
            )
        except ValueError as e:
            raise SettingsError(str(e))
        llm = await create_llm(
            think,
            credentials=self._credentials,
            http_session=http_session,
            system_instruction=prompt,
        )
        tts = create_tts(
            agent.get("speak"),
            credentials=self._credentials,
            http_session=http_session,
            sample_rate=PIPELINE_OUT_SAMPLE_RATE,
        )

        processor = DeepgramVoiceAgentProcessor(
            system_instruction=prompt, greeting=agent.get("greeting")
        )
        observer = DeepgramVoiceAgentObserver(
            processor, experimental=bool(settings.get("experimental"))
        )
        processor.listen_updater = DeepgramListenUpdater(
            listen,
            send=processor.send,
            push_frame=processor.push_frame,
            deepgram_api_key=self._credentials.deepgram_api_key,
            http_session=http_session,
        )
        processor.listen = listen
        processor.tts = tts
        processor.speak_provider = speak_provider_type(agent.get("speak"))
        observer.include_timing_data = bool(
            processor.speak_provider == "eleven_labs"
            and _speak_provider(agent.get("speak")).get("include_timing_data")
        )

        async def validate_think(think: Mapping[str, Any]):
            await validate_think_update(
                think, credentials=self._credentials, http_session=http_session
            )

        processor.validate_think = validate_think

        # As in the Voice Agent API, a call goes out as soon as the LLM makes it,
        # before the user's turn is confirmed, unless its function asks to wait
        # (defer_until_eot). Anything not declared still runs on the client.
        llm.register_client_function(None, timeout_secs=self._function_call_timeout_secs)
        for function in functions:
            speculative = not function.get("defer_until_eot", False)
            if function.get("endpoint"):
                llm.register_function(
                    function["name"],
                    endpoint_function_handler(
                        function["endpoint"],
                        http_session=http_session,
                        send=processor.send,
                        allow_nonpublic=self._allow_nonpublic_endpoints,
                    ),
                    timeout_secs=self._function_call_timeout_secs,
                    speculative=speculative,
                )
            else:
                llm.register_client_function(
                    function["name"],
                    timeout_secs=self._function_call_timeout_secs,
                    speculative=speculative,
                )

        context = LLMContext(tools=function_schemas(functions)) if functions else LLMContext()
        # The active listen service recommends the turn strategies: Flux detects
        # turns itself, and the v1 service needs no VAD either. Either ends a turn
        # when the service proposes it, as an interrupting message has it do.
        aggregators = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies_params=UserTurnStrategiesParams(stop_mode=StrategyMode.FIRST)
            ),
        )
        observer.languages_hinted = lambda: listen.settings[listen.active].get("language_hints")
        processor.speech_injector = DeepgramVoiceAgentSpeechInjector()
        processor.agent_busy = observer.agent_busy

        observer.on_latency = self.metrics.observe
        if websocket is None:
            # Validating only: everything that could be refused has been checked.
            return None

        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_sample_rate=PIPELINE_IN_SAMPLE_RATE,
                audio_out_sample_rate=PIPELINE_OUT_SAMPLE_RATE,
                add_wav_header=False,
                serializer=serializer,
            ),
        )

        pipeline = Pipeline(
            [
                transport.input(),
                processor,
                listen.switcher,
                aggregators.user(),
                processor.llm_input_gate,
                llm,
                processor.speech_injector,
                tts,
                transport.output(),
                aggregators.assistant(),
            ]
        )
        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=PIPELINE_IN_SAMPLE_RATE,
                audio_out_sample_rate=PIPELINE_OUT_SAMPLE_RATE,
            ),
            observers=[observer],
            enable_rtvi=False,
            idle_timeout_secs=None,
            # A service that can no longer work ends the session, so the client
            # sees the connection close, as with the Voice Agent API.
            processor_unusable_policy=ProcessorUnusablePolicy.CANCEL,
        )

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            await worker.cancel()

        return worker


def _speak_provider(speak: Any) -> Mapping[str, Any]:
    if isinstance(speak, list):
        speak = speak[0] if speak else None
    return (speak or {}).get("provider") or {}


def _json(body: Any, status: int = 200) -> Response:
    return JSONResponse(body, status_code=status, headers={"Access-Control-Allow-Origin": "*"})


async def _cors_preflight() -> Response:
    """Answer a CORS preflight, allowing the headers browser clients send."""
    return Response(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            "Access-Control-Allow-Headers": CORS_ALLOWED_HEADERS,
            "Access-Control-Max-Age": "86400",
        },
    )

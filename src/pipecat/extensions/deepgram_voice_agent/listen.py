#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The listen (speech-to-text) side of a Deepgram Voice Agent API session.

A Voice Agent client can move between Deepgram's two listen APIs mid-session
with ``UpdateListen``: v1 (Nova and other models) and v2 (Flux). Both STT
services sit behind a :class:`~pipecat.pipeline.service_switcher.ServiceSwitcher`,
and :class:`DeepgramListenUpdater` applies each update and reports it the way
the Voice Agent API does: ``ListenUpdated`` once Deepgram confirms the change,
``FAILED_TO_START_LISTENING`` if it can't be made.
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import aiohttp
from loguru import logger

from pipecat.frames.frames import Frame, ManuallySwitchServiceFrame, STTUpdateSettingsFrame
from pipecat.pipeline.service_switcher import ServiceSwitcher
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.stt_service import STTService
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

DEFAULT_V1_MODEL = "nova-3"
DEFAULT_V2_MODEL = "flux-general-en"
MAX_KEYTERMS = 100
LISTEN_UPDATE_TIMEOUT_SECS = 15.0

# Provider fields an UpdateListen may change.
UPDATABLE_PROVIDER_FIELDS = {
    "type",
    "model",
    "version",
    "keyterms",
    "eot_threshold",
    "eager_eot_threshold",
    "eot_timeout_ms",
    "language_hints",
}
# Provider fields that map onto Pipecat settings, by Voice Agent name.
SETTINGS_FIELDS = {
    "model": "model",
    "keyterms": "keyterm",
    "eot_threshold": "eot_threshold",
    "eager_eot_threshold": "eager_eot_threshold",
    "eot_timeout_ms": "eot_timeout_ms",
    "language_hints": "language_hints",
}
V1_SETTINGS_FIELDS = {"model", "keyterms"}

UPDATE_LISTEN_UNSUPPORTED_FIELDS_CHANGED = "UPDATE_LISTEN_UNSUPPORTED_FIELDS_CHANGED"
FAILED_TO_START_LISTENING = "FAILED_TO_START_LISTENING"
INVALID_SETTINGS = "INVALID_SETTINGS"


def is_v2_model(model: str) -> bool:
    """Whether a listen model is served by the v2 (Flux) API.

    Args:
        model: The model name.

    Returns:
        Whether it is.
    """
    return model.startswith("flux")


class DeepgramV1STTService(DeepgramSTTService):
    """:class:`DeepgramSTTService` that recommends turn strategies needing no VAD.

    A transcript starts the user's turn and a pause after it ends the turn, so
    the user aggregator needs no local VAD. Recommending them (rather than
    configuring the aggregator) lets a :class:`ServiceSwitcher` move between
    this service and Flux, each bringing its own turn detection.
    """

    def service_metadata_frame(self):
        """Build the metadata frame, recommending transcription-based turns.

        Returns:
            The STT metadata frame.
        """
        frame = super().service_metadata_frame()
        frame.user_turn_strategies = UserTurnStrategies(
            start=[TranscriptionUserTurnStartStrategy()],
            stop=[SpeechTimeoutUserTurnStopStrategy()],
        )
        return frame


@dataclass
class DeepgramListen:
    """The listen services of a session, and their current settings.

    Parameters:
        v1: The v1 (Nova) STT service.
        v2: The v2 (Flux) STT service.
        switcher: The switcher holding both; it goes in the pipeline.
        active: The service currently listening.
        settings: Each service's current provider settings, by Voice Agent name.
    """

    v1: DeepgramV1STTService
    v2: DeepgramFluxSTTService
    switcher: ServiceSwitcher
    active: STTService
    settings: dict[STTService, dict[str, Any]] = field(default_factory=dict)

    def service_for(self, model: str) -> STTService:
        """The service that serves a model.

        Args:
            model: The model name.

        Returns:
            The v2 service for Flux models, otherwise the v1 service.
        """
        return self.v2 if is_v2_model(model) else self.v1


def create_listen(
    listen: Mapping[str, Any] | None, *, deepgram_api_key: str, tag: str | None = None
) -> DeepgramListen:
    """Create the listen services for ``agent.listen``.

    Both APIs get a service, so ``UpdateListen`` can move between them; the
    one serving the requested model starts active.

    Args:
        listen: The ``agent.listen`` settings.
        deepgram_api_key: Deepgram API key.
        tag: Usage tag for the Deepgram requests.

    Returns:
        The listen services.

    Raises:
        ValueError: If the provider isn't Deepgram or has too many keyterms.
    """
    provider = dict((listen or {}).get("provider") or {})
    if provider.get("type", "deepgram") != "deepgram":
        raise ValueError(f"Unsupported listen provider: {provider.get('type')}")
    if len(provider.get("keyterms") or []) > MAX_KEYTERMS:
        raise ValueError(f"At most {MAX_KEYTERMS} keyterms are allowed.")
    model = provider.get("model") or DEFAULT_V1_MODEL
    tags = [tag] if tag else None

    v1_settings: dict[str, Any] = (
        provider if not is_v2_model(model) else {"model": DEFAULT_V1_MODEL}
    )
    v2_settings: dict[str, Any] = provider if is_v2_model(model) else {"model": DEFAULT_V2_MODEL}

    v1 = DeepgramV1STTService(
        api_key=deepgram_api_key,
        tag=tags,
        settings=DeepgramSTTService.Settings(
            model=v1_settings.get("model") or DEFAULT_V1_MODEL,
            smart_format=True,
            interim_results=True,
            keyterm=v1_settings.get("keyterms") or None,
        ),
    )
    v2 = DeepgramFluxSTTService(
        api_key=deepgram_api_key,
        tag=tags,
        settings=DeepgramFluxSTTService.Settings(
            model=v2_settings.get("model") or DEFAULT_V2_MODEL,
            keyterm=v2_settings.get("keyterms") or [],
            eot_threshold=v2_settings.get("eot_threshold"),
            eager_eot_threshold=v2_settings.get("eager_eot_threshold"),
            eot_timeout_ms=v2_settings.get("eot_timeout_ms"),
        ),
    )
    active = v2 if is_v2_model(model) else v1
    # The switcher starts with its first service active.
    switcher = ServiceSwitcher(services=[active, v1 if active is v2 else v2])
    return DeepgramListen(
        v1=v1,
        v2=v2,
        switcher=switcher,
        active=active,
        settings={v1: _known(v1_settings, v1=True), v2: _known(v2_settings, v1=False)},
    )


def _known(provider: Mapping[str, Any], *, v1: bool) -> dict[str, Any]:
    fields = V1_SETTINGS_FIELDS if v1 else SETTINGS_FIELDS.keys()
    return {name: provider.get(name) for name in fields}


class DeepgramListenUpdater:
    """Applies ``UpdateListen`` messages and reports them to the client.

    An update is checked first: fields that can't change are refused with a
    ``Warning``, and too many keyterms with an ``Error``. While the user is
    speaking an update waits for the turn to end, and a newer update replaces
    one still waiting. Applying it:

    - A new model is probed with a short connection first, so one Deepgram
      won't serve fails with ``FAILED_TO_START_LISTENING`` and leaves the
      working connection alone.
    - Settings go to the service serving the model, which reconnects or (for
      Flux keyterms and thresholds) sends Deepgram a ``Configure`` message. A
      model on the other API switches services.
    - ``ListenUpdated`` goes out when Deepgram confirms: a new connection
      (``on_connected``) or an accepted ``Configure``
      (``on_configure_success``). A refusal, or no confirmation in time, is a
      ``FAILED_TO_START_LISTENING`` error. Only the newest update is reported.
    """

    def __init__(
        self,
        listen: DeepgramListen,
        *,
        send: Callable[[dict[str, Any]], Awaitable[None]],
        push_frame: Callable[[Frame], Awaitable[None]],
        deepgram_api_key: str,
        http_session: aiohttp.ClientSession,
        timeout_secs: float = LISTEN_UPDATE_TIMEOUT_SECS,
    ):
        """Initialize the updater.

        Args:
            listen: The session's listen services.
            send: Sends a message to the client.
            push_frame: Pushes a frame toward the listen services.
            deepgram_api_key: Deepgram API key, for probing models.
            http_session: HTTP session, for probing models.
            timeout_secs: How long to wait for Deepgram to confirm an update.
        """
        self._listen = listen
        self._send = send
        self._push_frame = push_frame
        self._api_key = deepgram_api_key
        self._http_session = http_session
        self._timeout_secs = timeout_secs
        self._user_speaking = False
        self._latest = 0
        self._pending: tuple[int, dict[str, Any]] | None = None
        self._tasks: set[asyncio.Task] = set()
        # Deepgram's confirmations, per service.
        self._waiters: dict[STTService, asyncio.Future] = {}
        for service in (listen.v1, listen.v2):
            service.add_event_handler("on_connected", self._on_confirmed)
            service.add_event_handler("on_connection_error", self._on_refused)
        listen.v2.add_event_handler("on_configure_success", self._on_confirmed)
        listen.v2.add_event_handler("on_configure_failure", self._on_refused)

    async def _on_confirmed(self, service: STTService, *args):
        waiter = self._waiters.get(service)
        if waiter and not waiter.done():
            waiter.set_result(True)

    async def _on_refused(self, service: STTService, *args):
        waiter = self._waiters.get(service)
        if waiter and not waiter.done():
            waiter.set_result(False)

    async def set_user_speaking(self, speaking: bool):
        """Track the user's turn; an update waiting for it is applied when it ends.

        Args:
            speaking: Whether the user is speaking.
        """
        self._user_speaking = speaking
        if not speaking and self._pending:
            generation, provider = self._pending
            self._pending = None
            self._start(generation, provider)

    async def update(self, listen: Mapping[str, Any] | None):
        """Handle an ``UpdateListen``'s ``listen`` object.

        Args:
            listen: The requested listen settings.
        """
        listen = dict(listen or {})
        provider = dict(listen.get("provider") or {})
        unsupported = sorted(
            [name for name in listen if name != "provider"]
            + [name for name in provider if name not in UPDATABLE_PROVIDER_FIELDS]
        )
        if unsupported:
            await self._send(
                {
                    "type": "Warning",
                    "code": UPDATE_LISTEN_UNSUPPORTED_FIELDS_CHANGED,
                    "description": f"These listen fields can't be changed: {', '.join(unsupported)}",
                }
            )
            return
        if provider.get("type", "deepgram") != "deepgram":
            await self._error(INVALID_SETTINGS, f"Unsupported listen provider: {provider['type']}")
            return
        if len(provider.get("keyterms") or []) > MAX_KEYTERMS:
            await self._error(INVALID_SETTINGS, f"At most {MAX_KEYTERMS} keyterms are allowed.")
            return

        self._latest += 1
        if self._user_speaking:
            self._pending = (self._latest, provider)
        else:
            self._start(self._latest, provider)

    def _start(self, generation: int, provider: dict[str, Any]):
        task = asyncio.get_running_loop().create_task(self._apply(generation, provider))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _apply(self, generation: int, provider: dict[str, Any]):
        listen = self._listen
        current = listen.settings[listen.active]
        model = provider.get("model") or current.get("model") or DEFAULT_V1_MODEL
        target = listen.service_for(model)
        known = V1_SETTINGS_FIELDS if target is listen.v1 else SETTINGS_FIELDS.keys()
        target_settings = listen.settings[target]
        changes = {
            name: provider[name]
            for name in known
            if name in provider and provider[name] != target_settings.get(name)
        }
        if "model" not in changes and target_settings.get("model") != model:
            changes["model"] = model
        switching = target is not listen.active

        if "model" in changes and not await self._model_is_served(changes["model"]):
            await self._report(
                generation, False, f"Deepgram did not accept listen model {changes['model']}."
            )
            return

        confirmed = True
        if changes:
            loop = asyncio.get_running_loop()
            self._waiters[target] = loop.create_future()
            delta_class = (
                DeepgramFluxSTTService.Settings
                if target is listen.v2
                else DeepgramSTTService.Settings
            )
            delta = delta_class(**{SETTINGS_FIELDS[name]: value for name, value in changes.items()})
            await self._push_frame(STTUpdateSettingsFrame(delta=delta, service=target))
        if switching:
            await self._push_frame(ManuallySwitchServiceFrame(service=target))
            listen.active = target
        if changes:
            try:
                confirmed = await asyncio.wait_for(self._waiters[target], self._timeout_secs)
            except TimeoutError:
                confirmed = False
            finally:
                self._waiters.pop(target, None)
            if confirmed:
                target_settings.update(changes)
        await self._report(
            generation, confirmed, "Deepgram did not confirm the listen update in time."
        )

    async def _report(self, generation: int, ok: bool, failure: str):
        if generation != self._latest:
            return
        if ok:
            await self._send({"type": "ListenUpdated"})
        else:
            await self._error(FAILED_TO_START_LISTENING, failure)

    async def _error(self, code: str, description: str):
        await self._send({"type": "Error", "code": code, "description": description})

    async def _model_is_served(self, model: str) -> bool:
        """Probe Deepgram with a short connection for the model."""
        path = "/v2/listen" if is_v2_model(model) else "/v1/listen"
        query = urlencode({"model": model, "encoding": "linear16", "sample_rate": 16000})
        url = f"wss://api.deepgram.com{path}?{query}"
        try:
            async with asyncio.timeout(10):
                async with self._http_session.ws_connect(
                    url, headers={"Authorization": f"Token {self._api_key}"}
                ):
                    return True
        except aiohttp.WSServerHandshakeError as e:
            logger.info(f"Deepgram refused listen model {model}: {e.status}")
            return False
        except (aiohttp.ClientError, TimeoutError) as e:
            # Can't tell; let the update go ahead and Deepgram decide.
            logger.warning(f"Could not probe listen model {model}: {e}")
            return True

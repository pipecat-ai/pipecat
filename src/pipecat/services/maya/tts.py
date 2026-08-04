#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Maya Research text-to-speech service.

Maya 2 synthesizes ten Indian languages plus Indian English, with two voices
that all languages share.
"""

import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
)
from pipecat.services.settings import TTSSettings, assert_given
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Maya, you need to `pip install pipecat-ai[maya]`.")
    raise ImportError(f"Missing module: {e}") from e

MAYA_WEBSOCKET_URL = "wss://tts.mayaresearch.ai/v1/tts/stream"
"""Websocket endpoint for Maya's streaming API."""

MAYA_SAMPLE_RATE = 24000
"""Maya always returns raw PCM, 16-bit signed little-endian, mono, 24kHz."""

MAYA_LANGUAGES = ("hi", "bn", "gu", "kn", "ml", "mr", "or", "pa", "ta", "te", "en")
"""The language codes verified against Maya.

Codes outside this tuple are still sent, so languages Maya adds later work
without a release; Maya validates the field and reports unknown codes.
"""


def language_to_maya_language(language: Language) -> str:
    """Convert a Language enum to a Maya language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Maya language code. Languages outside the verified
        map fall back to their base code (e.g. ``fr`` from ``fr-CA``) with a
        warning, and Maya reports the ones it doesn't speak.
    """
    LANGUAGE_MAP = {
        # `en` is Indian English. There is no British or American variant, so
        # every English locale maps here.
        Language.EN: "en",
        Language.EN_GB: "en",
        Language.EN_IN: "en",
        Language.EN_US: "en",
        Language.BN: "bn",  # Bengali
        Language.BN_IN: "bn",
        Language.GU: "gu",  # Gujarati
        Language.GU_IN: "gu",
        Language.HI: "hi",  # Hindi
        Language.HI_IN: "hi",
        Language.KN: "kn",  # Kannada
        Language.KN_IN: "kn",
        Language.ML: "ml",  # Malayalam
        Language.ML_IN: "ml",
        Language.MR: "mr",  # Marathi
        Language.MR_IN: "mr",
        Language.OR: "or",  # Odia
        Language.OR_IN: "or",
        Language.PA: "pa",  # Punjabi
        Language.PA_IN: "pa",
        Language.TA: "ta",  # Tamil
        Language.TA_IN: "ta",
        Language.TE: "te",  # Telugu
        Language.TE_IN: "te",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


@dataclass
class MayaTTSSettings(TTSSettings):
    """Settings for MayaTTSService."""

    pass


class MayaTTSService(WebsocketTTSService):
    """Text-to-speech using Maya's streaming websocket API.

    Keeps one websocket open for the whole conversation, so the handshake is
    paid once rather than per utterance. Sentences of a turn are sent without
    waiting for the previous one's audio; every frame carries the turn's context
    ID, so audio routes correctly even when several sentences are in flight.
    Interruptions cancel the whole turn server-side. Voice and language can be
    changed mid-call and apply from the next turn onward.
    """

    Settings = MayaTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = MAYA_WEBSOCKET_URL,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Maya websocket TTS service.

        Args:
            api_key: Maya API key, sent as a bearer token on the upgrade request.
            url: Websocket endpoint. Defaults to Maya's hosted endpoint.
            sample_rate: Output sample rate in Hz. If None, uses the pipeline
                default. Maya synthesizes at 24000 Hz; other rates are resampled.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to the parent service.
        """
        default_settings = self.Settings(
            model=None,  # Maya exposes a single model over the websocket.
            voice="Ananya",
            language=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        language = default_settings.language
        if isinstance(language, Language):
            default_settings.language = self.language_to_service_language(language)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._websocket = None
        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Maya supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str:
        """Convert a Language enum to a Maya language code.

        Args:
            language: The language to convert.

        Returns:
            The Maya language code.
        """
        return language_to_maya_language(language)

    async def start(self, frame: StartFrame):
        """Start the service and open the websocket.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close the websocket.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close the websocket immediately.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self, context_id: str | None = None):
        """Close the current turn so Maya emits its final audio and ``end``.

        Maya keeps a turn open while sentences arrive with ``continue: true``.
        An empty text frame with ``continue: false`` marks the turn complete.

        Args:
            context_id: The context to close. If None, falls back to the
                currently active context.
        """
        flush_id = context_id or self.get_active_audio_context_id()
        if not flush_id or not self._websocket:
            return
        logger.trace(f"{self}: closing turn {flush_id}")
        await self._websocket.send(
            json.dumps({"type": "text", "context_id": flush_id, "text": "", "continue": False})
        )

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the turn server-side when the bot is interrupted.

        Maya replies ``cancelled`` instead of ``end`` and drops every queued and
        in-flight sentence of that turn. The socket stays usable.

        Args:
            context_id: The context that was cut short.
        """
        await self.stop_all_metrics()
        if context_id and self._websocket:
            await self._websocket.send(json.dumps({"type": "cancel", "context_id": context_id}))
        await super().on_audio_context_interrupted(context_id)

    async def _connect(self):
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to Maya TTS")
            self._websocket = await websocket_connect(
                self._url,
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
                max_size=None,
            )
            await self._send_start()
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Maya TTS: {e}", exception=e)
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _send_start(self):
        """Select the v2 protocol and set the voice and language for the session."""
        start = {"type": "start", "v2": True, "voice": assert_given(self._settings.voice)}
        language = assert_given(self._settings.language)
        if language:
            if language not in MAYA_LANGUAGES:
                logger.warning(
                    f"{self}: language '{language}' is not in the verified set "
                    f"({', '.join(MAYA_LANGUAGES)}); sending it anyway."
                )
            start["language"] = language
        await self._get_websocket().send(json.dumps(start))

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug("Disconnecting from Maya TTS")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing Maya websocket: {e}", exception=e)
        finally:
            await self.remove_active_audio_context()
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _update_settings(self, delta: Settings) -> dict:
        """Apply a settings delta, pushing voice and language to the session.

        Maya re-reads voice and language from a ``start`` frame sent mid-session,
        so the change applies from the next turn onward.

        Args:
            delta: A TTS settings delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if changed and self._websocket and self._websocket.state is State.OPEN:
            await self._send_start()
        return changed

    async def _receive_messages(self):
        """Route incoming frames into the audio context they belong to."""
        async for message in self._get_websocket():
            try:
                msg = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"{self}: discarding unparseable frame")
                continue

            kind = msg.get("type")
            context_id = msg.get("context_id")

            # An interrupted turn is already closed on our side, so anything
            # still arriving for it is discarded rather than warned about.
            known = bool(context_id) and self.audio_context_available(context_id)

            if kind == "audio":
                # A frame missing its payload must not take down the receive
                # task, or the socket goes deaf for the rest of the call.
                audio = msg.get("audio")
                if known and audio:
                    await self.stop_ttfb_metrics()
                    await self._append_audio(context_id, base64.b64decode(audio))
            elif kind in ("end", "cancelled"):
                # A turn ends with exactly one of these, never both.
                if known:
                    await self.remove_audio_context(context_id)
            elif kind == "error":
                await self.push_error(error_msg=f"Maya TTS error: {msg.get('error')}")
                if known:
                    await self.remove_audio_context(context_id)

    async def _append_audio(self, context_id: str, pcm: bytes):
        if not pcm:
            return
        async for frame in self._stream_audio_frames_from_iterator(
            _once(pcm), in_sample_rate=MAYA_SAMPLE_RATE, context_id=context_id
        ):
            await self.append_to_audio_context(context_id, frame)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send one sentence of a turn; audio arrives via the receive task.

        Sentences are sent with ``continue: true`` so the turn stays open for
        the rest of the LLM response. ``flush_audio`` closes it.

        Args:
            text: The text to synthesize.
            context_id: The audio context, which is Maya's turn ID.

        Yields:
            ``None`` — audio is delivered out of band by the receive task.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                await self._get_websocket().send(
                    json.dumps(
                        {
                            "type": "text",
                            "context_id": context_id,
                            "text": text,
                            "continue": True,
                        }
                    )
                )
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Maya TTS send failed: {e}", exception=e)
                await self._disconnect()
                await self._connect()
                return

            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Unknown error occurred: {e}", exception=e)


async def _once(data: bytes):
    """Yield a single chunk as an async iterator."""
    yield data

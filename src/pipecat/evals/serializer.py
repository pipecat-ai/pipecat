#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The two ends of the eval wire.

:class:`EvalSerializer` runs in the bot, as the serializer of its
:class:`~pipecat.evals.transport.EvalTransport`; :class:`EvalClientSerializer`
runs in the harness, in :class:`~pipecat.evals.client_transport.EvalClientTransport`.
Both speak RTVI. The eval adds messages of its own on top: ``eval-configure``,
``eval-context``, ``eval-cancel``, and ``eval-image`` from the harness, and
``eval-bot-audio`` from the bot when the harness asks to hear it.
"""

import base64
import contextlib
import json
from typing import Any

from loguru import logger
from pydantic import ValidationError

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    CancelWorkerFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    LLMMessagesUpdateFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    MetricsData,
    ProcessingMetricsData,
    TTFBMetricsData,
)
from pipecat.processors.frameworks.rtvi.frames import RTVIConfigureObserverFrame
from pipecat.processors.frameworks.rtvi.observer import RTVIFunctionCallReportLevel
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.serializers.rtvi_client import RTVIClientSerializer
from pipecat.utils.deprecation import deprecated

# A ``client-message`` with this ``t`` is intercepted by the serializer and
# turned into an ``LLMMessagesUpdateFrame`` instead of being forwarded to the
# RTVIProcessor. Keeps per-eval context seeding out of the bot.
EVAL_CONTEXT_MESSAGE_TYPE = "eval-context"

# A ``client-message`` with this ``t`` is intercepted and turned into an
# ``RTVIConfigureObserverFrame``. This is the trust boundary for raising the
# function-call report level: only the eval transport understands it, so a
# production RTVI serializer can't be elevated by a remote client.
EVAL_CONFIGURE_MESSAGE_TYPE = "eval-configure"

# A ``client-message`` with this ``t`` is intercepted and turned into a
# ``CancelWorkerFrame``, so the harness can end a scenario by gracefully tearing down the
# bot's pipeline (closing its service connections) instead of the orchestrator
# having to kill the process.
EVAL_CANCEL_MESSAGE_TYPE = "eval-cancel"

# A ``client-message`` with this ``t`` registers an image (base64-encoded
# PNG/JPEG/... bytes plus its MIME ``format``) on the serializer. The eval input
# transport hands it back as a ``UserImageRawFrame`` when the bot asks for a user
# image (``UserImageRequestFrame``), so a function-calling-video bot can be driven
# without a real camera. The harness sends the image encoded over the wire; the
# eval input transport decodes it and pushes a raw ``UserImageRawFrame`` (matching
# a real camera transport).
EVAL_IMAGE_MESSAGE_TYPE = "eval-image"

# Outbound message carrying a chunk of the bot's synthesized audio (base64), used
# only when a scenario asserts on ``tts_response``. Emitted under the RTVI label
# so the harness reader sees it; the harness transcribes the audio locally.
EVAL_BOT_AUDIO_TYPE = "eval-bot-audio"

# Rate the harness resamples the bot's audio to before its pipeline VAD/STT see
# it. 16 kHz is what Silero and the local STT models (Whisper/Moonshine) expect;
# the harness configures its input transport at this rate to match.
EVAL_STT_SAMPLE_RATE = 16000


class EvalSerializer(FrameSerializer):
    """Bridges RTVI messages and frames on the bot's side of an eval.

    The serializer of :class:`~pipecat.evals.transport.EvalTransport`;
    :class:`EvalClientSerializer` is the harness's end. The bot pipeline needs
    an ``RTVIProcessor`` and an ``RTVIObserver``.
    """

    def __init__(self, **kwargs):
        """Initialize the serializer.

        Args:
            **kwargs: Additional arguments passed to ``FrameSerializer``.
        """
        # Do not ignore RTVI messages: the whole point is to put them on the
        # wire so the harness can observe semantic events.
        super().__init__(params=FrameSerializer.InputParams(ignore_rtvi_messages=False), **kwargs)
        # Off by default; the eval transport flips this on per connection (from
        # the ?capture_bot_audio query param) only for tts_response scenarios, so we
        # don't ship the bot's audio over the wire unless something asserts on it.
        self._capture_audio = False
        # The most recent image the harness registered (still-encoded bytes, MIME
        # type), served back on a UserImageRequestFrame. See EVAL_IMAGE_MESSAGE_TYPE.
        self._user_image: tuple[bytes, str] | None = None

    def set_capture_audio(self, capture: bool) -> None:
        """Enable/disable forwarding the bot's synthesized audio to the harness."""
        self._capture_audio = capture

    def get_user_image(self) -> tuple[bytes, str] | None:
        """The image registered for the current turn as ``(bytes, mime)``, or None."""
        return self._user_image

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serialize an outbound frame for the harness; only RTVI server messages go out.

        Args:
            frame: The frame to serialize.

        Returns:
            JSON text for an RTVI server message, or ``None`` to drop the frame.
        """
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            message = frame.message
            if isinstance(message, dict) and message.get("label") == RTVI.MESSAGE_LABEL:
                return json.dumps(message)
        elif self._capture_audio and isinstance(frame, OutputAudioRawFrame):
            return json.dumps(
                {
                    "label": RTVI.MESSAGE_LABEL,
                    "type": EVAL_BOT_AUDIO_TYPE,
                    "data": {
                        "audio": base64.b64encode(frame.audio).decode("ascii"),
                        "sampleRate": frame.sample_rate,
                    },
                }
            )
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserialize an inbound JSON RTVI message into a frame.

        Args:
            data: JSON text (or bytes) sent by the harness.

        Returns:
            An ``LLMMessagesUpdateFrame`` for the eval-context control message, an
            ``InputTransportMessageFrame`` wrapping any other RTVI message, or
            ``None`` if the payload is not a valid RTVI message.
        """
        try:
            message = json.loads(data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"EvalSerializer: dropping non-JSON message: {e}")
            return None

        if not isinstance(message, dict) or message.get("label") != RTVI.MESSAGE_LABEL:
            logger.warning(f"EvalSerializer: ignoring non-RTVI message: {message!r}")
            return None

        context = self._maybe_context_frame(message)
        if context is not None:
            return context

        configure = self._maybe_configure_frame(message)
        if configure is not None:
            return configure

        cancel = self._maybe_cancel_frame(message)
        if cancel is not None:
            return cancel

        if self._maybe_store_image(message):
            return None

        return InputTransportMessageFrame(message=message)

    def _maybe_context_frame(self, message: dict) -> Frame | None:
        """Return an ``LLMMessagesUpdateFrame`` for the eval-context message, else None."""
        if message.get("type") != "client-message":
            return None
        data: Any = message.get("data") or {}
        if not isinstance(data, dict) or data.get("t") != EVAL_CONTEXT_MESSAGE_TYPE:
            return None
        payload = data.get("d") or {}
        messages = payload.get("messages", []) if isinstance(payload, dict) else []
        # run_llm=False: seed the context, don't trigger a response.
        return LLMMessagesUpdateFrame(messages=list(messages), run_llm=False)

    def _maybe_cancel_frame(self, message: dict) -> Frame | None:
        """The frame for an eval-cancel message, else None: it cancels the whole pipeline so the bot exits on its own."""
        if message.get("type") != "client-message":
            return None
        data: Any = message.get("data") or {}
        if not isinstance(data, dict) or data.get("t") != EVAL_CANCEL_MESSAGE_TYPE:
            return None
        return CancelWorkerFrame()

    def _maybe_store_image(self, message: dict) -> bool:
        """Keep the image from an eval-image message for the input transport to serve; True when consumed."""
        if message.get("type") != "client-message":
            return False
        data: Any = message.get("data") or {}
        if not isinstance(data, dict) or data.get("t") != EVAL_IMAGE_MESSAGE_TYPE:
            return False
        payload = data.get("d") or {}
        encoded = payload.get("image") if isinstance(payload, dict) else None
        if encoded:
            fmt = str(payload.get("format") or "image/jpeg")
            self._user_image = (base64.b64decode(encoded), fmt)
        return True

    def _maybe_configure_frame(self, message: dict) -> Frame | None:
        """Return an ``RTVIConfigureObserverFrame`` for the eval-configure message, else None."""
        if message.get("type") != "client-message":
            return None
        data: Any = message.get("data") or {}
        if not isinstance(data, dict) or data.get("t") != EVAL_CONFIGURE_MESSAGE_TYPE:
            return None
        payload = data.get("d") or {}
        if not isinstance(payload, dict):
            payload = {}
        levels = payload.get("function_call_report_level")
        report_level = None
        if isinstance(levels, dict):
            # Values arrive as strings ("none"/"name"/"full"); coerce to the enum.
            report_level = {k: RTVIFunctionCallReportLevel(v) for k, v in levels.items()}
        vad = payload.get("vad_user_speaking")
        vad_user_speaking_enabled = bool(vad) if vad is not None else None
        return RTVIConfigureObserverFrame(
            function_call_report_level=report_level,
            vad_user_speaking_enabled=vad_user_speaking_enabled,
        )


# The bot's reports about *the user it is talking to* (i.e. about the harness):
# its raw VAD, turn-level speaking, and the transcription of what it heard. The
# harness surfaces these as scenario events, but does not let them drive its own
# pipeline. They are kept as raw messages (mapped to events by the sink) for two
# reasons: a ``TranscriptionFrame`` in the eval pipeline already means "our STT
# transcribed the bot's audio" (the ``response``), and the VAD/speaking frames are
# computed locally by the harness's own user aggregator from the bot's audio (a
# different signal than the bot's VAD on the harness's audio). Routing the reports
# as messages keeps both sides available without colliding by frame type.
_REPORTED_EVENT_TYPES = frozenset(
    {
        "user-started-speaking",
        "user-stopped-speaking",
        "vad-user-started-speaking",
        "vad-user-stopped-speaking",
        "user-transcription",
        # The bot reporting *its* output was interrupted. Kept as a message so it
        # doesn't become an InterruptionFrame, which the harness's own user
        # aggregator already broadcasts when our VAD detects the bot speaking.
        "bot-interrupted",
    }
)


@deprecated(
    "`RTVIEvalSerializer` is deprecated since 1.9.0 and will be removed in 2.0.0. "
    "Use `EvalSerializer` instead."
)
class RTVIEvalSerializer(EvalSerializer):
    """Deprecated alias for :class:`EvalSerializer`.

    .. deprecated:: 1.9.0
        Use :class:`EvalSerializer` instead. Will be removed in 2.0.0.
    """


def _metrics_frame(data: dict) -> MetricsFrame | None:
    """The bot's ``metrics`` message as a ``MetricsFrame``, or ``None`` when it carries nothing usable.

    The RTVI observer sends each metric type as a list of dumped models under
    its own key; the ones a turn's timing keeps are rebuilt here. Token usage
    comes without its processor's name, which stays empty.
    """
    parsed: list[MetricsData] = []
    for item in data.get("ttfb") or []:
        with contextlib.suppress(ValidationError, TypeError):
            parsed.append(TTFBMetricsData.model_validate(item))
    for item in data.get("processing") or []:
        with contextlib.suppress(ValidationError, TypeError):
            parsed.append(ProcessingMetricsData.model_validate(item))
    for item in data.get("tokens") or []:
        with contextlib.suppress(ValidationError, TypeError):
            parsed.append(
                LLMUsageMetricsData(
                    processor=item.get("processor", ""),
                    model=item.get("model"),
                    value=LLMTokenUsage.model_validate(item),
                )
            )
    return MetricsFrame(data=parsed) if parsed else None


class EvalClientSerializer(RTVIClientSerializer):
    """The harness's serializer: RTVI client messages, plus the eval's own.

    ``eval-bot-audio`` becomes an ``InputAudioRawFrame`` so the harness's STT
    can transcribe what the bot actually said. The bot's reports about the
    harness (its transcription, VAD, and speaking messages) stay raw messages
    rather than becoming frames, so they cannot be mistaken for what the
    harness computes from the bot's audio. The bot's ``metrics`` become a
    ``MetricsFrame``, as they were in the bot's own pipeline.
    """

    def __init__(self, **kwargs):
        """Initialize the serializer.

        Args:
            **kwargs: Additional arguments passed to ``RTVIClientSerializer``.
        """
        super().__init__(**kwargs)
        # Lazily created on the first bot-audio chunk; resamples the bot's audio to
        # EVAL_STT_SAMPLE_RATE for the pipeline's VAD/STT.
        self._resampler = None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserialize an RTVI server message, handling the eval-specific types.

        Args:
            data: JSON text (or bytes) sent by the bot.

        Returns:
            The corresponding frame, or ``None`` to drop the message.
        """
        try:
            message = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return await super().deserialize(data)

        if isinstance(message, dict) and message.get("label") == RTVI.MESSAGE_LABEL:
            msg_type = message.get("type")
            if msg_type == EVAL_BOT_AUDIO_TYPE:
                payload = message.get("data") or {}
                audio = base64.b64decode(payload.get("audio", ""))
                sample_rate = int(payload.get("sampleRate", 0))
                if sample_rate and sample_rate != EVAL_STT_SAMPLE_RATE and audio:
                    audio = await self._resample(audio, sample_rate)
                    sample_rate = EVAL_STT_SAMPLE_RATE
                return InputAudioRawFrame(
                    audio=audio,
                    sample_rate=sample_rate,
                    num_channels=1,
                )
            if msg_type in _REPORTED_EVENT_TYPES:
                # Kept as the raw message so the sink maps it to a scenario event;
                # see the class docstring for why these aren't frames.
                return InputTransportMessageFrame(message=message)
            if msg_type == "metrics":
                # The base serializer has no frame for these; the turn's timing reads them.
                return _metrics_frame(message.get("data") or {})

        return await super().deserialize(data)

    async def _resample(self, audio: bytes, in_rate: int) -> bytes:
        """Resample the bot's audio to the STT rate, with a stream resampler so chunk boundaries do not garble it."""
        if self._resampler is None:
            from pipecat.audio.utils import create_stream_resampler

            self._resampler = create_stream_resampler()
        return await self._resampler.resample(audio, in_rate, EVAL_STT_SAMPLE_RATE)

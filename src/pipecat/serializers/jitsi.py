#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Jitsi agent media protocol (``jitsi-agent-media``) WebSocket serializer for Pipecat.

A Jitsi Meet or JaaS deployment with voice agents enabled dials *out* to the bot's
WebSocket server and speaks ``jitsi-agent-media``, a JSON protocol shaped like Twilio
Media Streams: ``info``, ``start``, ``media``, ``mark`` and ``clear`` envelopes with
base64 audio payloads. The audio is 16-bit little-endian PCM, mono, at 24 kHz by default.

Unlike a phone call, a meeting has several participants, and the gateway never mixes
them: every participant arrives as its own stream, identified by a ``tag``, with the
streams' 20 ms frames interleaved on the one socket. This serializer demultiplexes by
tag. By default it mixes the concurrent streams into the single input stream a
speech-to-text or speech-to-speech model expects; with ``mix_sources=False`` it emits
each stream as a :class:`~pipecat.frames.frames.UserAudioRawFrame` whose ``user_id`` is
the tag, for pipelines that handle speakers separately.

The protocol is specified in the opus-transcriber-proxy repository
(``AGENT_PROTOCOL.md``) and versioned ``MAJOR.MINOR``: both sides ignore unknown events
and fields, so a minor version adds nothing a serializer must handle.
"""

import base64
import json
import time
from collections.abc import Callable

import numpy as np
from loguru import logger

from pipecat.audio.resamplers.base_audio_resampler import BaseAudioResampler
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    UserAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.serializers.base_serializer import FrameSerializer

# The PCM sample rate the gateway speaks by default, in both directions.
JITSI_PCM_SAMPLE_RATE = 24000

# The protocol major version this serializer implements.
SUPPORTED_PROTOCOL_MAJOR = 1

# The gateway sends 20 ms of audio per media frame; the mixer emits in the same unit.
FRAME_MS = 20


class _SourceMixer:
    """Mixes the interleaved per-source PCM16 streams into one, paced by the sources.

    A mixed frame is emitted once every active source has contributed its 20 ms. A
    source that falls ``max_lag_frames`` behind is treated as silent so the others are
    not held back, and one unseen for ``active_window_s`` drops out of the mix (muted,
    or no longer among the speakers the bridge forwards).
    """

    def __init__(
        self,
        sample_rate: int,
        active_window_s: float,
        max_lag_frames: int,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._frame_bytes = sample_rate * 2 * FRAME_MS // 1000
        self._active_window_s = active_window_s
        self._max_lag_bytes = max_lag_frames * self._frame_bytes
        self._clock = clock
        self._buffers: dict[str, bytearray] = {}
        self._last_seen: dict[str, float] = {}

    @property
    def sources(self) -> list[str]:
        """The tags currently in the mix."""
        return list(self._buffers)

    def push(self, tag: str, pcm: bytes) -> bytes:
        """Add a source's PCM and return the mixed PCM that is now ready, possibly empty."""
        now = self._clock()
        self._buffers.setdefault(tag, bytearray()).extend(pcm)
        self._last_seen[tag] = now
        for stale in [
            t for t, seen in self._last_seen.items() if now - seen > self._active_window_s
        ]:
            del self._last_seen[stale]
            del self._buffers[stale]

        out = bytearray()
        while True:
            ready = [b for b in self._buffers.values() if len(b) >= self._frame_bytes]
            if not ready:
                break
            lagging = len(ready) < len(self._buffers)
            if lagging and not any(len(b) >= self._max_lag_bytes for b in ready):
                break
            mixed = np.zeros(self._frame_bytes // 2, dtype=np.int32)
            for buffer in ready:
                mixed += np.frombuffer(bytes(buffer[: self._frame_bytes]), dtype="<i2")
                del buffer[: self._frame_bytes]
            out += np.clip(mixed, -32768, 32767).astype("<i2").tobytes()
        return bytes(out)


class JitsiFrameSerializer(FrameSerializer):
    """Serializer for the Jitsi agent media protocol (``jitsi-agent-media``).

    Converts between Pipecat frames and the JSON envelopes a Jitsi voice-agent gateway
    exchanges with the bot over a WebSocket. Input audio is demultiplexed by participant
    and, by default, mixed into one stream; output audio is sent as the agent's speech.
    An :class:`~pipecat.frames.frames.InterruptionFrame` clears the audio the gateway
    has queued but not yet played (barge-in), and, with ``auto_end`` enabled, an
    :class:`~pipecat.frames.frames.EndFrame` or
    :class:`~pipecat.frames.frames.CancelFrame` tells the gateway the agent is done, so
    it leaves the meeting.
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for JitsiFrameSerializer.

        Parameters:
            jitsi_sample_rate: Sample rate of the PCM on the wire, defaults to 24000 Hz.
                The rate the gateway announces in ``start.mediaFormat`` takes precedence.
            sample_rate: Optional override for pipeline input sample rate.
            auto_end: Whether to send the ``end`` event on EndFrame or CancelFrame, so the
                agent leaves the meeting when the pipeline ends. Defaults to True.
            mix_sources: Whether to mix concurrent participants into one input stream.
                When False, each participant's audio is emitted as a ``UserAudioRawFrame``
                whose ``user_id`` is the participant's tag. Defaults to True.
            mix_active_window_ms: A participant unheard for this long drops out of the
                mix (muted, or no longer forwarded by the bridge). Defaults to 200.
            mix_max_lag_frames: A participant this many 20 ms frames behind the others is
                mixed in as silence rather than holding them back. Defaults to 3.
            ignore_rtvi_messages: Inherited from base FrameSerializer, defaults to True.
        """

        jitsi_sample_rate: int = JITSI_PCM_SAMPLE_RATE
        sample_rate: int | None = None
        auto_end: bool = True
        mix_sources: bool = True
        mix_active_window_ms: int = 200
        mix_max_lag_frames: int = 3

    def __init__(self, params: InputParams | None = None):
        """Initialize the JitsiFrameSerializer.

        Args:
            params: Configuration parameters.
        """
        params = params or JitsiFrameSerializer.InputParams()
        super().__init__(params)
        self._params: JitsiFrameSerializer.InputParams = params

        self._pcm_rate = self._params.jitsi_sample_rate
        self._sample_rate = 0  # Pipeline input rate
        self._info: dict = {}
        self._custom_parameters: dict = {}
        self._end_sent = False
        self._encoding_reported = False
        self._mixer: _SourceMixer | None = None
        # One input resampler per stream: the mixed stream (key None), or each participant.
        self._input_resamplers: dict[str | None, BaseAudioResampler] = {}
        self._output_resampler = create_stream_resampler(
            clear_after_secs=self._params.resampler_clear_after_secs
        )

    @property
    def info(self) -> dict:
        """The gateway's ``info`` handshake, once received.

        Carries the protocol version, the media format, and the ``agentId``,
        ``conference`` and ``meetingId`` of the dial, so one server can host many bots.
        """
        return self._info

    @property
    def custom_parameters(self) -> dict:
        """Provisioning metadata the gateway echoes from the invite, once received."""
        return self._custom_parameters

    async def setup(self, setup: FrameProcessorSetup):
        """Sets up the serializer with pipeline configuration.

        Args:
            setup: Configuration object containing setup parameters.
        """
        self._sample_rate = self._params.sample_rate or setup.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serializes a Pipecat frame to a Jitsi agent media protocol message.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            The JSON text message, or None if the frame isn't handled.
        """
        if (
            self._params.auto_end
            and not self._end_sent
            and isinstance(frame, (EndFrame, CancelFrame))
        ):
            self._end_sent = True
            return json.dumps({"event": "end"})
        elif isinstance(frame, InterruptionFrame):
            # Barge-in: drop the agent audio the gateway has queued but not yet played.
            return json.dumps({"event": "clear"})
        elif isinstance(frame, AudioRawFrame):
            data = await self._output_resampler.resample(
                frame.audio, frame.sample_rate, self._pcm_rate
            )
            if data is None or len(data) == 0:
                return None
            payload = base64.b64encode(data).decode("utf-8")
            # No tag: the gateway attaches the agent's own source.
            return json.dumps({"event": "media", "media": {"payload": payload}})
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserializes a Jitsi agent media protocol message to a Pipecat frame.

        Participant audio becomes an ``InputAudioRawFrame`` (mixed) or a
        ``UserAudioRawFrame`` per participant. The ``info`` and ``start`` handshakes are
        recorded and update the wire sample rate; every other event, known or not, is
        ignored, as the protocol requires.

        Args:
            data: The raw WebSocket message from the gateway.

        Returns:
            The corresponding Pipecat frame, or None if there is nothing to inject.
        """
        try:
            message = json.loads(data)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            logger.warning(f"Failed to parse JSON message: {data!r}")
            return None
        if not isinstance(message, dict):
            return None

        event = message.get("event")

        if event == "media":
            return await self._deserialize_media(message.get("media") or {})
        elif event == "start":
            start = message.get("start") or {}
            self._update_media_format(start.get("mediaFormat"))
            if isinstance(start.get("customParameters"), dict):
                self._custom_parameters = start["customParameters"]
            logger.debug(f"Jitsi participant stream started: tag={start.get('tag')}")
        elif event == "info":
            self._info = message
            self._check_protocol_version(message.get("version"))
            self._update_media_format(message.get("mediaFormat"))
            if isinstance(message.get("customParameters"), dict):
                self._custom_parameters = message["customParameters"]

        # mark (a playback checkpoint echoed back), speakers (the speaker order, a hint),
        # ping/pong and anything unknown: nothing to inject.
        return None

    async def _deserialize_media(self, media: dict) -> Frame | None:
        payload_base64 = media.get("payload")
        if not payload_base64:
            return None
        try:
            payload = base64.b64decode(payload_base64)
        except (ValueError, TypeError):
            return None
        tag = str(media.get("tag", ""))

        if self._params.mix_sources:
            if self._mixer is None:
                self._mixer = _SourceMixer(
                    self._pcm_rate,
                    self._params.mix_active_window_ms / 1000,
                    self._params.mix_max_lag_frames,
                )
            payload = self._mixer.push(tag, payload)
            stream = None
        else:
            stream = tag
        if not payload:
            return None

        resampler = self._input_resamplers.get(stream)
        if resampler is None:
            resampler = create_stream_resampler(
                clear_after_secs=self._params.resampler_clear_after_secs
            )
            self._input_resamplers[stream] = resampler
        payload = await resampler.resample(payload, self._pcm_rate, self._sample_rate)
        if payload is None or len(payload) == 0:
            return None

        if stream is None:
            return InputAudioRawFrame(audio=payload, num_channels=1, sample_rate=self._sample_rate)
        return UserAudioRawFrame(
            audio=payload, num_channels=1, sample_rate=self._sample_rate, user_id=stream
        )

    def _update_media_format(self, media_format: object):
        if not isinstance(media_format, dict):
            return
        rate = media_format.get("sampleRate")
        if isinstance(rate, int) and rate > 0:
            self._pcm_rate = rate
        encoding = media_format.get("encoding", "audio/l16")
        if encoding != "audio/l16" and not self._encoding_reported:
            self._encoding_reported = True
            logger.error(
                f"Jitsi gateway sends {encoding}; this serializer handles audio/l16 only "
                "(invite the agent without a mediaFormat, or with audio/l16)"
            )

    def _check_protocol_version(self, version: object):
        if not isinstance(version, str):
            return
        major = version.split(".", 1)[0]
        if major.isdigit() and int(major) != SUPPORTED_PROTOCOL_MAJOR:
            logger.warning(
                f"Jitsi gateway speaks protocol version {version}, this serializer implements "
                f"major version {SUPPORTED_PROTOCOL_MAJOR}; messages may differ"
            )

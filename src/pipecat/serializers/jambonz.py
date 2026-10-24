#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""jambonz WebSocket frame serializer for audio streaming."""

import json
import time
from typing import Any

from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.serializers.base_serializer import FrameSerializer


class JambonzFrameSerializer(FrameSerializer):
    """Serializer for the jambonz ``listen`` verb's WebSocket protocol.

    jambonz forks call audio to a WebSocket through the ``listen`` verb, with
    ``bidirectionalAudio`` providing the return path. The wire format differs
    from the other telephony serializers in three ways:

    - **Audio has no envelope.** Audio arrives and leaves as raw 16-bit signed
      little-endian mono PCM, not base64 inside a JSON event.
    - **The sample rate is per call.** It is set by the verb's ``sampleRate``
      and then stated by the server in a JSON preamble sent before any audio,
      so it is read from the connection rather than configured here.
    - **Interruptions need a gap.** ``killAudio`` flushes playing and buffered
      audio, but message ordering on the socket is not guaranteed, so audio
      written immediately afterwards can arrive after the flush and play over
      the caller. The serializer withholds audio briefly after sending it.

    Example::

        serializer = JambonzFrameSerializer()

        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(serializer=serializer),
        )

    Docs: https://docs.jambonz.org/verbs/verbs/listen
    """

    class InputParams(FrameSerializer.InputParams):
        """Configuration parameters for JambonzFrameSerializer.

        Parameters:
            jambonz_sample_rate: Override for the rate jambonz is forking at.
                Leave as None so the rate is taken from the connection's
                preamble, which is what the ``listen`` verb actually negotiated.
            sample_rate: Optional override for the pipeline input sample rate.
            kill_audio_guard_secs: How long to withhold outgoing audio after
                sending ``killAudio``, so a frame written in the same moment
                cannot arrive after the flush. Defaults to 0.1.
            resampler_clear_after_secs: Inherited from FrameSerializer. Defaults
                to None here, since callers pause mid-sentence and the resampler
                should not discard its history over an ordinary gap.
        """

        jambonz_sample_rate: int | None = None
        sample_rate: int | None = None
        kill_audio_guard_secs: float = 0.1
        resampler_clear_after_secs: float | None = None

    def __init__(self, params: InputParams | None = None, **kwargs):
        """Initialize the JambonzFrameSerializer.

        Args:
            params: Configuration parameters.
            **kwargs: Additional arguments passed to FrameSerializer.
        """
        params = params or JambonzFrameSerializer.InputParams()
        super().__init__(params, **kwargs)
        self._params: JambonzFrameSerializer.InputParams = params

        # Left unset until the preamble arrives rather than defaulting, so a
        # connection without one fails loudly instead of resampling from a rate
        # that was never negotiated.
        self._jambonz_sample_rate: int | None = params.jambonz_sample_rate
        self._sample_rate: int = 0

        self._call_sid: str | None = None
        self._trace_id: str | None = None
        self._call_meta: dict[str, Any] = {}

        self._guard_until: float = 0.0

        self._input_resampler = create_stream_resampler(
            clear_after_secs=self._params.resampler_clear_after_secs
        )
        self._output_resampler = create_stream_resampler(
            clear_after_secs=self._params.resampler_clear_after_secs
        )

    @property
    def jambonz_sample_rate(self) -> int | None:
        """The rate jambonz is forking this call at, from the preamble."""
        return self._jambonz_sample_rate

    @property
    def call_sid(self) -> str | None:
        """The jambonz call identifier, for correlating with jambonz logs."""
        return self._call_sid

    @property
    def trace_id(self) -> str | None:
        """The jambonz trace identifier, for correlating with jambonz logs."""
        return self._trace_id

    @property
    def call_meta(self) -> dict[str, Any]:
        """The full preamble, for fields this class does not expose."""
        return dict(self._call_meta)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the serializer with pipeline configuration.

        Args:
            setup: Configuration object containing setup parameters.
        """
        self._sample_rate = self._params.sample_rate or setup.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serialize a Pipecat frame to the jambonz WebSocket format.

        Audio is returned as ``bytes``; a ``str`` would be delivered as a text
        frame, which jambonz interprets as a command rather than as audio.

        Args:
            frame: The Pipecat frame to serialize.

        Returns:
            Raw PCM bytes for audio, a JSON string for commands, or None when
            there is nothing to send.
        """
        if isinstance(frame, InterruptionFrame):
            # Opened before returning, so audio produced in the same moment as
            # the flush is already being withheld.
            self._guard_until = time.monotonic() + self._params.kill_audio_guard_secs
            return json.dumps({"type": "killAudio"})

        if isinstance(frame, AudioRawFrame):
            if time.monotonic() < self._guard_until:
                # Discarded rather than queued: this audio belongs to the turn
                # the user interrupted, so delivering it late is worse than not
                # delivering it.
                return None

            if self._jambonz_sample_rate is None:
                logger.warning("Dropping audio: no jambonz audio preamble has arrived yet.")
                return None

            serialized_data = await self._output_resampler.resample(
                frame.audio, frame.sample_rate, self._jambonz_sample_rate
            )
            if not serialized_data:
                # Ignoring in case we don't have audio
                return None
            return serialized_data

        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserialize jambonz WebSocket data to Pipecat frames.

        Text data is the preamble jambonz sends once, before any audio, which
        carries the negotiated audio geometry and the call's identifiers.
        Everything else is raw PCM.

        Args:
            data: The raw WebSocket data from jambonz.

        Returns:
            An InputAudioRawFrame for audio, or None for the preamble and for
            audio that cannot yet be interpreted.
        """
        if isinstance(data, str):
            self._handle_preamble(data)
            return None

        if self._jambonz_sample_rate is None:
            logger.warning("Dropping audio: no jambonz audio preamble has arrived yet.")
            return None

        deserialized_data = await self._input_resampler.resample(
            data, self._jambonz_sample_rate, self._sample_rate
        )
        if not deserialized_data:
            # Ignoring in case we don't have audio
            return None

        return InputAudioRawFrame(
            audio=deserialized_data,
            num_channels=1,
            sample_rate=self._sample_rate,
        )

    def _handle_preamble(self, raw: str) -> None:
        """Read the JSON frame jambonz sends before the audio starts."""
        try:
            meta = json.loads(raw)
        except ValueError:
            logger.warning("Ignoring an undecodable text frame from jambonz.")
            return
        if not isinstance(meta, dict):
            return

        self._call_meta = meta
        self._call_sid = meta.get("callSid")
        self._trace_id = meta.get("traceId")

        rate = meta.get("sampleRate")
        if isinstance(rate, int) and rate > 0 and self._params.jambonz_sample_rate is None:
            self._jambonz_sample_rate = rate
            logger.debug(f"jambonz is forking this call at {rate} Hz.")

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""jambonz WebSocket frame serializer for audio streaming."""

import json
from typing import Any

from loguru import logger

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class JambonzFrameSerializer(FrameSerializer):
    """Serializer for the jambonz ``listen`` verb's WebSocket protocol.

    jambonz forks call audio to a WebSocket through the ``listen`` verb, with
    ``bidirectionalAudio`` providing the return path. The wire format differs
    from the other telephony serializers in five ways:

    - **Audio has no envelope.** Audio arrives and leaves as raw 16-bit signed
      little-endian mono PCM, not base64 inside a JSON event.
    - **Text and audio share the socket.** The first text frame is the
      preamble; later ones are events. With ``passDtmf`` set, a digit arrives
      as ``{"event": "dtmf", "dtmf": "5", ...}``, and ``mark`` and ``playDone``
      replies arrive the same way. Only the first is read as the preamble, so
      an event cannot overwrite the call's identifiers.
    - **The audio geometry is per call.** The verb's ``sampleRate`` and
      ``mixType`` are stated by the server in a JSON preamble sent before any
      audio, so both are read from the connection rather than configured here.
      The return path's rate is not in the preamble.
      Only ``mono`` is handled, and the other two are refused for different
      reasons. ``stereo`` sends both legs as two interleaved channels, which
      read as mono would play at double speed with the channels alternating.
      ``mixed`` is a single channel, so it would be interpreted correctly --
      but it is both legs summed together, which means it carries the bot's
      own audio back to it.
    - **The bot can end the call.** Sending ``disconnect`` ends the ``listen``
      verb, after which jambonz runs the verb's ``actionHook`` if it has one
      and otherwise hangs up. No API credentials are involved, unlike Twilio.
    - **Interruptions need no gap.** ``killAudio`` flushes playing and
      buffered audio, and jambonz reads the socket on a single loop, so binary
      frames sent after it are queued normally rather than racing it. Pipecat's
      transport sends sequentially and drops its own queued output on an
      interruption, so sending ``killAudio`` is sufficient on its own.

    The verb needs three things set explicitly::

        {
            "verb": "listen",
            "url": "wss://your-bot.example.com/ws",
            "sampleRate": 16000,
            "mixType": "mono",
            "bidirectionalAudio": {
                "enabled": true,
                "streaming": true,
                "sampleRate": 16000
            }
        }

    - ``streaming`` defaults to **false**, and binary audio sent back is
      ignored entirely while it is. Nothing reports this; the caller simply
      hears nothing.
    - ``sampleRate`` defaults to **8000**. The other allowed values are 16000,
      24000, 48000 and 64000.
    - ``bidirectionalAudio.sampleRate`` has no default. Omitted, the audio you
      send is read at the call's negotiated codec rate -- 8 kHz for PCMU and
      PCMA, higher for a wideband codec -- so it depends on the carrier and
      should always be stated.

      jambonz reads the audio sent back at ``bidirectionalAudio.sampleRate``,
      whatever ``sampleRate`` is, and nothing on the connection reports it.
      The serializer sends at the ``sampleRate`` it read from the preamble,
      unless ``audio_out_sample_rate`` is set. **So set
      ``bidirectionalAudio.sampleRate`` to the same value as ``sampleRate``,
      or set ``audio_out_sample_rate`` to it.** Mismatched, playback runs fast
      or slow.

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
            audio_in_sample_rate: Override for the rate jambonz is forking at.
                Leave as None so the rate is taken from the connection's
                preamble, which is what the ``listen`` verb actually negotiated.
            audio_out_sample_rate: Rate jambonz reads the audio sent back at, the
                verb's ``bidirectionalAudio.sampleRate``. Leave as None to send at
                the preamble's rate, which is right when the verb sets both rates
                to the same value.
            disconnect_on_end: Whether to send ``disconnect`` when the pipeline
                ends, which ends the ``listen`` verb so jambonz runs the verb's
                ``actionHook`` or hangs up. The jambonz equivalent of Twilio's
                ``auto_hang_up``, and unlike it needs no API credentials.
                Defaults to True.
        """

        disconnect_on_end: bool = True

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
        self._preamble_sample_rate: int | None = None

        self._call_sid: str | None = None
        self._trace_id: str | None = None
        self._call_meta: dict[str, Any] = {}
        # jambonz sends exactly one preamble, immediately, and every text frame
        # after it is an event. Without this, a DTMF digit went through the
        # preamble handler and replaced the call's identifiers with None.
        self._preamble_seen: bool = False
        # EndFrame and CancelFrame can both arrive on one shutdown, and a
        # second disconnect would be sent on a socket jambonz has closed.
        self._disconnect_sent: bool = False
        self._mix_type: str | None = None
        # The preamble arrives once but audio arrives fifty times a second, so
        # an unusable mixType has to be reported once rather than per frame.
        self._warned_mix_type: bool = False

    @property
    def audio_in_sample_rate(self) -> int | None:
        """The rate jambonz is forking this call at, configured or from the preamble."""
        return self._params.audio_in_sample_rate or self._preamble_sample_rate

    @property
    def audio_out_sample_rate(self) -> int | None:
        """The rate jambonz reads the bot's audio at, configured or from the preamble."""
        return self._params.audio_out_sample_rate or self._preamble_sample_rate

    @property
    def mix_type(self) -> str | None:
        """The channel layout jambonz stated in the preamble, if it stated one."""
        return self._mix_type

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
        if (
            self._params.disconnect_on_end
            and not self._disconnect_sent
            and isinstance(frame, (EndFrame, CancelFrame))
        ):
            self._disconnect_sent = True
            return json.dumps({"type": "disconnect"})

        if isinstance(frame, InterruptionFrame):
            await self._reset_audio_out()
            return json.dumps({"type": "killAudio"})

        if isinstance(frame, AudioRawFrame):
            if self.audio_out_sample_rate is None:
                logger.warning("Dropping audio: no jambonz audio preamble has arrived yet.")
                return None

            data = await self._resample_audio_out(frame)
            if not data:
                # Ignoring in case we don't have audio
                return None
            return data

        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Deserialize jambonz WebSocket data to Pipecat frames.

        The first text frame is the preamble, which carries the negotiated
        audio geometry and the call's identifiers. Later text frames are
        events: a DTMF digit becomes an InputDTMFFrame, and anything else is
        ignored. Binary data is raw PCM.

        Args:
            data: The raw WebSocket data from jambonz.

        Returns:
            An InputAudioRawFrame for audio, an InputDTMFFrame for a digit, or
            None for the preamble, for events this does not handle, and for
            audio that cannot yet be interpreted.
        """
        if isinstance(data, str):
            if not self._preamble_seen:
                # Set whether or not it parses: jambonz sends one preamble and
                # it is this frame, so a later event must never be read as one.
                self._preamble_seen = True
                self._handle_preamble(data)
                return None
            return self._handle_event(data)

        if self.audio_in_sample_rate is None:
            logger.warning("Dropping audio: no jambonz audio preamble has arrived yet.")
            return None

        if self._mix_type is not None and self._mix_type != "mono":
            if not self._warned_mix_type:
                self._warned_mix_type = True
                # The two layouts are unusable for different reasons, and the
                # reason is what tells someone which knob to turn.
                why = (
                    "it is both legs summed into one channel, so it carries the bot's own audio "
                    "back to it"
                    if self._mix_type == "mixed"
                    else "it interleaves two channels, which read as mono plays at double speed "
                    "with the channels alternating"
                )
                logger.warning(
                    f"Dropping audio: jambonz is sending mixType={self._mix_type!r} and {why}. "
                    "Set mixType to 'mono' on the listen verb."
                )
            return None

        audio = await self._resample_audio_in(data)
        if not audio:
            # Ignoring in case we don't have audio
            return None

        return InputAudioRawFrame(audio=audio, num_channels=1, sample_rate=self._sample_rate)

    def _handle_event(self, raw: str) -> Frame | None:
        """Read a text frame that arrived after the preamble."""
        try:
            event = json.loads(raw)
        except ValueError:
            logger.debug("Ignoring an undecodable text frame from jambonz.")
            return None
        if not isinstance(event, dict):
            return None

        # The digit is keyed on "event" and carries the digit directly, unlike
        # Twilio's nested {"event": "dtmf", "dtmf": {"digit": ...}}.
        if event.get("event") == "dtmf":
            digit = event.get("dtmf")
            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError:
                logger.debug(
                    f"Ignoring a DTMF digit jambonz sent that is not on a keypad: {digit!r}"
                )
                return None

        # mark and playDone replies, and anything jambonz adds later. Debug
        # rather than warning: these are normal traffic, not faults.
        logger.debug(
            f"Ignoring an unhandled jambonz event: {event.get('type') or event.get('event')!r}"
        )
        return None

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

        mix_type = meta.get("mixType")
        if isinstance(mix_type, str) and mix_type:
            self._mix_type = mix_type

        rate = meta.get("sampleRate")
        if isinstance(rate, int) and rate > 0:
            self._preamble_sample_rate = rate
            logger.debug(f"jambonz is forking this call at {rate} Hz.")

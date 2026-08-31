#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""SIP transport implementation for Pipecat.

:class:`SIPTransport` makes a pipeline a SIP endpoint: the bot registers
to a SIP server (or operates registration-less against a trunk), answers
and places calls, and exchanges audio and DTMF with the pipeline. The
event and method vocabulary is Daily's, verbatim, so a bot written
against :class:`~pipecat.transports.daily.transport.DailyTransport`'s
dial-in/dial-out surface swaps transports by changing only the
constructor and params.

Compatibility with DailyTransport, in three tiers:

**Identical or equivalent (registered and fired):** ``on_dialin_ready``
(payload: the registered address-of-record), ``on_dialin_connected/
stopped/error/warning``, ``on_dialout_connected/answered/stopped/error/
warning`` — payload keys match daily-js's event typings: every dial dict
carries ``sessionId`` and ``sipCallId`` (the SIP Call-ID, the cross-log
correlation key); dial-in adds ``sipFrom`` and ``sipHeaders`` (plus
SIP-only ``sipTo``); dial-out adds ``origin`` and ``destination``;
errors and warnings carry ``errorMsg``. Daily's ``provider`` and
``actionTraceId`` are cloud concepts, omitted rather than faked. Also:
``on_dtmf_event`` (``data["tone"]``), the establishment compound —
``on_first_participant_joined`` → ``on_participant_joined`` →
``on_client_connected`` → ``ClientConnectedFrame`` — and the close pair
``on_participant_left(participant, "leftCall")`` →
``on_client_disconnected``, ``on_participant_updated`` (hold/resume),
``on_connected``, ``on_left``, ``on_before_leave``, ``on_error``,
``on_call_state_updated``. Methods: ``start_dialout``, ``stop_dialout``,
``send_dtmf``, ``sip_call_transfer``, ``sip_refer`` — all return
``str | None`` errors and never raise.

**Equivalent with deltas:** ``sip_refer`` sends a true SIP REFER;
``sip_call_transfer`` is **rejected with an error for now** — Daily's
version re-anchors the call in its cloud, and substituting a REFER
would silently change the semantics (mediated transfer, with the bot
bridging both legs, is the planned faithful implementation).
``on_connected`` fires with no payload (as on LiveKit), while Daily
passes its join data — a cross-transport handler should accept an
optional second argument (``on_connected(transport, data=None)``).
``send_dtmf`` supports both ``"telephone-event"`` and ``"sip-info"``,
but the method is a property of the connection
(``SIPConnection(dtmf_mode=...)``) rather than switchable per request,
and pacing is by inter-digit gap rather than per-digit duration;
``on_dtmf_event`` data omits Daily's ``method`` field (the stack does
not report how a digit arrived). Dial-out ``codecs`` are account-level
(the connection's ``audio_codecs``), not per call; and Daily's
per-dialout ``videoSettings`` (width/height/fps/bitrate) has no
equivalent — video geometry here is runtime-wide on the connection.
Daily's per-dialout ``callerId`` also has no equivalent: the caller ID
presented on outbound calls is the connection's account identity (the
``user`` becomes the From header user part), so a different caller ID
means a different :class:`~pipecat.transports.sip.connection.SIPConnection`.

**Non-mappable (never registered, never faked):** rooms (``on_joined``),
``on_active_speaker_changed``, ``on_app_message`` (no data channel),
``on_transcription_*`` and ``on_recording_*`` (cloud services; the
pipeline does its own STT, and baresip's ``sndfile`` driver covers
recording at the stack level), participant permissions, instance ids.

Video (requires the ``sip-video`` extra for OpenCV): calls made or
answered with a video direction enabled exchange VP8, crossing the
pipeline boundary as RGB ``InputImageRawFrame``/``OutputImageRawFrame``
and converted to/from the call's packed I420 at the transport. The
transport aligns the call's geometry with ``video_out_width/height/
framerate`` at construction; a peer that declines video leaves a
working audio call.

Trunk mode (registration-less): ``SIPConnection(reg_interval=0)`` never
sends REGISTER — dial-out INVITEs go straight to the target (or the
trunk host), with ``auth_user`` for trunks that digest-challenge
INVITEs from a credential list. Consequences: without a registration
binding a SIP server cannot route inbound calls to the bot's AOR, so
dial-in only works when the peer or provider reaches the bot's
listening socket directly (direct IP dialing, or a trunk origination
URI with IP-ACL auth) — which requires a stable, reachable address —
and ``on_dialin_ready`` never fires (it maps from REGISTER OK); treat
``on_connected`` as the ready signal instead. Trunk *mode* is
independent of :attr:`SIPParams.trunk`, which is only the host used to
build ``sip:+E164@host`` dial-out URIs from ``phoneNumber`` settings;
PSTN trunking typically combines both. Provider-specific setup
(Twilio Elastic SIP Trunking, etc.) is documented in baresip-python's
``TRUNKS.md``.

Deployment model: one call per transport instance; the process-wide
runtime, and the user agent and registration for each account, are
shared between instances (see
:mod:`pipecat.transports.sip.connection`).
"""

import asyncio

from loguru import logger

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    ClientConnectedFrame,
    EndFrame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InputImageRawFrame,
    OutputAudioRawFrame,
    OutputDTMFFrame,
    OutputDTMFUrgentFrame,
    OutputImageRawFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.sip.connection import SIPConnection

# How often the input reader polls the call for received audio, and how
# much it will pull per poll (catch-up headroom, in multiples of the
# poll interval).
AUDIO_IN_POLL_SECS = 0.01
AUDIO_IN_CATCHUP = 4

# How often the input reader polls the call for a decoded video frame.
# The call delivers newest-frame semantics, so polling faster than the
# sender's fps just returns None between frames.
VIDEO_IN_POLL_SECS = 1 / 30

# The call's transmit buffer never blocks: writes return the bytes
# accepted and reject the rest, while the native transmit clock drains
# the buffer at exactly real time (padding silence when it runs dry).
# The output transport must therefore pace itself. It keeps the buffer
# filled at most AUDIO_OUT_BUFFER_SECS ahead of the transmit clock —
# enough cushion to ride out event-loop jitter, and small enough that
# an interruption leaves at most this much already-queued audio to play
# out — and retries a rejected remainder every AUDIO_OUT_RETRY_SECS.
AUDIO_OUT_BUFFER_SECS = 0.08
AUDIO_OUT_RETRY_SECS = 0.01

# WARNING: the stream resampler's quality preset directly buys or costs
# conversational latency. Soxr primes its filter before producing any
# output, and the priming is continuous pipeline delay, per direction
# (measured with 20 ms chunks, 16 kHz -> 8 kHz): VHQ ~105 ms, HQ ~96 ms,
# MQ ~92 ms, LQ ~24 ms, QQ ~0 ms. Telephony audio is 300-3400 Hz speech,
# where QQ is audibly indistinguishable — so this transport trades
# inaudible filter quality for ~100 ms less latency in each direction.
# Setting the pipeline rates to the call's rates bypasses resampling
# entirely (equal rates pass audio through untouched).
_RESAMPLER_QUALITY = "QQ"


def _get_cv2():
    """Import OpenCV at point of use; video needs the sip-video extra."""
    try:
        import cv2

        return cv2
    except ModuleNotFoundError as e:
        logger.error(f"Exception: {e}")
        logger.error('In order to use SIP video, you need to `uv add "pipecat-ai[sip-video]"`.')
        raise ImportError(f"Missing module: {e}") from e


def _i420_to_rgb(data: bytes, width: int, height: int) -> bytes:
    """Convert one packed I420 frame to RGB bytes."""
    cv2 = _get_cv2()
    import numpy as np

    yuv = np.frombuffer(data, dtype=np.uint8).reshape((height * 3 // 2, width))
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
    return rgb.tobytes()


def _rgb_to_i420(data: bytes, width: int, height: int) -> bytes:
    """Convert RGB bytes to one packed I420 frame."""
    cv2 = _get_cv2()
    import numpy as np

    rgb = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    i420 = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
    return i420.tobytes()


class SIPParams(TransportParams):
    """Configuration parameters for :class:`SIPTransport`.

    Parameters:
        auto_answer: Answer inbound calls as soon as they arrive. Video
            is accepted when the params enable a video direction.
        trunk: Domain (or host:port) that turns a ``phoneNumber``
            dial-out into ``sip:+E164@trunk``. Without it, dial-out
            requires a full ``sipUri``.
        audio_out_10ms_chunks: Overrides the base default of 4 with 2,
            so outbound audio reaches the call in 20 ms buffers matching
            typical RTP packet time.
    """

    auto_answer: bool = True
    trunk: str | None = None
    audio_out_10ms_chunks: int = 2


class SIPInputTransport(BaseInputTransport):
    """Receives call audio and feeds it into the pipeline."""

    def __init__(self, transport: "SIPTransport", connection: SIPConnection, params: SIPParams):
        """Initialize the input transport.

        Args:
            transport: The parent transport, for event emission.
            connection: The shared SIP connection.
            params: Transport configuration.
        """
        super().__init__(params)
        self._transport = transport
        self._connection = connection
        self._receive_audio_task: asyncio.Task | None = None
        self._receive_video_task: asyncio.Task | None = None
        self._resampler = create_stream_resampler(quality=_RESAMPLER_QUALITY)
        self._streaming = False
        connection.add_event_handler("media_restarted", self._on_media_restarted)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the transport and attach the shared connection."""
        await super().setup(setup)
        await self._connection.connect()

    async def start(self, frame: StartFrame):
        """Mark the transport ready, then start the audio reader."""
        await super().start(frame)
        self._streaming = self._params.audio_in_stream_on_start
        # Readiness must precede the reader: the input audio queue only
        # exists after set_transport_ready().
        await self.set_transport_ready(frame)
        if self._params.audio_in_enabled and self._receive_audio_task is None:
            self._receive_audio_task = self.create_task(self._receive_audio())
        if self._params.video_in_enabled and self._receive_video_task is None:
            self._receive_video_task = self.create_task(self._receive_video())

    async def _start_audio_in_streaming(self):
        """Begin pushing received audio, when not streaming from start."""
        self._streaming = True

    async def stop(self, frame: EndFrame):
        """Stop the reader and release the shared connection."""
        await super().stop(frame)
        await self._teardown()

    async def cancel(self, frame: CancelFrame):
        """Cancel the reader and release the shared connection."""
        await super().cancel(frame)
        await self._teardown()

    async def cleanup(self):
        """Clean up, releasing the shared connection if still held."""
        await super().cleanup()
        await self._teardown()

    async def _teardown(self):
        if self._receive_audio_task is not None:
            await self.cancel_task(self._receive_audio_task)
            self._receive_audio_task = None
        if self._receive_video_task is not None:
            await self.cancel_task(self._receive_video_task)
            self._receive_video_task = None
        await self._transport._before_leave()
        await self._connection.disconnect()

    async def _on_media_restarted(self, connection, kind: str):
        if kind == "audio":
            self._resampler = create_stream_resampler(quality=_RESAMPLER_QUALITY)

    async def _receive_video(self):
        try:
            while True:
                await asyncio.sleep(VIDEO_IN_POLL_SECS)
                frame = self._connection.read_video_frame()
                if frame is None:
                    continue
                rgb = _i420_to_rgb(frame.data, frame.width, frame.height)
                await self.push_video_frame(
                    InputImageRawFrame(image=rgb, size=(frame.width, frame.height), format="RGB")
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self.push_error(f"SIP video reader failed: {e}", exception=e)

    async def _receive_audio(self):
        # Expected conditions (no call, audio not active, renegotiation)
        # are absorbed by the connection's media taps; anything escaping
        # here is unexpected, and dying silently would leave the pipeline
        # deaf — surface it as a pipeline error instead.
        try:
            while True:
                await asyncio.sleep(AUDIO_IN_POLL_SECS)
                if not self._streaming:
                    continue
                info = self._connection.audio_info()
                rx_rate = info.rx_sample_rate if info else 0
                if not rx_rate:
                    continue
                max_bytes = int(rx_rate * AUDIO_IN_POLL_SECS) * 2 * AUDIO_IN_CATCHUP
                pcm = self._connection.read_audio(max_bytes)
                if not pcm:
                    continue
                if rx_rate != self.sample_rate:
                    pcm = await self._resampler.resample(pcm, rx_rate, self.sample_rate)
                    if not pcm:
                        continue
                await self.push_audio_frame(
                    InputAudioRawFrame(audio=pcm, sample_rate=self.sample_rate, num_channels=1)
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await self.push_error(f"SIP audio reader failed: {e}", exception=e)


class SIPOutputTransport(BaseOutputTransport):
    """Writes pipeline audio and DTMF to the call."""

    def __init__(self, transport: "SIPTransport", connection: SIPConnection, params: SIPParams):
        """Initialize the output transport.

        Args:
            transport: The parent transport, for event emission.
            connection: The shared SIP connection.
            params: Transport configuration.
        """
        super().__init__(params)
        self._transport = transport
        self._connection = connection
        self._resampler = create_stream_resampler(quality=_RESAMPLER_QUALITY)
        connection.add_event_handler("media_restarted", self._on_media_restarted)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the transport and attach the shared connection."""
        await super().setup(setup)
        await self._connection.connect()

    async def start(self, frame: StartFrame):
        """Mark the transport ready."""
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Flush the base transport and release the shared connection."""
        await super().stop(frame)
        await self._teardown()

    async def cancel(self, frame: CancelFrame):
        """Tear down immediately and release the shared connection."""
        await super().cancel(frame)
        await self._teardown()

    async def cleanup(self):
        """Clean up, releasing the shared connection if still held."""
        await super().cleanup()
        await self._teardown()

    async def _teardown(self):
        await self._transport._before_leave()
        await self._connection.disconnect()

    async def _on_media_restarted(self, connection, kind: str):
        if kind == "audio":
            self._resampler = create_stream_resampler(quality=_RESAMPLER_QUALITY)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Write one audio buffer to the call, resampling to its rate.

        The call's transmit buffer never blocks and rejects what it
        cannot hold, so this paces the pipeline against the transmit
        clock: excess buffer fill is slept off before writing, and a
        rejected remainder is retried until the call takes it.
        """
        info = self._connection.audio_info()
        if info is None or not info.tx_sample_rate:
            return False
        tx_rate = info.tx_sample_rate
        audio = frame.audio
        if tx_rate != frame.sample_rate:
            audio = await self._resampler.resample(audio, frame.sample_rate, tx_rate)
            if not audio:
                # The stream resampler is still priming; the frame was
                # consumed and its audio arrives with a later chunk.
                return True
        bytes_per_sec = tx_rate * 2 * (info.tx_channels or 1)
        high_water = int(AUDIO_OUT_BUFFER_SECS * bytes_per_sec)
        if info.tx_buffered > high_water:
            await asyncio.sleep((info.tx_buffered - high_water) / bytes_per_sec)
        while audio:
            accepted = self._connection.write_audio(audio)
            audio = audio[accepted:]
            if audio:
                info = self._connection.audio_info()
                if info is None or not info.tx_ready:
                    return False
                await asyncio.sleep(AUDIO_OUT_RETRY_SECS)
        return True

    async def write_video_frame(self, frame: OutputImageRawFrame) -> bool:
        """Write one video frame to the call, converting RGB to I420.

        The base sender has already resized the frame to
        ``video_out_width`` × ``video_out_height``, which the transport
        aligned with the call's geometry at construction.
        """
        if frame.format != "RGB":
            logger.warning(f"{self} dropping video frame with format {frame.format!r} (need RGB)")
            return False
        width, height = frame.size
        i420 = _rgb_to_i420(frame.image, width, height)
        try:
            return self._connection.write_video_frame(i420)
        except ValueError as e:
            # Geometry drifted from the call's configured size.
            logger.warning(f"{self} video frame refused: {e}")
            return False

    def _supports_native_dtmf(self) -> bool:
        return True

    async def _write_dtmf_native(self, frame: OutputDTMFFrame | OutputDTMFUrgentFrame):
        await self._connection.send_dtmf(frame.to_string())


class SIPTransport(BaseTransport):
    """A SIP endpoint as a Pipecat transport.

    The application constructs a
    :class:`~pipecat.transports.sip.connection.SIPConnection` (the
    account and stack settings) and hands it to the transport; the
    transport owns the call's relationship to the pipeline. One call per
    transport instance — run several instances to hold several
    conversations on one account.

    See the module docstring for the DailyTransport compatibility
    matrix.

    Example::

        connection = SIPConnection(user="1001", domain="example.com", password="...")
        transport = SIPTransport(connection, SIPParams(audio_in_enabled=True,
                                                       audio_out_enabled=True))

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, participant):
            ...
    """

    def __init__(
        self,
        connection: SIPConnection,
        params: SIPParams | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize the transport.

        Args:
            connection: The SIP connection to run the call over.
            params: Transport configuration.
            input_name: Optional name for the input processor.
            output_name: Optional name for the output processor.
        """
        super().__init__(input_name=input_name, output_name=output_name)
        self._connection = connection
        self._params = params or SIPParams()
        self._input: SIPInputTransport | None = None
        self._output: SIPOutputTransport | None = None

        self._dial_in_session_id = ""
        self._dial_out_session_id = ""
        self._dialout_progressed = False
        self._other_participant_has_joined = False
        self._left = False

        # The call's video geometry must match what the base sender
        # produces; align the (not yet connected) connection with the
        # params — a late mismatch raises in configure_video.
        if self._params.video_out_enabled:
            connection.configure_video(
                (self._params.video_out_width, self._params.video_out_height),
                float(self._params.video_out_framerate),
            )

        # Register supported handlers. The user will only be able to
        # register these handlers.
        self._register_event_handler("on_connected")
        self._register_event_handler("on_left")
        self._register_event_handler("on_before_leave", sync=True)
        self._register_event_handler("on_error")
        self._register_event_handler("on_call_state_updated")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_dialin_connected")
        self._register_event_handler("on_dialin_ready")
        self._register_event_handler("on_dialin_stopped")
        self._register_event_handler("on_dialin_error")
        self._register_event_handler("on_dialin_warning")
        self._register_event_handler("on_dialout_answered")
        self._register_event_handler("on_dialout_connected")
        self._register_event_handler("on_dialout_stopped")
        self._register_event_handler("on_dialout_error")
        self._register_event_handler("on_dialout_warning")
        self._register_event_handler("on_dtmf_event")
        self._register_event_handler("on_first_participant_joined")
        self._register_event_handler("on_participant_joined")
        self._register_event_handler("on_participant_left")
        self._register_event_handler("on_participant_updated")
        self._register_event_handler("on_call_quality_stats")

        connection.add_event_handler("connected", self._on_connected)
        connection.add_event_handler("disconnected", self._on_disconnected)
        connection.add_event_handler("registered", self._on_registered)
        connection.add_event_handler("incoming", self._on_incoming)
        connection.add_event_handler("call_progress", self._on_call_progress)
        connection.add_event_handler("call_established", self._on_call_established)
        connection.add_event_handler("call_closed", self._on_call_closed)
        connection.add_event_handler("call_failed", self._on_call_failed)
        connection.add_event_handler("dtmf", self._on_dtmf)
        connection.add_event_handler("remote_hold", self._on_remote_hold)
        connection.add_event_handler("audio_warning", self._on_audio_warning)

    def input(self) -> FrameProcessor:
        """The input frame processor, created on first use."""
        if self._input is None:
            self._input = SIPInputTransport(self, self._connection, self._params)
        return self._input

    def output(self) -> FrameProcessor:
        """The output frame processor, created on first use."""
        if self._output is None:
            self._output = SIPOutputTransport(self, self._connection, self._params)
        return self._output

    #
    # Daily-verbatim call control
    #

    async def start_dialout(self, settings=None) -> tuple[str, str | None]:
        """Start a dial-out call.

        Args:
            settings: ``sipUri`` for a direct SIP target, or
                ``phoneNumber`` (with :attr:`SIPParams.trunk`) for a
                PSTN-style target; optional ``video`` (bool) and
                ``headers`` (dict of extra INVITE headers — a SIP-only
                addition).

        Returns:
            A ``(session_id, error)`` pair; ``error`` is None on success.
        """
        settings = settings or {}
        if self._connection.has_active_call:
            return "", "transport already has an active call"
        uri = settings.get("sipUri")
        phone = settings.get("phoneNumber")
        if not uri and phone:
            if not self._params.trunk:
                return "", "phoneNumber dial-out requires SIPParams.trunk"
            uri = f"sip:{phone}@{self._params.trunk}"
        if not uri:
            return "", "settings must include 'sipUri' or 'phoneNumber'"
        try:
            session_id = await self._connection.dial(
                uri,
                headers=settings.get("headers"),
                video=bool(settings.get("video", False)),
            )
        except Exception as e:
            logger.error(f"{self} unable to start dialout: {e}")
            return "", str(e)
        self._dial_out_session_id = session_id
        self._dialout_progressed = False
        return session_id, None

    async def stop_dialout(self, participant_id) -> str | None:
        """Stop a dial-out call.

        Args:
            participant_id: The dial-out call's session id.

        Returns:
            An error description, or None on success.
        """
        if not participant_id or participant_id != self._connection.session_id:
            return "no such dial-out session"
        try:
            await self._connection.hangup()
        except Exception as e:
            return str(e)
        return None

    async def send_dtmf(self, settings) -> str | None:
        """Send DTMF tones on the active call.

        The wire method is a property of the connection's account
        (``SIPConnection(dtmf_mode=...)``), not a per-request choice as
        it is on Daily: a ``method`` in the settings is validated
        against the connection's mode rather than switching it.

        Args:
            settings: ``tones`` (the digits); optional ``sessionId``
                (defaults to the transport's live call) and ``method``
                (``"telephone-event"`` or ``"sip-info"``; must match the
                connection's ``dtmf_mode``).

        Returns:
            An error description, or None on success.
        """
        settings = settings or {}
        session_id = (
            settings.get("sessionId") or self._dial_out_session_id or self._dial_in_session_id
        )
        if not session_id:
            return "Can't send DTMF if 'sessionId' is not set"
        if session_id != self._connection.session_id:
            return "no such session"
        method = settings.get("method")
        if method is not None:
            allowed_modes = {"telephone-event": ("rtpevent", "auto"), "sip-info": ("info", "auto")}
            modes = allowed_modes.get(method)
            if modes is None:
                return f"unknown DTMF method {method!r}"
            if self._connection.dtmf_mode not in modes:
                return (
                    f"connection is configured for {self._connection.dtmf_mode!r} DTMF; "
                    f"construct SIPConnection(dtmf_mode=...) to use {method!r}"
                )
        tones = settings.get("tones")
        if not tones:
            return "Can't send DTMF if 'tones' is not set"
        try:
            await self._connection.send_dtmf(tones)
        except Exception as e:
            return str(e)
        return None

    async def sip_call_transfer(self, settings) -> str | None:
        """Transfer the call the way Daily's infrastructure would; not supported yet.

        Daily's ``sip_call_transfer`` re-anchors the call inside its
        cloud, needing nothing from the far end. A user agent has no
        such infrastructure, and quietly substituting a REFER (which
        requires far-end support and different semantics) would mislead;
        until mediated transfer lands — the bot bridging both legs
        itself — this returns an error. Use :meth:`sip_refer` for a SIP
        REFER, or ``SIPConnection.attended_transfer`` for a
        Replaces-based splice.

        Args:
            settings: Ignored until mediated transfer lands.

        Returns:
            The not-supported error description.
        """
        return (
            "sip_call_transfer is not supported yet: a SIP user agent has no cloud to "
            "re-anchor media through. Use sip_refer (SIP REFER), or wait for mediated "
            "transfer."
        )

    async def sip_refer(self, settings) -> str | None:
        """Send a SIP REFER for the call.

        Args:
            settings: ``toEndPoint`` (the destination URI); optional
                ``sessionId``.

        Returns:
            An error description, or None on success.
        """
        return await self._refer(settings)

    async def request_keyframe(self):
        """Ask the far end for a video keyframe (SIP-only addition).

        Useful for a consumer joining mid-stream that wants a decodable
        starting point sooner than the next natural keyframe.
        """
        await self._connection.request_keyframe()

    async def _refer(self, settings) -> str | None:
        settings = settings or {}
        session_id = (
            settings.get("sessionId") or self._dial_out_session_id or self._dial_in_session_id
        )
        if not session_id:
            return "Can't transfer SIP call if 'sessionId' is not set"
        if session_id != self._connection.session_id:
            return "no such session"
        to_end_point = settings.get("toEndPoint")
        if not to_end_point:
            return "Can't transfer SIP call if 'toEndPoint' is not set"
        try:
            await self._connection.transfer(to_end_point)
        except Exception as e:
            return str(e)
        return None

    #
    # Connection event translation
    #

    def _participant(self, data: dict) -> dict:
        user_name = data.get("displayName") or data.get("sipFrom") or data.get("destination")
        return {"id": data.get("sessionId"), "info": {"userName": user_name}}

    async def _before_leave(self):
        if self._left:
            return
        self._left = True
        await self._call_event_handler("on_before_leave")

    async def _on_connected(self, connection):
        await self._call_event_handler("on_connected")

    async def _on_disconnected(self, connection):
        await self._call_event_handler("on_left")

    async def _on_registered(self, connection, aor: str):
        await self._call_event_handler("on_dialin_ready", aor)

    async def _on_incoming(self, connection, data: dict):
        self._dial_in_session_id = data.get("sessionId") or ""
        if not self._params.auto_answer:
            return
        video = self._params.video_in_enabled or self._params.video_out_enabled
        try:
            await connection.answer(video=video)
        except Exception as e:
            logger.error(f"{self} unable to answer incoming call: {e}")
            await self._call_event_handler("on_dialin_error", self._error_data(data, str(e)))

    def _dialout_data(self, data: dict) -> dict:
        return {
            "sessionId": data.get("sessionId"),
            "sipCallId": data.get("sipCallId"),
            "destination": data.get("destination"),
        }

    async def _on_call_progress(self, connection, data: dict):
        await self._call_event_handler("on_call_state_updated", "ringing")
        if data.get("direction") == "out" and not self._dialout_progressed:
            self._dialout_progressed = True
            await self._call_event_handler("on_dialout_connected", self._dialout_data(data))

    async def _on_call_established(self, connection, data: dict):
        await self._call_event_handler("on_call_state_updated", "established")
        if data.get("direction") == "in":
            await self._call_event_handler("on_dialin_connected", data)
        else:
            await self._call_event_handler("on_dialout_answered", self._dialout_data(data))
        participant = self._participant(data)
        if not self._other_participant_has_joined:
            self._other_participant_has_joined = True
            await self._call_event_handler("on_first_participant_joined", participant)
        await self._call_event_handler("on_participant_joined", participant)
        await self._call_event_handler("on_client_connected", participant)
        if self._input:
            await self._input.push_frame(ClientConnectedFrame())

    async def _on_call_closed(self, connection, data: dict):
        await self._call_event_handler("on_call_state_updated", "closed")
        stats = connection.final_stats
        if stats is not None:
            await self._call_event_handler("on_call_quality_stats", stats)
        if data.get("direction") == "in":
            self._dial_in_session_id = ""
            stopped = {k: v for k, v in data.items() if k not in ("established", "direction")}
            await self._call_event_handler("on_dialin_stopped", stopped)
        else:
            self._dial_out_session_id = ""
            if data.get("established"):
                stopped = self._dialout_data(data)
                stopped["reason"] = data.get("reason", "")
                await self._call_event_handler("on_dialout_stopped", stopped)
        if data.get("established"):
            participant = self._participant(data)
            await self._call_event_handler("on_participant_left", participant, "leftCall")
            await self._call_event_handler("on_client_disconnected", participant)

    async def _on_call_failed(self, connection, data: dict):
        self._dial_out_session_id = ""
        error = {
            "sessionId": data.get("sessionId"),
            "errorMsg": data.get("message", ""),
            "error": data.get("error"),
        }
        await self._call_event_handler("on_dialout_error", error)

    async def _on_dtmf(self, connection, digit_event):
        # No "method" key: the binding's DigitEvent does not say how the
        # digit arrived, and guessing would fake Daily's field.
        data = {"sessionId": self._connection.session_id, "tone": digit_event.digit}
        await self._call_event_handler("on_dtmf_event", data)
        try:
            button = KeypadEntry(digit_event.digit)
        except ValueError:
            # RFC 4733 defines A-D, but KeypadEntry (and every dialpad)
            # does not.
            logger.debug(f"{self} dropping non-keypad DTMF digit: {digit_event.digit!r}")
            return
        if self._input:
            await self._input.push_frame(InputDTMFFrame(button=button))

    async def _on_remote_hold(self, connection, data: dict):
        participant = self._participant(data)
        participant["media"] = {"onHold": data.get("on", False)}
        await self._call_event_handler("on_participant_updated", participant)

    async def _on_audio_warning(self, connection, warning):
        data = {"sessionId": self._connection.session_id, "errorMsg": str(warning)}
        if self._connection.call_direction == "in":
            await self._call_event_handler("on_dialin_warning", data)
        else:
            await self._call_event_handler("on_dialout_warning", data)

    def _error_data(self, data: dict, message: str) -> dict:
        return {"sessionId": data.get("sessionId"), "errorMsg": message}

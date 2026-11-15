#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Keyframe transport for Pipecat.

This module adds Keyframe avatars to Daily rooms, enabling real-time voice
conversations with synchronized avatars.
"""

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from typing import Any

import aiohttp
from daily.daily import AudioData
from loguru import logger
from pydantic import BaseModel, model_validator

from pipecat.expressive.expression import (
    EMPHASIS_DELIVERY,
    FAST_DELIVERY,
    SLOW_DELIVERY,
    ExpressionDelivery,
)
from pipecat.expressive.policy import ExpressionPolicy
from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    ControlFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
)
from pipecat.pipeline.capabilities import BotCapabilities
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import (
    DailyCallbacks,
    DailyOutputTransportMessageUrgentFrame,
    DailyParams,
    DailyTransportClient,
)
from pipecat.transports.keyframe.api import KeyframeApi
from pipecat.transports.keyframe.protocol import (
    SAMPLE_RATE,
    audio_message,
    audio_record,
    cue_record,
    emotion_now_message,
    interrupt_message,
    message_type,
)
from pipecat.utils.shared import acquires, releases
from pipecat.utils.time import nanoseconds_to_seconds

# How long after a turn's audio should have played to wait for the avatar's
# playback report before pushing the held BotStoppedSpeakingFrame anyway.
PLAYBACK_GRACE_SECS = 5.0
# How often a BotSpeakingFrame is pushed while a BotStoppedSpeakingFrame is held.
BOT_SPEAKING_PERIOD_SECS = 0.2
# Messages held for an avatar that is not ready yet (a minute of audio).
MAX_PENDING_MESSAGES = 1500


class KeyframeEmotion(StrEnum):
    """A facial emotion the Keyframe avatar can express."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    CONCERNED = "concerned"
    CONFUSED = "confused"
    DISGUSTED = "disgusted"
    EXCITED = "excited"
    SCARED = "scared"
    SKEPTICAL = "skeptical"


# The Keyframe emotion each delivery converts to: the avatar's emotions by their
# own names, Pipecat's core deliveries (``CORE_DELIVERY``), and the plainest
# deliveries of each emotion those leave out. Any other delivery, and the
# voice's baseline, converts to neutral.
KEYFRAME_DELIVERY_EMOTIONS: dict[str, KeyframeEmotion] = {
    **{emotion.value: emotion for emotion in KeyframeEmotion},
    ExpressionDelivery.CALM: KeyframeEmotion.NEUTRAL,
    ExpressionDelivery.CURIOUS: KeyframeEmotion.NEUTRAL,
    ExpressionDelivery.CONFIDENT: KeyframeEmotion.NEUTRAL,
    ExpressionDelivery.WARM: KeyframeEmotion.HAPPY,
    ExpressionDelivery.GRATEFUL: KeyframeEmotion.HAPPY,
    ExpressionDelivery.SYMPATHETIC: KeyframeEmotion.CONCERNED,
    ExpressionDelivery.APOLOGETIC: KeyframeEmotion.CONCERNED,
    ExpressionDelivery.WORRIED: KeyframeEmotion.CONCERNED,
    ExpressionDelivery.FEARFUL: KeyframeEmotion.SCARED,
    ExpressionDelivery.DOUBTFUL: KeyframeEmotion.SKEPTICAL,
    ExpressionDelivery.UNCERTAIN: KeyframeEmotion.CONFUSED,
}


def keyframe_policy_warning(policy: ExpressionPolicy | None) -> str | None:
    """Tell which of an expression policy's deliveries the avatar has no emotion for.

    The avatar shows neutral when the bot speaks in one of them.

    Args:
        policy: The pipeline's expression policy, or None for the default.

    Returns:
        The warning, or None when the avatar has an emotion for every delivery
        the policy offers, as it has for the core ones offered without a list.
        Slow and fast set the pace, not an emotion, so they're never named.
    """
    if policy is None or policy.delivery is None:
        return None
    handled = {*KEYFRAME_DELIVERY_EMOTIONS, SLOW_DELIVERY, FAST_DELIVERY}
    unmapped = [name for name in policy.delivery if name not in handled]
    if not unmapped:
        return None
    return (
        f"the Keyframe avatar has no emotion for {', '.join(unmapped)} from the expression "
        "policy, so it shows neutral when the bot speaks in them"
    )


def _delivery_emotion(delivery: list[str] | None) -> KeyframeEmotion | None:
    """The emotion of a delivery's mood, neutral without one, or None to keep the current one."""
    if delivery is None:
        return None
    mood = next(
        (
            name
            for name in delivery
            if name not in (EMPHASIS_DELIVERY, SLOW_DELIVERY, FAST_DELIVERY)
        ),
        None,
    )
    if mood is None:
        return KeyframeEmotion.NEUTRAL
    return KEYFRAME_DELIVERY_EMOTIONS.get(mood, KeyframeEmotion.NEUTRAL)


class KeyframeNewSessionRequest(BaseModel):
    """Request model for creating a new Keyframe session.

    Parameters:
        persona_id: ID of the Keyframe persona. Provide exactly one of ``persona_id``
            or ``persona_slug``.
        persona_slug: Slug of the Keyframe persona, such as ``public:cosmo_persona-2.0-live``. Provide
            exactly one of ``persona_id`` or ``persona_slug``.
        daily_room_url: Daily room URL to use for the session. When unset, Keyframe
            creates a private Daily room for the session.
        daily_token: Daily token the bot joins ``daily_room_url`` with.
        daily_user_id: The ``user_id`` of ``daily_token``. With it the avatar
            speaks only the audio of the participant with that ID. Without it
            the avatar goes by the bot's user name, which any participant of
            the room can change theirs to.
        avatar_daily_token: Daily token the avatar joins ``daily_room_url`` with.
            Required when that room is private.
        api_url: Override the Keyframe API URL.
    """

    persona_id: str | None = None
    persona_slug: str | None = None
    daily_room_url: str | None = None
    daily_token: str | None = None
    daily_user_id: str | None = None
    avatar_daily_token: str | None = None
    api_url: str | None = None

    @model_validator(mode="after")
    def validate_request(self) -> "KeyframeNewSessionRequest":
        """Ensure exactly one persona is given, and tokens only with a room URL."""
        if (self.persona_id is None) == (self.persona_slug is None):
            raise ValueError("Provide exactly one of persona_id or persona_slug")
        if self.daily_room_url is None and (
            self.daily_token or self.daily_user_id or self.avatar_daily_token
        ):
            raise ValueError(
                "daily_token, daily_user_id and avatar_daily_token require daily_room_url"
            )
        if self.daily_user_id and not self.daily_token:
            raise ValueError("daily_user_id is the user_id of daily_token; provide both")
        return self


class KeyframeSession(BaseModel):
    """A Keyframe session and the Daily room it runs in.

    Parameters:
        reservation_id: ID of the Keyframe session.
        model: The Keyframe model rendering the persona.
        room_url: URL of the Daily room.
        avatar_participant_name: User name of the avatar's participant in the room.
        avatar_participant_id: User ID of the avatar's participant, when Keyframe
            created the room. Only the avatar can join with it.
        bot_token: Daily token the bot joins the room with, or None for a public room.
        user_token: Daily token for the user to join the room with, when Keyframe
            created the room. The room is private, so a client needs this token
            along with ``room_url``.
    """

    reservation_id: str
    model: str | None = None
    room_url: str
    avatar_participant_name: str
    avatar_participant_id: str | None = None
    bot_token: str | None = None
    user_token: str | None = None


class KeyframeCallbacks(BaseModel):
    """Callback handlers for Keyframe events.

    Parameters:
        on_session_created: Called when the Keyframe session has been created.
        on_participant_joined: Called when a participant joins the conversation.
        on_participant_left: Called when a participant leaves the conversation.
    """

    on_session_created: Callable[[KeyframeSession], Awaitable[None]]
    on_participant_joined: Callable[[Mapping[str, Any]], Awaitable[None]]
    on_participant_left: Callable[[Mapping[str, Any], str], Awaitable[None]]


class KeyframeParams(DailyParams):
    """Configuration parameters for the Keyframe transport.

    Parameters:
        audio_in_enabled: Whether to enable audio input from participants.
        audio_out_enabled: Whether to enable audio output to participants.
        microphone_out_enabled: Whether to enable microphone output track.
        audio_out_sample_rate: Output audio sample rate in Hz. The avatar expects
            24000, so this must be left at its default.
        audio_out_end_silence_secs: How much silence to send after an EndFrame.
    """

    audio_in_enabled: bool = True
    audio_out_enabled: bool = True
    microphone_out_enabled: bool = False
    audio_out_sample_rate: int | None = SAMPLE_RATE
    audio_out_end_silence_secs: int = 0


class KeyframeTransportClient:
    """Transport client that integrates Pipecat with the Keyframe platform.

    A transport client that integrates a Pipecat Bot with the Keyframe platform by
    creating a session using the Keyframe API and joining its Daily room.

    This client uses `KeyframeApi` to interact with the Keyframe backend. Keyframe adds
    its avatar to a Daily room the user supplies, or to a room it creates for the
    session. The bot's audio reaches the avatar as Daily app messages, which are held
    until the avatar has joined the room and reported that it is ready.
    """

    def __init__(
        self,
        *,
        bot_name: str,
        params: KeyframeParams | None = None,
        callbacks: KeyframeCallbacks,
        api_key: str,
        session_request: KeyframeNewSessionRequest,
        session: aiohttp.ClientSession,
    ) -> None:
        """Initialize the Keyframe transport client.

        Args:
            bot_name: The name of the Pipecat bot instance.
            params: Optional parameters for Keyframe operation.
            callbacks: Callback handlers for Keyframe-related events.
            api_key: API key for authenticating with Keyframe API.
            session_request: Session creation parameters.
            session: The aiohttp session for making async HTTP requests.
        """
        self._bot_name = bot_name
        self._api = KeyframeApi(api_key, session)
        self._session_request = session_request
        self._session: KeyframeSession | None = None
        self._daily_transport_client: DailyTransportClient | None = None
        self._callbacks = callbacks
        self._params = params or KeyframeParams()

        self._avatar_id: str | None = None
        # Participants that have sent a hello message.
        self._ready_ids: set[str] = set()
        # Messages for the avatar, until it is ready.
        self._pending: deque[dict[str, Any]] = deque()
        self._pending_dropped = False
        self._sending_pending = False
        # Called with a turn's name when the avatar reports it played to its end.
        self.on_playback_finished: Callable[[str], Awaitable[None]] | None = None

    async def _initialize(self) -> KeyframeSession:
        """Create the Keyframe session."""
        request = self._session_request
        response = await self._api.create_session(
            source_participant_name=self._bot_name,
            persona_id=request.persona_id,
            persona_slug=request.persona_slug,
            daily_room_url=request.daily_room_url,
            avatar_daily_token=request.avatar_daily_token,
            source_participant_id=request.daily_user_id,
            api_url=request.api_url,
        )
        return KeyframeSession(
            reservation_id=response["reservation_id"],
            model=response.get("model_id"),
            room_url=response["room_url"],
            avatar_participant_name=response["avatar_participant_name"],
            avatar_participant_id=response.get("avatar_participant_id"),
            bot_token=response.get("bot_token") or request.daily_token,
            user_token=response.get("user_token"),
        )

    @acquires("client")
    async def setup(self, setup: FrameProcessorSetup):
        """Setup the client and create the session.

        Args:
            setup: The frame processor setup configuration.
        """
        try:
            self._session = await self._initialize()
            await self._callbacks.on_session_created(self._session)
            daily_callbacks = DailyCallbacks(
                on_active_speaker_changed=partial(
                    self._on_handle_callback, "on_active_speaker_changed"
                ),
                on_joined=self._on_joined,
                on_left=self._on_left,
                on_before_leave=partial(self._on_handle_callback, "on_before_leave"),
                on_error=partial(self._on_handle_callback, "on_error"),
                on_app_message=self._on_app_message,
                on_call_state_updated=partial(self._on_handle_callback, "on_call_state_updated"),
                on_client_connected=partial(self._on_handle_callback, "on_client_connected"),
                on_client_disconnected=partial(self._on_handle_callback, "on_client_disconnected"),
                on_dialin_connected=partial(self._on_handle_callback, "on_dialin_connected"),
                on_dialin_ready=partial(self._on_handle_callback, "on_dialin_ready"),
                on_dialin_stopped=partial(self._on_handle_callback, "on_dialin_stopped"),
                on_dialin_error=partial(self._on_handle_callback, "on_dialin_error"),
                on_dialin_warning=partial(self._on_handle_callback, "on_dialin_warning"),
                on_dialout_answered=partial(self._on_handle_callback, "on_dialout_answered"),
                on_dialout_connected=partial(self._on_handle_callback, "on_dialout_connected"),
                on_dialout_stopped=partial(self._on_handle_callback, "on_dialout_stopped"),
                on_dialout_error=partial(self._on_handle_callback, "on_dialout_error"),
                on_dialout_warning=partial(self._on_handle_callback, "on_dialout_warning"),
                on_dtmf_event=partial(self._on_handle_callback, "on_dtmf_event"),
                on_network_connection_status_updated=partial(
                    self._on_handle_callback, "on_network_connection_status_updated"
                ),
                on_participant_joined=self._callbacks.on_participant_joined,
                on_participant_left=self._callbacks.on_participant_left,
                on_participant_updated=partial(self._on_handle_callback, "on_participant_updated"),
                on_transcription_message=partial(
                    self._on_handle_callback, "on_transcription_message"
                ),
                on_recording_started=partial(self._on_handle_callback, "on_recording_started"),
                on_recording_stopped=partial(self._on_handle_callback, "on_recording_stopped"),
                on_recording_error=partial(self._on_handle_callback, "on_recording_error"),
                on_transcription_stopped=partial(
                    self._on_handle_callback, "on_transcription_stopped"
                ),
                on_transcription_error=partial(self._on_handle_callback, "on_transcription_error"),
            )
            self._daily_transport_client = DailyTransportClient(
                self._session.room_url,
                self._session.bot_token,
                self._bot_name,
                self._params,
                daily_callbacks,
                "KeyframePipecat",
            )
            await self._daily_transport_client.setup(setup)
        except Exception as e:
            logger.error(f"Failed to setup KeyframeTransportClient: {e}")
            raise

    @releases("client")
    async def cleanup(self):
        """Cleanup client resources."""
        if not self._daily_transport_client:
            return

        try:
            await self._daily_transport_client.cleanup()
        except Exception as e:
            logger.error(f"Exception during cleanup: {e}")

    async def _on_joined(self, data):
        """Handle joined event."""
        logger.debug("KeyframeTransportClient joined!")

    async def _on_left(self):
        """Handle left event."""
        logger.debug("KeyframeTransportClient left!")

    async def _on_handle_callback(self, event_name, *args, **kwargs):
        """Handle generic callback events."""
        logger.trace(f"[Callback] {event_name} called with args={args}, kwargs={kwargs}")

    @property
    def session(self) -> KeyframeSession | None:
        """Get the Keyframe session.

        Returns:
            The session, or None before setup() has run.
        """
        return self._session

    @acquires("room")
    async def join(self):
        """Join the Daily room of the session."""
        if not self._daily_transport_client:
            return

        await self._daily_transport_client.join()

    @releases("room")
    async def stop(self):
        """Stop the client and leave the Daily room.

        Runs once the last of the transports sharing this client has stopped,
        so the bot stays in the room while the output transport flushes its
        audio. The Keyframe session ends when the bot leaves the room.
        """
        if not self._daily_transport_client:
            return

        await self._daily_transport_client.leave()

    async def capture_participant_audio(
        self,
        participant_id: str,
        callback: Callable,
        audio_source: str = "microphone",
        sample_rate: int = 16000,
        callback_interval_ms: int = 20,
    ):
        """Capture audio from a participant.

        Args:
            participant_id: ID of the participant to capture audio from.
            callback: Callback function to handle audio data.
            audio_source: Audio source to capture from.
            sample_rate: Desired sample rate for audio capture.
            callback_interval_ms: Interval between audio callbacks in milliseconds.
        """
        if not self._daily_transport_client:
            return

        await self._daily_transport_client.capture_participant_audio(
            participant_id, callback, audio_source, sample_rate, callback_interval_ms
        )

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a message to participants.

        Args:
            frame: The message frame to send.
        """
        if self._daily_transport_client is None:
            return

        await self._daily_transport_client.send_message(frame)

    @property
    def in_sample_rate(self) -> int:
        """Get the input sample rate.

        Returns:
            The input sample rate in Hz.
        """
        if not self._daily_transport_client:
            # No client until setup() runs; 0 is what Daily reports before starting.
            return 0

        return self._daily_transport_client.in_sample_rate

    async def update_subscriptions(self, participant_settings=None, profile_settings=None):
        """Update subscription settings for participants.

        Args:
            participant_settings: Per-participant subscription settings.
            profile_settings: Global subscription profile settings.
        """
        if not self._daily_transport_client:
            return

        await self._daily_transport_client.update_subscriptions(
            participant_settings=participant_settings, profile_settings=profile_settings
        )

    def is_avatar(self, participant: Mapping[str, Any]) -> bool:
        """Check whether a participant is the Keyframe avatar.

        Args:
            participant: A participant of the Daily room.

        Returns:
            True if the participant has the avatar's user ID, in a room Keyframe
            created. In any other room, True if it has the avatar's user name,
            which a participant can change theirs to.
        """
        if self._session is None:
            return False
        info = participant.get("info", {})
        if self._session.avatar_participant_id is not None:
            return info.get("userId") == self._session.avatar_participant_id
        return info.get("userName", "") == self._session.avatar_participant_name

    async def avatar_joined(self, participant_id: str):
        """Record the avatar's participant, and send it any held messages if it is ready.

        Args:
            participant_id: ID of the avatar's participant.
        """
        self._avatar_id = participant_id
        await self._send_pending()

    def avatar_left(self):
        """Record that the avatar left the room."""
        if self._avatar_id:
            self._ready_ids.discard(self._avatar_id)
        self._avatar_id = None

    @property
    def _avatar_ready(self) -> bool:
        return self._avatar_id is not None and self._avatar_id in self._ready_ids

    async def send_to_avatar(self, message: dict[str, Any]):
        """Send an app message to the avatar, holding it until the avatar is ready.

        Args:
            message: The message payload.
        """
        # Behind any held messages, which are still being sent: the avatar
        # expects a turn's messages in order.
        if self._avatar_ready and not self._pending:
            await self._send_now(message)
            return
        if len(self._pending) >= MAX_PENDING_MESSAGES:
            self._pending.popleft()
            if not self._pending_dropped:
                self._pending_dropped = True
                logger.warning(
                    "The Keyframe avatar is not ready; dropping the oldest messages held for it"
                )
        self._pending.append(message)
        await self._send_pending()

    async def send_interrupt_message(self) -> None:
        """Send an interrupt message to the avatar, discarding any held messages."""
        logger.debug("Sending interrupt message")
        self._pending.clear()
        if self._avatar_ready:
            await self._send_now(interrupt_message())

    async def _send_now(self, message: dict[str, Any]):
        # Addressed to the avatar, so other participants don't receive the audio.
        await self.send_message(
            DailyOutputTransportMessageUrgentFrame(message=message, participant_id=self._avatar_id)
        )

    async def _send_pending(self):
        """Send the held messages in order, unless they are being sent already."""
        if not self._avatar_ready or self._sending_pending:
            return
        self._sending_pending = True
        try:
            self._pending_dropped = False
            # Stops if the avatar leaves, so nothing held is sent to the room.
            while self._pending and self._avatar_ready:
                await self._send_now(self._pending.popleft())
        finally:
            self._sending_pending = False

    async def _on_app_message(self, message: Any, sender: str):
        """Handle app messages from the avatar."""
        kind = message_type(message)
        if kind == "hello":
            # The hello can arrive before the avatar's participant is known.
            self._ready_ids.add(sender)
            await self._send_pending()
        elif kind == "playback_finished" and sender == self._avatar_id:
            turn = message.get("turn")
            if isinstance(turn, str) and not message.get("interrupted"):
                if self.on_playback_finished:
                    await self.on_playback_finished(turn)


class KeyframeInputTransport(BaseInputTransport):
    """Input transport for receiving audio and events from Keyframe.

    Handles incoming audio streams from participants and manages audio capture
    from the Daily room connected to Keyframe.
    """

    def __init__(
        self,
        client: KeyframeTransportClient,
        params: TransportParams,
        **kwargs,
    ):
        """Initialize the Keyframe input transport.

        Args:
            client: The Keyframe transport client instance.
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params

    async def setup(self, setup: FrameProcessorSetup):
        """Setup the input transport.

        Args:
            setup: The frame processor setup configuration.
        """
        await super().setup(setup)
        await self._client.setup(setup)
        await self._client.join()

    async def cleanup(self):
        """Cleanup input transport resources."""
        await super().cleanup()
        await self._client.cleanup()

    async def start(self, frame: StartFrame):
        """Start the input transport.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the input transport.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._client.stop()

    async def cancel(self, frame: CancelFrame):
        """Cancel the input transport.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._client.stop()

    async def start_capturing_audio(self, participant):
        """Start capturing audio from a participant.

        Args:
            participant: The participant to capture audio from.
        """
        if self._params.audio_in_enabled:
            logger.debug(
                f"KeyframeTransportClient start capturing audio for participant {participant['id']}"
            )
            await self._client.capture_participant_audio(
                participant_id=participant["id"],
                callback=self._on_participant_audio_data,
                sample_rate=self._client.in_sample_rate,
            )

    async def _on_participant_audio_data(
        self, participant_id: str, audio: AudioData, audio_source: str
    ):
        """Handle received participant audio data.

        Args:
            participant_id: ID of the participant who sent the audio.
            audio: The audio data from the participant.
            audio_source: The source of the audio (e.g., microphone).
        """
        frame = InputAudioRawFrame(
            audio=audio.audio_frames,
            sample_rate=audio.sample_rate,
            num_channels=audio.num_channels,
        )
        frame.transport_source = audio_source
        await self.push_audio_frame(frame)


@dataclass
class _ContextStartFrame(ControlFrame):
    """Marks the start of a TTS context's audio in the output audio queue.

    Parameters:
        context_id: The TTS context.
    """

    context_id: str | None = None


@dataclass
class _EmotionCueFrame(ControlFrame):
    """An emotion change queued in order with the output audio.

    Parameters:
        emotion: The emotion to change to.
        context_id: The TTS context the change is timed against.
        offset_secs: Seconds from the start of that context's audio, or None to
            change at the audio sent so far.
    """

    emotion: str = KeyframeEmotion.NEUTRAL
    context_id: str | None = None
    offset_secs: float | None = None


@dataclass
class _Turn:
    """One run of bot speech sent to the avatar."""

    name: str
    seq: int = 0
    samples: int = 0
    started_at: float = field(default_factory=time.monotonic)
    # When the avatar should have played the turn, known once all of it is sent.
    plays_until: float = 0.0
    # BotStoppedSpeakingFrames held until the avatar has played the turn.
    held: list[tuple[Frame, FrameDirection]] = field(default_factory=list)
    wait_task: asyncio.Task | None = None


class KeyframeOutputTransport(BaseOutputTransport):
    """Output transport for sending audio and events to Keyframe.

    Sends the bot's audio to the avatar as Daily app messages instead of
    publishing it to the room; the avatar publishes the audio along with its
    video. Audio is sent as soon as it is available, without pacing it to
    playback time, so ``BotStoppedSpeakingFrame`` is pushed when the avatar
    reports that it finished playing the audio, not when the audio was sent.

    The avatar's emotion follows the delivery of the bot's speech
    (``TTSTextFrame.delivery``), and changes on the word where the delivery
    changes.
    """

    def __init__(
        self,
        client: KeyframeTransportClient,
        params: TransportParams,
        **kwargs,
    ):
        """Initialize the Keyframe output transport.

        Args:
            client: The Keyframe transport client instance.
            params: Transport configuration parameters.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params
        self._client.on_playback_finished = self._on_playback_finished

        # The turn being sent, and the turns sent that the avatar is still playing.
        self._turn_count = 0
        self._turn: _Turn | None = None
        self._playing: dict[str, _Turn] = {}
        # When the avatar should have played every turn sent so far.
        self._plays_until = 0.0
        # Emotion cue records that arrived before the turn's audio, to send
        # with its first message.
        self._cues: list[bytes] = []
        # Where each TTS context's audio starts, in samples of the turn sent.
        self._context_starts: dict[str | None, int] = {}
        # True while an interruption is being handled, when audio the base
        # class still holds must not be sent.
        self._interrupting = False

        # The TTS context whose audio is arriving and the clock time its word
        # timestamps count from.
        self._context_id: str | None = None
        self._context_start_pts = 0
        self._last_word_pts = 0
        self._last_uninterruptible_word_pts = 0
        # The emotion of the last cue queued.
        self._emotion: KeyframeEmotion | None = None

    @property
    def capabilities(self) -> BotCapabilities:
        """The media this transport sends to the user.

        The Keyframe avatar publishes its video to the room itself, so the user
        receives bot video even though this transport sends none.

        Returns:
            The ``audio_out`` capability, with ``video_out`` set.
        """
        return super().capabilities.override(BotCapabilities(video_out=True))

    async def setup(self, setup: FrameProcessorSetup):
        """Setup the output transport.

        Args:
            setup: The frame processor setup configuration.
        """
        await super().setup(setup)
        if self.sample_rate != SAMPLE_RATE or self._params.audio_out_channels != 1:
            raise ValueError(
                f"{self}: the Keyframe avatar expects {SAMPLE_RATE} Hz mono audio; "
                "leave audio_out_sample_rate and audio_out_channels at their defaults"
            )
        if warning := keyframe_policy_warning(setup.expression_policy):
            logger.warning(f"{self}: {warning}")

        await self._client.setup(setup)
        await self._client.join()

    async def cleanup(self):
        """Cleanup output transport resources."""
        await super().cleanup()
        await self._cancel_playback_waits()
        await self._client.cleanup()

    async def start(self, frame: StartFrame):
        """Start the output transport.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        """Stop the output transport.

        Waits for the avatar to play the audio already sent before leaving the
        room.

        Args:
            frame: The end frame signaling transport shutdown.
        """
        await super().stop(frame)
        await self._end_turn()
        waits = [turn.wait_task for turn in self._playing.values() if turn.wait_task]
        if waits:
            await asyncio.gather(*waits, return_exceptions=True)
        await self._client.stop()

    async def cancel(self, frame: CancelFrame):
        """Cancel the output transport.

        Args:
            frame: The cancel frame signaling immediate cancellation.
        """
        await super().cancel(frame)
        await self._cancel_playback_waits()
        await self._client.stop()

    async def send_message(
        self, frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame
    ):
        """Send a message to participants.

        Args:
            frame: The message frame to send.
        """
        logger.trace(f"KeyframeTransport sending message {frame}")
        await self._client.send_message(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, tracking the bot's delivery and handling interruptions.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        if isinstance(frame, InterruptionFrame):
            self._interrupting = True
            try:
                await self._handle_interruptions()
                await super().process_frame(frame, direction)
            finally:
                self._interrupting = False
            return

        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, TTSAudioRawFrame):
                await self._track_context(frame)
            elif isinstance(frame, TTSTextFrame):
                await self._track_delivery(frame)
        await super().process_frame(frame, direction)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame, holding ``BotStoppedSpeakingFrame`` until the avatar finishes playing.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        if isinstance(frame, (_ContextStartFrame, _EmotionCueFrame)):
            return
        if isinstance(frame, BotStoppedSpeakingFrame) and await self._hold_bot_stopped_speaking(
            frame, direction
        ):
            return
        if isinstance(frame, BotStartedSpeakingFrame) and self._drop_held_bot_stopped_speaking(
            direction
        ):
            return
        await super().push_frame(frame, direction)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        """Send an audio frame to the avatar as an app message.

        Args:
            frame: The audio frame to write.

        Returns:
            True if the audio frame was sent, False if it was dropped by an
            interruption.
        """
        if self._interrupting:
            return False
        turn = self._turn
        if turn is None:
            self._turn_count += 1
            turn = self._turn = _Turn(f"t{self._turn_count}")
        records = b"".join(self._cues) + audio_record(frame.audio)
        self._cues.clear()
        await self._client.send_to_avatar(audio_message(turn.name, turn.seq, records))
        turn.seq += 1
        turn.samples += frame.num_frames
        return True

    async def write_transport_frame(self, frame: Frame):
        """Handle a context start or emotion cue once the audio before it was sent.

        Args:
            frame: The frame to handle.
        """
        turn = self._turn
        sent = turn.samples if turn else 0
        if isinstance(frame, _ContextStartFrame):
            self._context_starts[frame.context_id] = sent
        elif isinstance(frame, _EmotionCueFrame):
            at = sent
            start = self._context_starts.get(frame.context_id)
            if frame.offset_secs is not None and start is not None:
                at = start + round(frame.offset_secs * SAMPLE_RATE)
            if self._interrupting:
                # For audio the interruption discards.
                return
            cue = cue_record(frame.emotion, at)
            if turn is None:
                self._cues.append(cue)
                return
            # Audio reaches the avatar ahead of its playback, so a cue sent now
            # can still land on its word; one held for the next audio may not.
            await self._client.send_to_avatar(audio_message(turn.name, turn.seq, cue))
            turn.seq += 1

    async def _track_context(self, frame: TTSAudioRawFrame):
        """Queue a marker where a new TTS context's audio starts."""
        if frame.context_id is None or frame.context_id == self._context_id:
            return
        self._context_id = frame.context_id
        # The baseline the TTS service times the context's words from.
        self._context_start_pts = max(self._last_word_pts, self.get_clock().get_time())
        await self._handle_frame(_ContextStartFrame(context_id=frame.context_id))

    async def _track_delivery(self, frame: TTSTextFrame):
        """Queue an emotion cue when the delivery of the bot's speech changes.

        The cue is queued when the frame arrives rather than at its
        presentation time, because the audio it belongs to is sent ahead of
        playback.
        """
        timed = (
            bool(frame.pts)
            and self._context_id is not None
            and frame.context_id == self._context_id
        )
        if frame.pts:
            self._last_word_pts = frame.pts
            if not frame.interruptible:
                self._last_uninterruptible_word_pts = frame.pts
        emotion = _delivery_emotion(frame.delivery)
        if emotion is None or emotion == self._emotion:
            return
        self._emotion = emotion
        offset = None
        if timed and frame.pts:
            offset = max(0.0, nanoseconds_to_seconds(frame.pts - self._context_start_pts))
        await self._handle_frame(
            _EmotionCueFrame(emotion=emotion.value, context_id=frame.context_id, offset_secs=offset)
        )

    async def _handle_interruptions(self):
        """Handle interruption events by discarding the turns and sending interrupt message."""
        self._turn = None
        self._cues.clear()
        self._context_starts.clear()
        self._context_id = None
        self._emotion = None
        # The TTS service times the words after an interruption from the last
        # uninterruptible word, not from the interrupted ones.
        self._last_word_pts = self._last_uninterruptible_word_pts
        self._plays_until = 0.0
        playing, self._playing = self._playing, {}
        await self._client.send_interrupt_message()
        for turn in playing.values():
            if turn.wait_task:
                task, turn.wait_task = turn.wait_task, None
                await self.cancel_task(task)
            for frame, direction in turn.held:
                await super().push_frame(frame, direction)

    async def _end_turn(self):
        """Mark the turn being sent as done and start waiting for its playback."""
        turn = self._turn
        if turn is None:
            return
        self._turn = None
        self._context_starts.clear()
        records = b"".join(self._cues)
        self._cues.clear()
        await self._client.send_to_avatar(audio_message(turn.name, turn.seq, records, done=True))
        # The avatar plays a turn after the ones sent before it.
        turn.plays_until = max(turn.started_at, self._plays_until) + turn.samples / SAMPLE_RATE
        self._plays_until = turn.plays_until
        self._playing[turn.name] = turn
        turn.wait_task = self.create_task(
            self._wait_for_playback(turn), f"{self}::playback::{turn.name}"
        )

    async def _wait_for_playback(self, turn: _Turn):
        """Push the frames held for a turn if the avatar never reports its playback."""
        deadline = turn.plays_until + PLAYBACK_GRACE_SECS
        while (remaining := deadline - time.monotonic()) > 0:
            await asyncio.sleep(min(BOT_SPEAKING_PERIOD_SECS, remaining))
            if turn.held:
                await self.broadcast_frame(BotSpeakingFrame)
        logger.debug(f"{self} no playback report for turn {turn.name}")
        turn.wait_task = None
        await self._push_held_frames(turn)

    async def _on_playback_finished(self, name: str):
        """Handle the avatar's report that it played a turn to its end."""
        turn = self._playing.get(name)
        if turn is None:
            return
        if turn.wait_task:
            task, turn.wait_task = turn.wait_task, None
            await self.cancel_task(task)
        await self._push_held_frames(turn)

    async def _push_held_frames(self, turn: _Turn):
        """Push the BotStoppedSpeakingFrames held for a turn."""
        self._playing.pop(turn.name, None)
        held, turn.held = turn.held, []
        for frame, direction in held:
            await super().push_frame(frame, direction)

    async def _cancel_playback_waits(self):
        """Stop waiting for playback reports. Idempotent."""
        playing, self._playing = self._playing, {}
        for turn in playing.values():
            if turn.wait_task:
                task, turn.wait_task = turn.wait_task, None
                await self.cancel_task(task)

    async def _hold_bot_stopped_speaking(
        self, frame: BotStoppedSpeakingFrame, direction: FrameDirection
    ) -> bool:
        """Hold a BotStoppedSpeakingFrame until the avatar has played the turn.

        The downstream frame arrives once the turn's audio has been sent, so it
        also marks the turn as done. Its upstream sibling is held with it.

        Returns:
            True if the frame is held, False if it should be pushed now.
        """
        if direction == FrameDirection.DOWNSTREAM and self._turn is not None:
            turn = self._turn
            await self._end_turn()
            turn.held.append((frame, direction))
            return True
        for turn in self._playing.values():
            if any(held.id == frame.broadcast_sibling_id for held, _ in turn.held):
                turn.held.append((frame, direction))
                return True
        return False

    def _drop_held_bot_stopped_speaking(self, direction: FrameDirection) -> bool:
        """Drop a held BotStoppedSpeakingFrame when the bot starts speaking again.

        Returns:
            True if a frame held in ``direction`` was dropped. The
            BotStartedSpeakingFrame is then dropped too, since the stop it
            follows was never pushed.
        """
        for turn in self._playing.values():
            for held in turn.held:
                if held[1] == direction:
                    turn.held.remove(held)
                    return True
        return False


class KeyframeTransport(BaseTransport):
    """Transport implementation to add a Keyframe avatar to Daily calls.

    When used, the Pipecat bot joins the same virtual room as the Keyframe Avatar and the user.
    This is achieved by using `KeyframeTransportClient`, which creates the session via
    `KeyframeApi` and obtains a room URL that all participants connect to.

    Event handlers available:

    - on_session_created(transport, session): Keyframe session created, with the room URL
      and, for a room Keyframe created, the token for the user to join it with
    - on_client_connected(transport, participant): Participant connected to the session
    - on_client_disconnected(transport, participant): Participant disconnected from the session
    - on_avatar_connected(transport, participant): Keyframe avatar connected to the session
    - on_avatar_disconnected(transport, participant, reason): Keyframe avatar disconnected
      from the session

    Example::

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, participant):
            ...
    """

    def __init__(
        self,
        bot_name: str,
        session: aiohttp.ClientSession,
        api_key: str,
        session_request: KeyframeNewSessionRequest,
        params: KeyframeParams | None = None,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        """Initialize the Keyframe transport.

        Args:
            bot_name: The name of the Pipecat bot.
            session: aiohttp session used for async HTTP requests.
            api_key: Keyframe API key for authentication.
            session_request: Session creation parameters. Exactly one of persona_id
                or persona_slug must be provided.
            params: Optional Keyframe-specific configuration parameters.
            input_name: Optional name for the input transport.
            output_name: Optional name for the output transport.
        """
        params = params or KeyframeParams()
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params

        callbacks = KeyframeCallbacks(
            on_session_created=self._on_session_created,
            on_participant_joined=self._on_participant_joined,
            on_participant_left=self._on_participant_left,
        )
        self._client = KeyframeTransportClient(
            bot_name=bot_name,
            callbacks=callbacks,
            api_key=api_key,
            session_request=session_request,
            session=session,
            params=params,
        )
        self._input: KeyframeInputTransport | None = None
        self._output: KeyframeOutputTransport | None = None

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_session_created")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_avatar_connected")
        self._register_event_handler("on_avatar_disconnected")

    @property
    def session(self) -> KeyframeSession | None:
        """Get the Keyframe session.

        Returns:
            The session, or None before the transport is set up.
        """
        return self._client.session

    async def set_emotion(self, emotion: KeyframeEmotion | str):
        """Set the avatar's emotion.

        The avatar follows the delivery of the bot's speech on its own. Use this
        to set an emotion while the bot is not speaking.

        Args:
            emotion: The emotion to change to.
        """
        await self._client.send_to_avatar(emotion_now_message(KeyframeEmotion(emotion).value))

    async def _on_session_created(self, session: KeyframeSession):
        """Handle session created events."""
        await self._call_event_handler("on_session_created", session)

    async def _on_participant_left(self, participant, reason):
        """Handle participant left events."""
        if self._client.is_avatar(participant):
            self._client.avatar_left()
            await self._on_avatar_disconnected(participant, reason)
        else:
            await self._on_client_disconnected(participant)

    async def _on_participant_joined(self, participant):
        """Handle participant joined events."""
        if self._client.is_avatar(participant):
            await self._client.avatar_joined(participant["id"])
            # Ignore the Keyframe avatar's microphone
            logger.debug(f"Ignoring {participant['id']}'s microphone")
            await self.update_subscriptions(
                participant_settings={participant["id"]: {"media": {"microphone": "unsubscribed"}}}
            )
            await self._on_avatar_connected(participant)
        else:
            await self._on_client_connected(participant)
            if self._input:
                await self._input.start_capturing_audio(participant)

    async def update_subscriptions(self, participant_settings=None, profile_settings=None):
        """Update subscription settings for participants.

        Args:
            participant_settings: Per-participant subscription settings.
            profile_settings: Global subscription profile settings.
        """
        await self._client.update_subscriptions(
            participant_settings=participant_settings,
            profile_settings=profile_settings,
        )

    def input(self) -> FrameProcessor:
        """Get the input transport for receiving media and events.

        Returns:
            The Keyframe input transport instance.
        """
        if not self._input:
            self._input = KeyframeInputTransport(client=self._client, params=self._params)
        return self._input

    def output(self) -> FrameProcessor:
        """Get the output transport for sending media and events.

        Returns:
            The Keyframe output transport instance.
        """
        if not self._output:
            self._output = KeyframeOutputTransport(client=self._client, params=self._params)
        return self._output

    async def _on_avatar_connected(self, participant: Any):
        """Handle avatar connected events."""
        await self._call_event_handler("on_avatar_connected", participant)

    async def _on_avatar_disconnected(self, participant: Any, reason: str):
        """Handle avatar disconnected events."""
        await self._call_event_handler("on_avatar_disconnected", participant, reason)

    async def _on_client_connected(self, participant: Any):
        """Handle client connected events."""
        await self._call_event_handler("on_client_connected", participant)

    async def _on_client_disconnected(self, participant: Any):
        """Handle client disconnected events."""
        await self._call_event_handler("on_client_disconnected", participant)

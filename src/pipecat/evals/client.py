#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The harness's connection to the bot under test.

:class:`EvalClient` runs the eval pipeline, a Pipecat pipeline acting as an
RTVI client: it waits for the bot to listen, connects, completes the
handshake, sends the user's turns (text, synthesized speech, a recording,
DTMF keys, an image), records the conversation, and tears down. The bot's
output reaches the rest of the harness as events on the stream the client
is given. In a simulation the pipeline also carries the persona LLM, which
answers the bot on its own.
"""

import asyncio
import base64
import mimetypes
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.evals.audio import load_user_audio
from pipecat.evals.client_transport import EvalClientRecorder, EvalClientTransport
from pipecat.evals.events import EvalEventStream
from pipecat.evals.persona import EvalPersona
from pipecat.evals.results import EvalTrace
from pipecat.evals.serializer import (
    EVAL_CANCEL_MESSAGE_TYPE,
    EVAL_CONFIGURE_MESSAGE_TYPE,
    EVAL_CONTEXT_MESSAGE_TYPE,
    EVAL_IMAGE_MESSAGE_TYPE,
    EVAL_STT_SAMPLE_RATE,
    EvalClientSerializer,
)
from pipecat.evals.session import EvalSessionParams
from pipecat.evals.tts import CachingTTSService, tts_sample_rate
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    LLMConfigureOutputFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import LLMSettings
from pipecat.services.stt_service import STTService
from pipecat.transports.websocket.client import WebsocketClientParams
from pipecat.workers.runner import WorkerRunner

BOT_READY_TIMEOUT_S = 10.0


# Frames the bot produced: the sink turns them into events and stops them here.
# Everything downstream of the sink speaks for the user (a persona LLM, the user
# TTS, the output) and must never see the bot's text or reports as its own
# input. The aggregator's context frame and lifecycle frames pass.
_BOT_FRAMES = (
    LLMFullResponseStartFrame,
    LLMTextFrame,
    LLMFullResponseEndFrame,
    TTSTextFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallCancelFrame,
    InputTransportMessageFrame,
)


class _BotFrameSink(FrameProcessor):
    """Where the bot's output stops and the user's turns start.

    The bot's frames become events on the stream and go no further. The
    user's turns are injected here, past the bot's side of the pipeline, so
    an interruption there cannot drop them.

    Two rules apply. The aggregator's interruption, raised when the bot
    speaks, stops here for a scripted turn (it would flush the user audio
    being sent) and passes for a persona (a caller who hears the bot keep
    talking stops). And the persona answers from here: in text mode it hears
    each finished bot response through the sink, in audio mode the aggregator
    hands it the bot's turn; either way nothing reaches it once it hung up.
    """

    def __init__(
        self,
        stream: EvalEventStream,
        *,
        persona: EvalPersona | None = None,
        persona_hears: bool = False,
    ):
        """Initialize the sink.

        Args:
            stream: Where the bot's frames go as events.
            persona: The simulated caller, else ``None``.
            persona_hears: Whether the persona hears the bot's responses through
                the sink (text mode, with no aggregator to hand it the turn).
        """
        super().__init__()
        self._stream = stream
        self._persona = persona
        self._persona_hears = persona_hears and persona is not None

    async def inject(self, frame: Frame) -> None:
        """Push a user turn into the pipeline, past the bot's side.

        Args:
            frame: The frame to push (a ``TTSSpeakFrame``, or a spoken recording's
                TTS-bracketed frames).
        """
        await self.push_frame(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # The bot's frames arrive downstream, from the input transport. What
        # travels upstream is the user side's own (a persona's function call, a
        # TTS's lifecycle), not bot output, and passes through as is.
        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return
        event = self._stream.frame_to_event(frame)
        if event is not None:
            await self._stream.append(event)
            if self._persona_hears and self._persona is not None:
                await self._answer(self._persona.hear(event))
        if isinstance(frame, _BOT_FRAMES):
            return
        elif isinstance(frame, InterruptionFrame) and self._persona is None:
            return
        elif isinstance(frame, LLMContextFrame):
            # The aggregator handing the persona the bot's turn (audio mode).
            await self._answer(frame)
            return
        await self.push_frame(frame, direction)

    async def _answer(self, frame: LLMContextFrame | None) -> None:
        """Push the frame that has the persona answer, unless there is none or it hung up."""
        if frame is None or self._persona is None or self._persona.hung_up:
            return
        await self.push_frame(frame)


# The event the relay appends for each response the persona completes, and the
# one the client appends when the bot ends the call by closing the connection.
PERSONA_TURN_EVENT = "persona_turn"
BOT_ENDED_EVENT = "bot_ended"
# The harness's own pipeline reported an error: a service in it (the persona
# LLM, the user TTS, the bot STT) failed, not the bot.
HARNESS_ERROR_EVENT = "harness_error"


class _PersonaTurnRelay(FrameProcessor):
    """Sends the persona's replies to the bot and reports each one as a turn.

    In text mode a whole reply goes to the bot as one ``send-text``; in audio
    mode the user TTS has already spoken it and this only logs the text. A
    reply with words in it is one ``persona_turn`` event; a reply that only
    calls ``end_call``, or that the bot's speech cut off before a word went
    out, is not a turn.
    """

    def __init__(
        self, text_turn_message: Callable[[str], dict], trace: EvalTrace, stream: EvalEventStream
    ):
        """Initialize the relay.

        Args:
            text_turn_message: Builds the RTVI ``send-text`` message for a turn.
            trace: The run's trace.
            stream: Where the persona's turns are counted.
        """
        super().__init__()
        self._text_turn_message = text_turn_message
        self._trace = trace
        self._stream = stream
        self._text: list[str] = []
        # What the persona said in the current response: the one text turn, or
        # the sentences the user TTS spoke.
        self._spoken: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, (LLMFullResponseStartFrame, InterruptionFrame)):
            self._text = []
            self._spoken = []
        elif isinstance(frame, LLMTextFrame) and frame.skip_tts:
            self._text.append(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            text = "".join(self._text).strip()
            self._text = []
            if frame.skip_tts and text:
                self._trace.log(f"send: {text!r} (persona, text)")
                await self.push_frame(
                    OutputTransportMessageUrgentFrame(message=self._text_turn_message(text))
                )
                self._spoken.append(text)
            if self._spoken:
                await self._stream.append(
                    {"type": PERSONA_TURN_EVENT, "text": " ".join(self._spoken)}
                )
            self._spoken = []
        elif isinstance(frame, TTSTextFrame):
            self._trace.log(f"send: {frame.text!r} (persona, audio)")
            self._spoken.append(frame.text)
        await self.push_frame(frame, direction)


class EvalClientParams(BaseModel):
    """What the scenario asks of the bot; the session derives it from its scenario.

    Parameters:
        bot_audio: Whether the bot speaks; in text mode it is asked to skip TTS
            at connect.
        user_audio: Whether the user's turns reach the bot as audio.
        user_speech: TTS config the user's turns are synthesized with, or
            ``None``; sets the user audio rate.
        capture_bot_audio: Whether the bot forwards its synthesized audio, for
            the ``response`` transcription or a persona that listens.
        report_level: Function-call report level to ask of the bot, or ``None``
            for its default.
        vad_events: Whether to ask the bot for its raw VAD events.
        context: Messages the bot's context starts from, sent right after the
            handshake; empty sends nothing.
        trigger_disconnect: Whether the scenario itself asks for the bot's
            ``on_client_disconnected`` handler to fire when this connection
            ends; the run's :class:`~pipecat.evals.session.EvalSessionParams`
            can ask too.
    """

    bot_audio: bool = False
    user_audio: bool = False
    user_speech: dict | None = None
    capture_bot_audio: bool = False
    report_level: str | None = None
    vad_events: bool = False
    context: list[dict] = Field(default_factory=list)
    trigger_disconnect: bool = False


class EvalClient:
    """The eval pipeline that talks to the bot, and the user's sends into it.

    Built once per run: :meth:`wait_for_bot`, :meth:`start`, :meth:`handshake`,
    the turns (:meth:`send_text`, :meth:`say`, :meth:`play`, :meth:`send_dtmf`,
    :meth:`send_image`), then :meth:`stop`. The pipeline is the same for both
    modalities and both kinds of eval::

        input -> [STT -> user aggregator] -> sink -> [persona LLM] -> [user TTS]
              -> [persona relay] -> output -> [assistant aggregator]

    The bracketed stages exist only in audio mode (the STT, the aggregator,
    the user TTS) or in a simulation (the persona LLM, its relay, the
    aggregator that records its replies). The bot's output comes in through
    the input and stops at the sink, as events; the user's turns start at the
    sink and leave through the output. A session builds it, with the params
    its scenario asks for.
    """

    def __init__(
        self,
        bot_url: str,
        *,
        params: EvalClientParams | None = None,
        session_params: EvalSessionParams | None = None,
        stream: EvalEventStream,
        trace: EvalTrace,
        user_tts: CachingTTSService | None = None,
        bot_stt: STTService | None = None,
        persona: EvalPersona | None = None,
    ):
        """Initialize the client.

        Args:
            bot_url: WebSocket URL of the bot's eval transport.
            params: What the scenario asks of the bot; ``None`` asks nothing
                beyond a text-mode conversation.
            session_params: How the run behaves: the connect timeout, the
                recording, and the teardown are the client's part of it.
                ``None`` for the defaults.
            stream: Where the bot's output is appended as events.
            trace: The run's trace.
            user_tts: The :class:`~pipecat.evals.tts.CachingTTSService` that
                synthesizes spoken user turns, or ``None`` for text mode.
            bot_stt: The ``STTService`` that transcribes the bot's audio into
                the ``response`` event, or ``None`` when unused.
            persona: A simulation's :class:`~pipecat.evals.persona.EvalPersona`,
                whose LLM rides in the pipeline and whose context the
                aggregators keep up to date with both sides of the conversation;
                ``None`` for a scripted scenario.
        """
        self._bot_url = bot_url
        self._stream = stream
        self._trace = trace
        self._params = params or EvalClientParams()
        self._session_params = session_params or EvalSessionParams()
        self._user_tts = user_tts
        self._bot_stt = bot_stt
        self._persona = persona

        # The eval pipeline's worker (built by start()) and the runner task driving it.
        self._worker: PipelineWorker | None = None
        # Set once stop() begins: a disconnect from then on is our own teardown.
        self._stopping = False
        # Where the user's turns enter the pipeline (built with the processors).
        self._sink: _BotFrameSink | None = None
        self._run_task: asyncio.Task | None = None
        # Records the conversation audio (bot + user) when record_path is set and
        # the scenario is audio mode; fed raw audio by the transport, written on stop().
        self._recorder: EvalClientRecorder | None = None
        # Set by the transport's on_bot_ready handler once the bot completes the
        # RTVI handshake; handshake() waits on it.
        self._bot_ready_event = asyncio.Event()
        self._next_id = 0

    @property
    def has_user_tts(self) -> bool:
        """Whether text user turns are spoken to the bot rather than sent as text."""
        return self._user_tts is not None

    @property
    def sends_user_audio(self) -> bool:
        """Whether the user's side streams audio to the bot (the output is enabled)."""
        return self._user_tts is not None or self._params.user_audio

    async def wait_for_bot(self) -> None:
        """Wait until the bot accepts connections.

        Retries a plain TCP connect until the bot is listening, and stops short of
        the WebSocket handshake on purpose: a full connection would make the bot
        greet, and that connection would then be thrown away.

        Raises:
            OSError: The last connect error, when the bot never accepted within
                the connect timeout.
            TimeoutError: When the timeout passed without a single attempt failing.
        """
        u = urlsplit(self._bot_url)
        host, port = u.hostname or "localhost", u.port or (443 if u.scheme == "wss" else 80)
        deadline = time.monotonic() + self._session_params.connect_timeout_s
        connect_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                _reader, writer = await asyncio.open_connection(host, port)
            except OSError as e:  # not accepting connections yet
                connect_error = e
                await asyncio.sleep(0.25)
                continue
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
            return
        raise connect_error or TimeoutError("timed out")

    async def start(self) -> None:
        """Build the eval pipeline and start running it, which connects to the bot."""
        transport = self._transport()
        pipeline = Pipeline(self._processors(transport))
        self._worker = PipelineWorker(
            pipeline,
            params=self._pipeline_params(),
            enable_rtvi=False,
            cancel_on_idle_timeout=False,
        )

        @self._worker.event_handler("on_pipeline_error")
        async def _on_pipeline_error(_worker, frame):
            await self._stream.append({"type": HARNESS_ERROR_EVENT, "text": str(frame.error)})

        runner = WorkerRunner()
        await runner.add_workers(self._worker)
        self._run_task = asyncio.create_task(runner.run())

    async def handshake(self) -> None:
        """Wait for ``bot-ready``, then send what the scenario asks of the bot and its context.

        A bot that never says ready is not a usable eval target, so this raises
        instead of sending turns to a half-started bot.

        Raises:
            TimeoutError: If the bot never sends ``bot-ready``.
        """
        # The transport sends client-ready on connect and fires on_bot_ready when
        # the bot answers; our handler sets _bot_ready_event.
        try:
            await asyncio.wait_for(self._bot_ready_event.wait(), timeout=BOT_READY_TIMEOUT_S)
        except TimeoutError:
            raise TimeoutError(
                f"bot-ready not received within {int(BOT_READY_TIMEOUT_S * 1000)}ms"
            ) from None

        # Ask the bot's RTVIObserver to expose what this scenario needs, for the
        # duration of this eval only (bots keep their defaults; only the eval
        # transport understands this): raise the function-call report level if it
        # asserts on call name/args, and enable raw VAD speaking events if it uses
        # them.
        config: dict = {}
        if self._params.report_level is not None:
            config["function_call_report_level"] = {"*": self._params.report_level}
        if self._params.vad_events:
            config["vad_user_speaking"] = True
        if config:
            await self._send_eval(EVAL_CONFIGURE_MESSAGE_TYPE, config)

        # Only send the eval-context when the scenario provides starting context.
        # An implicit empty one would race with bot startup flows (e.g. a greeting
        # added in on_client_connected), wiping the bot's context right after it
        # set it up.
        if self._params.context:
            await self._send_eval(EVAL_CONTEXT_MESSAGE_TYPE, {"messages": self._params.context})

    async def stop(self) -> None:
        """Save the recording, optionally cancel the bot, and end the pipeline."""
        self._stopping = True
        # Write the recording first: the recorder is harness-owned and fed raw
        # audio by the transport, so nothing below clears it, but writing here
        # lands it even if the teardown raises.
        await self._write_recording()
        # Optionally ask the bot to tear its pipeline down gracefully so it exits
        # on its own (best-effort; skipped by default so it stays up for more
        # scenarios).
        if self._session_params.stop_bot:
            await self._send_cancel()
        if self._worker is None or self._run_task is None:
            return
        # End the worker (which disconnects the transport), falling back to
        # cancel if it doesn't wind down cleanly.
        try:
            await self._worker.queue_frame(EndFrame())
            await asyncio.wait_for(self._run_task, timeout=5.0)
        except (TimeoutError, asyncio.CancelledError, Exception):
            self._run_task.cancel()
            try:
                await self._run_task
            except (asyncio.CancelledError, Exception):
                pass

    async def send(self, message: RTVI.Message) -> None:
        """Send an RTVI client message to the bot through the transport pipeline.

        Args:
            message: The message to send.
        """
        assert self._worker is not None  # pipeline built before any send
        await self._worker.queue_frame(
            OutputTransportMessageUrgentFrame(message=message.model_dump())
        )

    async def send_text(self, text: str) -> None:
        """Send a text user turn via the RTVI ``send-text`` message.

        Args:
            text: The user's turn.
        """
        await self.send(RTVI.Message(**self._text_turn_message(text)))

    async def send_dtmf(self, keys: str) -> None:
        """Send a DTMF keypress turn as one RTVI ``dtmf`` message.

        The bot handles the keys the way it handles a phone keypad.

        Args:
            keys: The keys to press, in order.
        """
        await self.send(self._message("dtmf", {"buttons": list(keys)}))

    async def send_image(self, image_path: str) -> None:
        """Register an image for the current turn.

        The bot's eval transport serves it back when the bot asks for a user
        image. The file is sent as is.

        Args:
            image_path: Path to the image file.
        """
        path = Path(image_path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        self._trace.log(f"send: image {path.name} ({mime})")
        await self._send_eval(EVAL_IMAGE_MESSAGE_TYPE, {"image": encoded, "format": mime})

    async def say(self, text: str) -> None:
        """Speak ``text`` as the user.

        The user TTS synthesizes it (cached) and the output paces it to the bot
        as live audio.

        Args:
            text: What the user says.
        """
        assert self._sink is not None  # pipeline built before any send
        await self._sink.inject(TTSSpeakFrame(text))

    async def play(self, path: str) -> None:
        """Play a recording to the bot as the user's turn, instead of synthesizing one.

        It goes out like a spoken turn: resampled to the user audio rate, paced
        to the bot, and recorded.

        Args:
            path: Path to the audio file.
        """
        assert self._sink is not None  # pipeline built before any send
        pcm, sample_rate = await load_user_audio(path)
        for frame in (
            TTSStartedFrame(),
            TTSAudioRawFrame(audio=pcm, sample_rate=sample_rate, num_channels=1),
            TTSStoppedFrame(),
        ):
            await self._sink.inject(frame)

    async def configure_persona(self, instruction: str) -> None:
        """Give the persona LLM its instruction, and say whether its replies are spoken.

        In text mode a reply goes to the bot as text; in audio mode the user TTS
        speaks it.

        Args:
            instruction: The persona's system instruction.
        """
        assert self._sink is not None  # pipeline built before any send
        await self._sink.inject(
            LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=instruction))
        )
        await self._sink.inject(LLMConfigureOutputFrame(skip_tts=not self._params.user_audio))

    async def hang_up(self) -> None:
        """End the persona's part of the conversation: it answers nothing more."""
        assert self._persona is not None
        self._persona.hang_up()

    def _connect_url(self) -> str:
        """The bot's URL with this connection's eval flags.

        ``skip_tts`` in text mode, ``capture_bot_audio`` when the harness needs
        the bot's audio, ``trigger_disconnect`` when the run asks for it.
        """
        flags = []
        if not self._params.bot_audio:
            flags.append("skip_tts=true")
        # Forward the bot's audio when something listens to it (the response
        # transcription, a persona) or when recording an audio run.
        if self._params.capture_bot_audio or (self._record_path and self._params.bot_audio):
            flags.append("capture_bot_audio=true")
        if self._session_params.trigger_disconnect or self._params.trigger_disconnect:
            flags.append("trigger_disconnect=true")
        if not flags:
            return self._bot_url
        sep = "&" if "?" in self._bot_url else "?"
        return f"{self._bot_url}{sep}{'&'.join(flags)}"

    @property
    def _record_path(self) -> str | None:
        """Where the run records the conversation, or ``None``."""
        return self._session_params.record_path

    @property
    def _user_audio_rate(self) -> int:
        """The rate the user's audio is synthesized at; 0 in text mode."""
        return tts_sample_rate(self._params.user_speech) if self._params.user_speech else 0

    def _transport(self) -> EvalClientTransport:
        """The transport to the bot, with the recorder when the run records, and its two handlers."""
        params = WebsocketClientParams(
            audio_in_enabled=self._params.bot_audio,
            audio_out_enabled=self.sends_user_audio,
            audio_in_sample_rate=EVAL_STT_SAMPLE_RATE if self._params.bot_audio else 0,
            audio_out_sample_rate=self._user_audio_rate,
            serializer=EvalClientSerializer(),
        )
        # The recorder is fed the raw audio by both edges; the paced streams
        # would make the recording stutter.
        if self._record_path and self._params.bot_audio:
            self._recorder = EvalClientRecorder(self._user_audio_rate or EVAL_STT_SAMPLE_RATE)
        transport = EvalClientTransport(self._connect_url(), params, recorder=self._recorder)

        @transport.event_handler("on_bot_ready")
        async def _on_bot_ready(_transport):
            self._bot_ready_event.set()

        @transport.event_handler("on_disconnected")
        async def _on_disconnected(_transport, _websocket):
            # The bot ended the call by closing the connection; our own teardown
            # closes it too, and that is not the bot's doing.
            if not self._stopping:
                await self._stream.append({"type": BOT_ENDED_EVENT})

        return transport

    def _processors(self, transport: EvalClientTransport) -> list:
        """The pipeline's processors: inbound from the bot, the sink, outbound to the bot.

        The STT and the turn aggregator exist in audio mode; the persona LLM, its
        relay, and the assistant aggregator in a simulation.
        """
        # The context the aggregators keep: the persona's in a simulation, else a
        # throwaway that only gives the user aggregator somewhere to put turns.
        persona, user_tts, bot_stt = self._persona, self._user_tts, self._bot_stt
        context = persona.context if persona is not None else LLMContext()
        # With no STT there is no aggregator to hand the persona the bot's turns;
        # it hears them through the sink instead.
        self._sink = _BotFrameSink(self._stream, persona=persona, persona_hears=bot_stt is None)

        inbound: list = [transport.input()]
        if bot_stt is not None:
            inbound += [bot_stt, self._bot_turn_aggregator(context)]

        speech = [user_tts] if user_tts is not None else []
        if persona is None:
            outbound: list = [*speech, transport.output()]
        else:
            relay = _PersonaTurnRelay(self._text_turn_message, self._trace, self._stream)
            outbound = [
                persona.llm,
                *speech,
                relay,
                transport.output(),
                LLMAssistantAggregator(context),
            ]
        return inbound + [self._sink] + outbound

    def _bot_turn_aggregator(self, context: LLMContext) -> LLMUserAggregator:
        """The aggregator that cuts the bot's transcribed speech into turns (audio mode).

        Its turn hooks give the stream the timing of the bot's replies, and its
        finished turn is the persona's next user message.
        """
        aggregator = LLMUserAggregator(
            context,
            params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.4))
            ),
        )

        # The aggregator consumes the STT's TranscriptionFrames to build the
        # bot's turn, so the response comes from the aggregated turn text here
        # (not from a frame at the sink). The stream decides whether the
        # finished turn is the bot's reply or an earlier turn finalized late
        # (see EvalEventStream.bot_turn_stopped).
        @aggregator.event_handler("on_user_turn_started")
        async def _on_user_turn_started(_aggregator, _strategy):
            self._stream.bot_turn_started()

        @aggregator.event_handler("on_user_turn_stopped")
        async def _on_user_turn_stopped(_aggregator, _strategy, message):
            await self._stream.bot_turn_stopped(message.content or "")

        return aggregator

    def _pipeline_params(self) -> PipelineParams:
        """The StartFrame's rates: the bot STT reads the input rate, the user TTS produces the output rate.

        They come from the scenario so the user TTS synthesizes at the
        configured rate; in text mode the STT rate is a placeholder.
        """
        return PipelineParams(
            audio_in_sample_rate=EVAL_STT_SAMPLE_RATE,
            audio_out_sample_rate=self._user_audio_rate or EVAL_STT_SAMPLE_RATE,
        )

    def _message(self, message_type: str, data: dict) -> RTVI.Message:
        """One RTVI client message, numbered in order of creation."""
        self._next_id += 1
        return RTVI.Message(type=message_type, id=str(self._next_id), data=data)

    async def _send_eval(self, eval_type: str, data: dict) -> None:
        """Send one of the eval transport's own messages, in the ``client-message`` envelope."""
        await self.send(self._message("client-message", {"t": eval_type, "d": data}))

    def _text_turn_message(self, text: str) -> dict:
        """The RTVI ``send-text`` message for a user turn, as a dict.

        The bot answers right away, and speaks the answer only in audio mode.
        """
        data = RTVI.SendTextData(
            content=text,
            options=RTVI.SendTextOptions(
                run_immediately=True, audio_response=self._params.bot_audio
            ),
        )
        return self._message("send-text", data.model_dump()).model_dump()

    async def _send_cancel(self) -> None:
        """Ask the bot to cancel its pipeline and exit.

        Best-effort: the connection may already be gone.
        """
        try:
            await self._send_eval(EVAL_CANCEL_MESSAGE_TYPE, {})
        except Exception:
            pass

    async def _write_recording(self) -> None:
        """Write the recorded conversation audio (bot + user) to ``record_path``."""
        if self._recorder is None or not self._record_path or not self._recorder.has_audio():
            return
        if await self._recorder.write(self._record_path):
            self._trace.log(f"recording saved: {self._record_path}")

#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The harness's connection to the bot under test.

:class:`EvalClient` is the runtime every driver shares. It waits for the bot to
listen, runs the eval pipeline whose bot-facing edge is
:class:`~pipecat.evals.client_transport.EvalHarnessTransport`, completes the RTVI
handshake, sends the user's turns (text, synthesized speech, a recording, DTMF
keys, an image), records the conversation, and tears everything down. The bot's
output reaches the rest of the harness through the
:class:`~pipecat.evals.events.EvalEventStream` the client is given. For a
simulation the pipeline also carries the persona LLM, which answers the bot on
its own; the client then relays its turns instead of being told what to send.
"""

import asyncio
import base64
import mimetypes
import time
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.evals.audio import load_user_audio
from pipecat.evals.client_transport import EvalHarnessTransport, HarnessRecorder
from pipecat.evals.events import EvalEventStream
from pipecat.evals.results import EvalTrace
from pipecat.evals.scenario import EvalScriptScenario
from pipecat.evals.serializer import (
    EVAL_CANCEL_MESSAGE_TYPE,
    EVAL_CONFIGURE_MESSAGE_TYPE,
    EVAL_CONTEXT_MESSAGE_TYPE,
    EVAL_IMAGE_MESSAGE_TYPE,
    HARNESS_STT_SAMPLE_RATE,
    RTVIHarnessSerializer,
)
from pipecat.evals.simulation import EvalSimulationScenario
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
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.services.stt_service import STTService
from pipecat.transports.websocket.client import WebsocketClientParams
from pipecat.workers.runner import WorkerRunner

BOT_READY_TIMEOUT_S = 10.0


# Frames the bot produced: the sink turns them into events and stops them here.
# Everything downstream of the sink is the user's side (a persona LLM, the user
# TTS, the output), which must never see the bot's text or reports as its own
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
    """The boundary between the bot's side of the eval pipeline and the user's.

    For every frame it calls the stream's
    :meth:`~pipecat.evals.events.EvalEventStream.frames_to_events` and appends
    the results with :meth:`~pipecat.evals.events.EvalEventStream.append`. The
    bot's own frames (its text, spoken text, speaking reports, function calls,
    and the messages it reports about the harness) stop here; what passes is
    the pipeline's lifecycle and the aggregator's context frame.

    The harness's *computed* interruptions are decided here: the user aggregator
    runs a VAD on the bot's incoming audio, so when the bot speaks it broadcasts
    an ``InterruptionFrame`` downstream. For a scripted turn the sink swallows
    it, since letting it reach the user TTS / output would flush the user audio
    being paced out (dropping a barge-in turn, and the user's side of the
    recording). For a persona it passes: a caller who hears the bot keep
    talking stops, and the aggregator has the persona answer again once the bot
    is done, with everything it said.

    The user's turns enter the pipeline here as well (:meth:`inject`), straight
    into the user-audio side. A turn queued at the pipeline head would first
    cross the bot-audio side, and every processor there (this sink included)
    resets its queue when the aggregator's interruption reaches it, dropping a
    turn still queued behind a slow transcription at that moment.

    In a text-mode simulation the sink is
    also what hands the bot's turns to the persona: with no audio there is no
    STT or aggregator to do it, so each finished bot response is appended to
    the persona's context as a user message and the context is pushed on for
    the persona LLM to answer. A response the bot gives while one of its
    function calls is still running is held and joined with the response that
    follows the call, so the persona answers the bot's complete turn rather
    than its "let me check". Once the persona has hung up (:meth:`hang_up`),
    nothing more reaches it.
    """

    def __init__(
        self, stream: EvalEventStream, *, persona: LLMContext | None = None, feed: bool = False
    ):
        """Initialize the sink.

        Args:
            stream: Where the bot's frames go as events.
            persona: The persona's context in a simulation, else ``None``.
            feed: Whether the sink hands the bot's finished responses to the
                persona itself (a text-mode simulation, with no aggregator to).
        """
        super().__init__()
        self._stream = stream
        self._persona = persona
        self._feed = feed and persona is not None
        self._hung_up = False
        self._held_response: list[str] = []
        self._calls_in_progress = 0

    def hang_up(self) -> None:
        """End the persona's part: no further bot turn reaches the persona LLM."""
        self._hung_up = True

    async def inject(self, frame: Frame) -> None:
        """Push a user-turn frame downstream, into the user TTS and the output.

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
        events = self._stream.frames_to_events(frame)
        for event in events:
            await self._stream.append(event)
        if self._feed and not self._hung_up:
            await self._feed_persona(events)
        if isinstance(frame, _BOT_FRAMES):
            return
        # The computed interruption stops here for a scripted turn (see the
        # class docstring); a persona reacts to it like a caller.
        if isinstance(frame, InterruptionFrame) and self._persona is None:
            return
        if isinstance(frame, LLMContextFrame):
            # The aggregator asking the persona to answer (audio mode).
            await self._run_persona(frame)
            return
        await self.push_frame(frame, direction)

    async def _run_persona(self, frame: LLMContextFrame) -> None:
        """Hand a context frame to the persona LLM, unless the persona hung up."""
        if self._persona is None or self._hung_up:
            return
        await self.push_frame(frame)

    async def _feed_persona(self, events: list[dict]) -> None:
        """Hand a finished bot response to the persona, once its function calls are done."""
        assert self._persona is not None
        for event in events:
            match event["type"]:
                case "function_call":
                    self._calls_in_progress += 1
                case "function_call_stopped":
                    self._calls_in_progress = max(0, self._calls_in_progress - 1)
                case "llm_response":
                    if event["text"]:
                        self._held_response.append(event["text"])
                    if self._calls_in_progress or not self._held_response:
                        continue
                    text = " ".join(self._held_response)
                    self._held_response = []
                    self._persona.add_message({"role": "user", "content": text})
                    await self._run_persona(LLMContextFrame(self._persona))


# The event the relay appends for each response the persona completes, and the
# one the client appends when the bot ends the call by closing the connection.
PERSONA_TURN_EVENT = "persona_turn"
BOT_ENDED_EVENT = "bot_ended"


class _PersonaTurnRelay(FrameProcessor):
    """Relays the persona's turns to the bot, counts them, and traces them.

    In text mode the persona LLM's output carries ``skip_tts`` and reaches this
    processor as text: one whole response is sent to the bot as a single RTVI
    ``send-text``. In audio mode the user TTS has already turned the response
    into audio and this only traces the spoken text as it passes. Either way
    the frames go on to the assistant aggregator that records the persona's
    side of the conversation, and each response in which the persona said
    something is one persona turn, reported as a ``persona_turn`` event. A
    response that only calls ``end_call``, or one the bot's speech cut off
    before a word went out, is not a turn.
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
        self._spoke = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, (LLMFullResponseStartFrame, InterruptionFrame)):
            self._text = []
            self._spoke = False
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
                self._spoke = True
            if self._spoke:
                await self._stream.append({"type": PERSONA_TURN_EVENT})
            self._spoke = False
        elif isinstance(frame, TTSTextFrame):
            self._trace.log(f"send: {frame.text!r} (persona, audio)")
            self._spoke = True
        await self.push_frame(frame, direction)


class EvalClient:
    """The eval pipeline that talks to the bot, and the user's sends into it.

    Built once per run: :meth:`wait_for_bot`, :meth:`start`, :meth:`handshake`,
    the turns (:meth:`send_text`, :meth:`say`, :meth:`play`, :meth:`send_dtmf`,
    :meth:`send_image`), then :meth:`stop`. The pipeline is the same for both
    modalities and both kinds of eval::

        input -> [STT -> user aggregator] -> sink -> [persona LLM] -> [user TTS]
              -> [persona relay] -> output -> [assistant aggregator]

    The STT + aggregator (with the aggregator's own VAD) transcribe the bot's
    audio into the ``response``; the user TTS turns spoken turns into the audio
    sent to the bot. Each bracketed stage is present only when its service was
    built (audio scenarios); in text mode they're simply absent and no audio
    flows. The sink turns the bot's frames into events either way. A simulation
    adds the persona LLM, which answers the bot's turns on its own: the
    aggregator's context is the persona's, the relay turns text-mode replies
    into ``send-text``, and the assistant aggregator records what the persona
    said. Build with :meth:`for_scenario` or :meth:`for_simulation`.
    """

    def __init__(
        self,
        *,
        bot_url: str,
        stream: EvalEventStream,
        trace: EvalTrace,
        bot_audio: bool,
        user_audio: bool,
        user_speech: dict | None = None,
        capture_bot_audio: bool = False,
        report_level: str | None = None,
        vad_events: bool = False,
        context: list[dict] | None = None,
        trigger_disconnect: bool = False,
        connect_timeout_s: float = 5.0,
        record_path: str | None = None,
        stop_bot: bool = False,
        user_tts: CachingTTSService | None = None,
        bot_stt: STTService | None = None,
        persona_llm: LLMService | None = None,
        persona_context: LLMContext | None = None,
    ):
        """Initialize the client.

        Args:
            bot_url: WebSocket URL of the bot's eval transport.
            stream: Where the bot's output is appended as events.
            trace: The run's trace.
            bot_audio: Whether the bot speaks; in text mode it is asked to skip
                TTS at connect.
            user_audio: Whether the user's turns reach the bot as audio.
            user_speech: TTS config the user's turns are synthesized with, or
                ``None``; sets the user audio rate.
            capture_bot_audio: Whether the bot forwards its synthesized audio,
                for the ``response`` transcription or a persona that listens.
            report_level: Function-call report level to ask of the bot, or
                ``None`` for its default.
            vad_events: Whether to ask the bot for its raw VAD events.
            context: Messages the bot's context starts from, sent right after
                the handshake; ``None`` or empty sends nothing.
            trigger_disconnect: Whether to ask the eval transport to fire the
                bot's ``on_client_disconnected`` handler when this connection
                ends. Bots often cancel their pipeline there, so it is off by
                default to avoid that between scenarios.
            connect_timeout_s: How long :meth:`wait_for_bot` waits for the bot
                to accept connections.
            record_path: When set (and ``bot_audio``), the conversation audio is
                recorded to this path on :meth:`stop`.
            stop_bot: When True, ask the bot to cancel its pipeline (and exit) on
                :meth:`stop` via ``eval-cancel``. The suite enables it to clean
                up each spawned bot.
            user_tts: The :class:`~pipecat.evals.tts.CachingTTSService` that
                synthesizes spoken user turns, or ``None`` for text mode.
            bot_stt: The ``STTService`` that transcribes the bot's audio into the
                ``response`` event, or ``None`` when unused.
            persona_llm: A simulation's persona LLM service, or ``None``.
            persona_context: The persona LLM's context; the aggregators keep it
                up to date with both sides of the conversation.
        """
        self._bot_url = bot_url
        self._stream = stream
        self._trace = trace
        self._bot_audio = bot_audio
        self._user_audio = user_audio
        self._user_speech = user_speech
        self._capture_bot_audio = capture_bot_audio
        self._report_level = report_level
        self._vad_events = vad_events
        self._context = list(context or [])
        self._trigger_disconnect = trigger_disconnect
        self._connect_timeout_s = connect_timeout_s
        self._record_path = record_path
        self._stop_bot = stop_bot
        self._user_tts = user_tts
        self._bot_stt = bot_stt
        self._persona_llm = persona_llm
        self._persona_context = persona_context

        # The eval pipeline's worker (built by start()) and the runner task driving it.
        self._worker: PipelineWorker | None = None
        # Set once stop() begins: a disconnect from then on is our own teardown.
        self._stopping = False
        # Where the user's turns enter the pipeline (built with the processors).
        self._sink: _BotFrameSink | None = None
        self._run_task: asyncio.Task | None = None
        # Records the conversation audio (bot + user) when record_path is set and
        # the scenario is audio mode; fed raw audio by the transport, written on stop().
        self._recorder: HarnessRecorder | None = None
        # Set by the transport's on_bot_ready handler once the bot completes the
        # RTVI handshake; handshake() waits on it.
        self._bot_ready_event = asyncio.Event()
        self._next_id = 0

    @classmethod
    def for_scenario(
        cls, scenario: EvalScriptScenario, *, trigger_disconnect: bool = False, **kwargs
    ) -> "EvalClient":
        """A client for a scripted scenario, asking the bot for what its assertions need.

        Args:
            scenario: The scenario being run.
            trigger_disconnect: Run-wide opt-in; the scenario's own field also opts in.
            **kwargs: The remaining :class:`EvalClient` arguments.
        """
        return cls(
            bot_audio=scenario.bot_audio,
            user_audio=scenario.user_audio,
            user_speech=scenario.user_speech,
            capture_bot_audio=scenario.wants_response(),
            report_level=scenario.required_report_level(),
            vad_events=scenario.needs_vad_events(),
            context=scenario.context,
            trigger_disconnect=trigger_disconnect or scenario.trigger_disconnect,
            **kwargs,
        )

    @classmethod
    def for_simulation(
        cls, simulation: EvalSimulationScenario, *, trigger_disconnect: bool = False, **kwargs
    ) -> "EvalClient":
        """A client for a simulation: the persona hears the bot, and the judge sees its tool calls.

        Args:
            simulation: The simulation being run.
            trigger_disconnect: Run-wide opt-in; the simulation's own field also opts in.
            **kwargs: The remaining :class:`EvalClient` arguments, including the
                persona LLM and its context.
        """
        return cls(
            bot_audio=simulation.bot_audio,
            user_audio=simulation.user_audio,
            user_speech=simulation.user_speech,
            capture_bot_audio=simulation.bot_audio,
            # The bot's function calls, with their arguments, are the judge's
            # evidence of what the bot actually did.
            report_level="full",
            trigger_disconnect=trigger_disconnect or simulation.trigger_disconnect,
            **kwargs,
        )

    @property
    def has_user_tts(self) -> bool:
        """Whether text user turns are spoken to the bot rather than sent as text."""
        return self._user_tts is not None

    @property
    def sends_user_audio(self) -> bool:
        """Whether the user's side streams audio to the bot (the output is enabled)."""
        return self._user_tts is not None or self._user_audio

    async def wait_for_bot(self) -> None:
        """Wait until the bot's server accepts connections.

        The suite spawns a bot and immediately drives it, so this retries a
        lightweight TCP connect until the bot is listening; it is the only
        readiness wait. It deliberately does *not* complete the WebSocket/RTVI
        handshake: a full connect would fire the bot's ``on_client_connected``
        (kicking off a greeting and mutating its context) and then throw it away,
        leaving the real session with a duplicated opening. The transport owns
        the one real connection.

        Raises:
            OSError: The last connect error, when the bot never accepted within
                the connect timeout.
            TimeoutError: When the timeout passed without a single attempt failing.
        """
        u = urlsplit(self._bot_url)
        host, port = u.hostname or "localhost", u.port or (443 if u.scheme == "wss" else 80)
        deadline = time.monotonic() + self._connect_timeout_s
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
        user_audio_rate = tts_sample_rate(self._user_speech) if self._user_speech else 0
        params = WebsocketClientParams(
            audio_in_enabled=self._bot_audio,
            audio_out_enabled=self.sends_user_audio,
            audio_in_sample_rate=HARNESS_STT_SAMPLE_RATE if self._bot_audio else 0,
            audio_out_sample_rate=user_audio_rate,
            serializer=RTVIHarnessSerializer(),
        )
        # Record the conversation from the *raw* audio the transport sees on each
        # edge (the user TTS as produced, the bot's chunks as received), not the
        # paced/filled pipeline frames: Python can't hold the 40ms pacing tick
        # precisely, and recording the paced streams stutters. The recorder
        # reconstructs gapless turns and pads only the real between-turn pauses.
        if self._record_path and self._bot_audio:
            self._recorder = HarnessRecorder(user_audio_rate or HARNESS_STT_SAMPLE_RATE)
        # EvalHarnessTransport reshapes both audio edges into the continuous
        # real-time stream VAD/STT expect: its output paces the user TTS to the bot
        # and its input fills gaps in the bot's audio (both audio-mode only). When a
        # recorder is set, both edges also feed it the raw audio for the recording.
        transport = EvalHarnessTransport(self._connect_url(), params, recorder=self._recorder)

        @transport.event_handler("on_bot_ready")
        async def _on_bot_ready(_transport):
            self._bot_ready_event.set()

        @transport.event_handler("on_disconnected")
        async def _on_disconnected(_transport, _websocket):
            # The bot ended the call (a Flows bot's end_conversation, an
            # EndFrame) by closing the connection; our own teardown closes it
            # too, and that is not the bot's doing.
            if not self._stopping:
                await self._stream.append({"type": BOT_ENDED_EVENT})

        pipeline = Pipeline(self._processors(transport))
        # The StartFrame's rates drive the in-pipeline services: the bot-audio STT
        # reads `audio_in_sample_rate`, the user TTS produces `audio_out_sample_rate`.
        # Set them from the scenario so the user TTS synthesizes at the configured
        # user_audio rate rather than the PipelineParams `audio_out` default (a
        # mismatch that would mislabel the cached audio). Text-mode scenarios have no
        # audio in/out, so the STT rate is a harmless placeholder for `audio_out`.
        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=HARNESS_STT_SAMPLE_RATE,
                audio_out_sample_rate=user_audio_rate or HARNESS_STT_SAMPLE_RATE,
            ),
            enable_rtvi=False,
            cancel_on_idle_timeout=False,
        )
        self._worker = worker
        runner = WorkerRunner()
        await runner.add_workers(worker)
        self._run_task = asyncio.create_task(runner.run())

    async def handshake(self) -> None:
        """Wait for ``bot-ready``, then configure the bot and seed its context.

        ``bot-ready`` is a hard gate: the eval framework requires an RTVI bot, so a
        bot that never announces readiness either isn't a valid eval target or
        hasn't finished starting (services still connecting). Rather than fire
        turns at a half-started bot — which produces flaky, hard-to-read failures —
        this raises so the caller reports a clean connect-level failure.

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
        level = self._report_level
        vad = self._vad_events
        if level is not None or vad:
            config: dict = {}
            if level is not None:
                config["function_call_report_level"] = {"*": level}
            if vad:
                config["vad_user_speaking"] = True
            configure = RTVI.Message(
                type="client-message",
                id=self._message_id(),
                data={"t": EVAL_CONFIGURE_MESSAGE_TYPE, "d": config},
            )
            await self.send(configure)

        # Only send the eval-context when the scenario provides starting context.
        # An implicit empty one would race with bot startup flows (e.g. a greeting
        # added in on_client_connected), wiping the bot's context right after it
        # set it up.
        if self._context:
            context_message = RTVI.Message(
                type="client-message",
                id=self._message_id(),
                data={"t": EVAL_CONTEXT_MESSAGE_TYPE, "d": {"messages": self._context}},
            )
            await self.send(context_message)

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
        if self._stop_bot:
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

    def _text_turn_message(self, text: str) -> dict:
        """The RTVI ``send-text`` message for a user turn, as a message dict.

        The bot runs the LLM immediately and speaks its reply only in audio mode;
        in text mode the LLM bypasses TTS for this turn (content-only evals).
        """
        return RTVI.Message(
            type="send-text",
            id=self._message_id(),
            data=RTVI.SendTextData(
                content=text,
                options=RTVI.SendTextOptions(run_immediately=True, audio_response=self._bot_audio),
            ).model_dump(),
        ).model_dump()

    async def hang_up(self) -> None:
        """End the persona's part of the conversation: it answers nothing more."""
        assert self._sink is not None  # pipeline built before any send
        self._sink.hang_up()

    async def configure_persona(self, instruction: str) -> None:
        """Give the persona LLM its instruction and tell it how its replies go out.

        The instruction becomes the service's system instruction. In text mode
        the LLM's output carries ``skip_tts`` so the relay sends each response
        as one ``send-text``; in audio mode the user TTS speaks it.

        Args:
            instruction: The persona's system instruction.
        """
        assert self._sink is not None  # pipeline built before any send
        await self._sink.inject(
            LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=instruction))
        )
        await self._sink.inject(LLMConfigureOutputFrame(skip_tts=not self._user_audio))

    async def send_dtmf(self, keys: str) -> None:
        """Send a DTMF keypress turn as one RTVI ``dtmf`` message.

        The bot's ``RTVIProcessor`` turns each key into an ``InputDTMFFrame``
        pushed downstream, the same path a telephony transport's keypress takes.
        The bot's ``DTMFAggregator`` (if any) accumulates them and flushes — on
        the ``#`` terminator or its idle timeout — into a transcription the bot
        reacts to.

        Args:
            keys: The keys to press, in order.
        """
        message = RTVI.Message(
            type="dtmf",
            id=self._message_id(),
            data={"buttons": list(keys)},
        )
        await self.send(message)

    async def send_image(self, image_path: str) -> None:
        """Register an image (base64, with its MIME type) for the current turn.

        The eval transport serves it back as a ``UserImageRawFrame`` when the bot
        requests a user image. The file is sent as-is (already PNG/JPEG/...), so
        nothing is decoded or re-encoded.

        Args:
            image_path: Path to the image file.
        """
        path = Path(image_path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        self._trace.log(f"send: image {path.name} ({mime})")
        message = RTVI.Message(
            type="client-message",
            id=self._message_id(),
            data={"t": EVAL_IMAGE_MESSAGE_TYPE, "d": {"image": encoded, "format": mime}},
        )
        await self.send(message)

    async def say(self, text: str) -> None:
        """Speak ``text`` as the user by pushing a ``TTSSpeakFrame`` into the pipeline.

        The frame enters at the sink, on the user-audio side (see
        :class:`_BotFrameSink`). The user TTS
        (:class:`~pipecat.evals.tts.CachingTTSService`) renders it to audio
        (cached), which the output transport
        (:class:`~pipecat.evals.client_transport.EvalHarnessOutputTransport`) paces
        to the bot as a continuous real-time stream.

        Args:
            text: What the user says.
        """
        assert self._sink is not None  # pipeline built before any send
        await self._sink.inject(TTSSpeakFrame(text))

    async def play(self, path: str) -> None:
        """Play a recording to the bot as the user's turn, in place of synthesizing it.

        The recording is spoken exactly like a user TTS utterance: one audio frame
        bracketed by ``TTSStartedFrame`` / ``TTSStoppedFrame``, pushed into the
        pipeline at the sink like :meth:`say`. The output transport resamples it
        to the user-audio rate, paces it to the bot, flushes its final partial
        chunk on the stop frame, and records it.

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

    def _processors(self, transport: EvalHarnessTransport) -> list:
        """The eval pipeline's processors, in order (see the class docstring)."""
        processors: list = [transport.input()]
        # The context both aggregators keep: the persona's in a simulation, else a
        # throwaway that only gives the user aggregator somewhere to put turns.
        context = self._persona_context or LLMContext()
        aggregators: LLMContextAggregatorPair | None = None
        if self._bot_stt is not None:
            aggregators = LLMContextAggregatorPair(
                context,
                user_params=LLMUserAggregatorParams(
                    vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.4))
                ),
            )
            user_aggregator = aggregators.user()

            # The aggregator consumes the STT's TranscriptionFrames to build the
            # bot's turn, so the response comes from the aggregated turn text here
            # (not from a frame at the sink). The stream decides whether the
            # finished turn is the bot's reply or an earlier turn finalized late
            # (see EvalEventStream.bot_turn_stopped).
            @user_aggregator.event_handler("on_user_turn_started")
            async def _on_user_turn_started(_aggregator, _strategy):
                self._stream.bot_turn_started()

            @user_aggregator.event_handler("on_user_turn_stopped")
            async def _on_user_turn_stopped(_aggregator, _strategy, message):
                await self._stream.bot_turn_stopped(message.content or "")

            processors += [self._bot_stt, user_aggregator]
        # With no STT there is no aggregator to hand the bot's turns to the
        # persona; the sink does it from the bot's text.
        self._sink = _BotFrameSink(
            self._stream,
            persona=self._persona_context if self._persona_llm is not None else None,
            feed=self._bot_stt is None,
        )
        processors.append(self._sink)
        if self._persona_llm is not None:
            processors.append(self._persona_llm)
        if self._user_tts is not None:
            processors.append(self._user_tts)
        if self._persona_llm is not None:
            processors.append(_PersonaTurnRelay(self._text_turn_message, self._trace, self._stream))
        processors.append(transport.output())
        if self._persona_llm is not None:
            if aggregators is None:
                aggregators = LLMContextAggregatorPair(context)
            processors.append(aggregators.assistant())
        return processors

    def _connect_url(self) -> str:
        """Bot URL with the per-connection eval query flags.

        ``skip_tts`` (text mode) silences the bot before any LLM runs; the eval
        transport must read it at connect time because frames are ordered and a
        later message can't precede an on-connect greeting (see
        :mod:`pipecat.evals.transport`). ``capture_bot_audio`` makes the bot forward
        its synthesized audio to the harness, both for ``response``/``tts_response``
        transcription and so the harness can record the bot's side.
        ``trigger_disconnect`` asks the transport to fire the bot's
        ``on_client_disconnected`` handler when the connection ends (off by
        default, since bots often cancel there).
        """
        flags = []
        if not self._bot_audio:
            flags.append("skip_tts=true")
        # Forward the bot's audio when something listens to it (the response
        # transcription, a persona) or when recording an audio run.
        if self._capture_bot_audio or (self._record_path and self._bot_audio):
            flags.append("capture_bot_audio=true")
        if self._trigger_disconnect:
            flags.append("trigger_disconnect=true")
        if not flags:
            return self._bot_url
        sep = "&" if "?" in self._bot_url else "?"
        return f"{self._bot_url}{sep}{'&'.join(flags)}"

    def _message_id(self) -> str:
        self._next_id += 1
        return str(self._next_id)

    async def _send_cancel(self) -> None:
        """Ask the bot to cancel its pipeline so it shuts down gracefully.

        Best-effort: the connection may already be gone, in which case the
        orchestrator's kill fallback handles teardown.
        """
        try:
            message = RTVI.Message(
                type="client-message",
                id=self._message_id(),
                data={"t": EVAL_CANCEL_MESSAGE_TYPE, "d": {}},
            )
            await self.send(message)
        except Exception:
            pass

    async def _write_recording(self) -> None:
        """Write the recorded conversation audio (bot + user) to ``record_path``."""
        if self._recorder is None or not self._record_path or not self._recorder.has_audio():
            return
        if await self._recorder.write(self._record_path):
            self._trace.log(f"recording saved: {self._record_path}")

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
:class:`~pipecat.evals.events.EvalEventStream` the client is given.
"""

import asyncio
import base64
import mimetypes
import time
from pathlib import Path
from urllib.parse import urlsplit

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.evals.audio import load_user_audio
from pipecat.evals.client_transport import EvalHarnessTransport, HarnessRecorder
from pipecat.evals.events import EvalEventStream
from pipecat.evals.results import EvalTrace
from pipecat.evals.scenario import FUNCTION_CALL_EVENTS, EvalScenario
from pipecat.evals.serializer import (
    EVAL_CANCEL_MESSAGE_TYPE,
    EVAL_CONFIGURE_MESSAGE_TYPE,
    EVAL_CONTEXT_MESSAGE_TYPE,
    EVAL_IMAGE_MESSAGE_TYPE,
    HARNESS_STT_SAMPLE_RATE,
    RTVIHarnessSerializer,
)
from pipecat.evals.tts import CachingTTSService, tts_sample_rate
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InterruptionFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.stt_service import STTService
from pipecat.transports.websocket.client import WebsocketClientParams
from pipecat.workers.runner import WorkerRunner

BOT_READY_TIMEOUT_S = 10.0


class _BotFrameSink(FrameProcessor):
    """Pipeline tap that turns the bot's incoming frames into matcher events.

    Sits between the bot-audio side and the user-audio side of the eval pipeline;
    for every frame it calls the stream's
    :meth:`~pipecat.evals.events.EvalEventStream.frames_to_events` and appends
    the results with :meth:`~pipecat.evals.events.EvalEventStream.append`, then
    passes the frame on. Outgoing frames (the RTVI client messages the client
    injects) flow through untouched — they don't map to events.

    It also stops the harness's *computed* interruptions here: the user aggregator
    runs a VAD on the bot's incoming audio, so when the bot speaks it broadcasts an
    ``InterruptionFrame`` downstream. Letting that reach the user TTS / output would
    flush the user audio we're paced-sending (dropping a barge-in turn, and the
    user's side of the recording), so the sink swallows it.

    The user's turns enter the pipeline here as well (:meth:`inject`), straight
    into the user-audio side. A turn queued at the pipeline head would first
    cross the bot-audio side, and every processor there (this sink included)
    resets its queue when the aggregator's interruption reaches it, dropping a
    turn still queued behind a slow transcription at that moment.
    """

    def __init__(self, stream: EvalEventStream):
        super().__init__()
        self._stream = stream

    async def inject(self, frame: Frame) -> None:
        """Push a user-turn frame downstream, into the user TTS and the output.

        Args:
            frame: The frame to push (a ``TTSSpeakFrame``, or a spoken recording's
                TTS-bracketed frames).
        """
        await self.push_frame(frame)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        for event in self._stream.frames_to_events(frame):
            await self._stream.append(event)
        # The bot-audio VAD's interruption must not propagate into the user-audio
        # path (user TTS + output); see the class docstring.
        if isinstance(frame, InterruptionFrame) and direction == FrameDirection.DOWNSTREAM:
            return
        await self.push_frame(frame, direction)


class EvalClient:
    """The eval pipeline that talks to the bot, and the user's sends into it.

    Built once per run: :meth:`wait_for_bot`, :meth:`start`, :meth:`handshake`,
    the turns (:meth:`send_text`, :meth:`say`, :meth:`play`, :meth:`send_dtmf`,
    :meth:`send_image`), then :meth:`stop`. The pipeline is the same for both
    modalities::

        input -> [STT -> user aggregator] -> sink -> [user TTS] -> output

    The STT + aggregator (with the aggregator's own VAD) transcribe the bot's
    audio into the ``response``; the user TTS turns spoken turns into the audio
    sent to the bot. Each bracketed stage is present only when its service was
    built (audio scenarios); in text mode they're simply absent and no audio
    flows. The sink turns the bot's frames into events either way.
    """

    def __init__(
        self,
        *,
        scenario: EvalScenario,
        bot_url: str,
        stream: EvalEventStream,
        trace: EvalTrace,
        connect_timeout_s: float = 5.0,
        record_path: str | None = None,
        stop_bot: bool = False,
        trigger_disconnect: bool = False,
        user_tts: CachingTTSService | None = None,
        bot_stt: STTService | None = None,
    ):
        """Initialize the client.

        Args:
            scenario: The scenario being run; its modality, context, and
                assertions decide what the bot is asked to expose.
            bot_url: WebSocket URL of the bot's eval transport.
            stream: Where the bot's output is appended as events.
            trace: The run's trace.
            connect_timeout_s: How long :meth:`wait_for_bot` waits for the bot
                to accept connections.
            record_path: When set (and the scenario is audio mode), the
                conversation audio is recorded to this path on :meth:`stop`.
            stop_bot: When True, ask the bot to cancel its pipeline (and exit) on
                :meth:`stop` via ``eval-cancel``. The suite enables it to clean
                up each spawned bot.
            trigger_disconnect: When True (or when the scenario sets
                ``trigger_disconnect``), ask the eval transport to fire the bot's
                ``on_client_disconnected`` handler when this connection ends.
                Bots often cancel their pipeline there, so it is off by default
                to avoid that between scenarios.
            user_tts: The :class:`~pipecat.evals.tts.CachingTTSService` that
                synthesizes spoken user turns, or ``None`` for text-mode scenarios.
            bot_stt: The ``STTService`` that transcribes the bot's audio into the
                ``response`` event, or ``None`` when unused.
        """
        self._scenario = scenario
        self._bot_url = bot_url
        self._stream = stream
        self._trace = trace
        self._connect_timeout_s = connect_timeout_s
        self._record_path = record_path
        self._stop_bot = stop_bot
        # Either the run-wide flag or the scenario's own field opts in.
        self._trigger_disconnect = trigger_disconnect or scenario.trigger_disconnect
        self._user_tts = user_tts
        self._bot_stt = bot_stt
        # response (audio modality): the STT transcribes the bot's actual audio,
        # which the bot forwards only when asked (see _connect_url).
        self._wants_response = any(
            exp.event == "response" for turn in scenario.turns for exp in turn.expect
        )

        # The eval pipeline's worker (built by start()) and the runner task driving it.
        self._worker: PipelineWorker | None = None
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

    @property
    def has_user_tts(self) -> bool:
        """Whether text user turns are spoken to the bot rather than sent as text."""
        return self._user_tts is not None

    @property
    def sends_user_audio(self) -> bool:
        """Whether any user turn reaches the bot as audio, synthesized or from a file."""
        return self._user_tts is not None or any(t.audio for t in self._scenario.turns)

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
        user_audio_rate = (
            tts_sample_rate(self._scenario.user_speech) if self._scenario.user_speech else 0
        )
        params = WebsocketClientParams(
            audio_in_enabled=self._scenario.bot_audio,
            audio_out_enabled=self.sends_user_audio,
            audio_in_sample_rate=HARNESS_STT_SAMPLE_RATE if self._scenario.bot_audio else 0,
            audio_out_sample_rate=user_audio_rate,
            serializer=RTVIHarnessSerializer(),
        )
        # Record the conversation from the *raw* audio the transport sees on each
        # edge (the user TTS as produced, the bot's chunks as received), not the
        # paced/filled pipeline frames: Python can't hold the 40ms pacing tick
        # precisely, and recording the paced streams stutters. The recorder
        # reconstructs gapless turns and pads only the real between-turn pauses.
        if self._record_path and self._scenario.bot_audio:
            self._recorder = HarnessRecorder(user_audio_rate or HARNESS_STT_SAMPLE_RATE)
        # EvalHarnessTransport reshapes both audio edges into the continuous
        # real-time stream VAD/STT expect: its output paces the user TTS to the bot
        # and its input fills gaps in the bot's audio (both audio-mode only). When a
        # recorder is set, both edges also feed it the raw audio for the recording.
        transport = EvalHarnessTransport(self._connect_url(), params, recorder=self._recorder)

        @transport.event_handler("on_bot_ready")
        async def _on_bot_ready(_transport):
            self._bot_ready_event.set()

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
        level = self._required_report_level()
        vad = self._needs_vad_events()
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
        if self._scenario.context:
            context_message = RTVI.Message(
                type="client-message",
                id=self._message_id(),
                data={"t": EVAL_CONTEXT_MESSAGE_TYPE, "d": {"messages": self._scenario.context}},
            )
            await self.send(context_message)

    async def stop(self) -> None:
        """Save the recording, optionally cancel the bot, and end the pipeline."""
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

    async def send_text(self, text: str, *, audio_response: bool) -> None:
        """Send a text user turn via the RTVI ``send-text`` message.

        Args:
            text: The user's turn.
            audio_response: Whether the bot should speak its reply; when False the
                LLM bypasses TTS for this turn (content-only evals).
        """
        message = RTVI.Message(
            type="send-text",
            id=self._message_id(),
            data=RTVI.SendTextData(
                content=text,
                options=RTVI.SendTextOptions(run_immediately=True, audio_response=audio_response),
            ).model_dump(),
        )
        await self.send(message)

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
        if self._bot_stt is not None:
            user_aggregator = LLMContextAggregatorPair(
                LLMContext(),
                user_params=LLMUserAggregatorParams(
                    vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.4))
                ),
            ).user()

            # The aggregator consumes the STT's TranscriptionFrames to build the
            # bot's turn, so the response comes from the aggregated turn text here
            # (not from a frame at the sink). This is also the judge hook for sims.
            # Skip while awaiting an LLM restart: an interrupted turn finalizes
            # *after* the interruption, and that straggler must not be matched
            # against the next turn (mirrors the llm_response suppression).
            @user_aggregator.event_handler("on_user_turn_stopped")
            async def _on_user_turn_stopped(_aggregator, _strategy, message):
                if message.content and not self._stream.awaiting_llm_restart:
                    await self._stream.append({"type": "response", "text": message.content})

            processors += [self._bot_stt, user_aggregator]
        self._sink = _BotFrameSink(self._stream)
        processors.append(self._sink)
        if self._user_tts is not None:
            # The user TTS speaks only the harness's TTSSpeakFrames; the bot's text
            # flowing past it is passed through unspoken (see CachingTTSService), so
            # no echo and no gate needed.
            processors.append(self._user_tts)
        processors.append(transport.output())
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
        if not self._scenario.bot_audio:
            flags.append("skip_tts=true")
        # Forward the bot's audio when a scenario asserts on it (response) or when
        # recording an audio scenario (the harness records the bot's side from it).
        if self._wants_response or (self._record_path and self._scenario.bot_audio):
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

    def _required_report_level(self) -> str | None:
        """Minimal function-call report level the scenario's assertions need.

        Returns ``"full"`` if any ``function_call`` expectation checks ``args``,
        ``"name"`` if one checks ``name`` only, else ``None`` (no elevation —
        the bot's default applies and a function_call event still arrives).
        """
        needs_name = False
        for turn in self._scenario.turns:
            for exp in turn.expect:
                if exp.event not in FUNCTION_CALL_EVENTS:
                    continue
                # name/args live in exp.calls (the parser normalizes the single
                # name:/args: shorthand into it too).
                for call in exp.calls or []:
                    if call.args is not None:
                        return "full"
                    if call.name is not None:
                        needs_name = True
        return "name" if needs_name else None

    def _needs_vad_events(self) -> bool:
        """Whether the scenario references raw VAD speaking events.

        These (``vad_user_started_speaking`` / ``vad_user_stopped_speaking``) are
        off by default; the harness asks the bot's RTVIObserver to emit them only
        when a scenario asserts on or schedules from them.
        """
        vad_events = {"vad_user_started_speaking", "vad_user_stopped_speaking"}
        for turn in self._scenario.turns:
            if turn.send_after is not None and turn.send_after.event in vad_events:
                return True
            if any(exp.event in vad_events for exp in turn.expect):
                return True
        return False

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

#!/usr/bin/env python3
"""Standalone script to test Krisp STT and TTS end to end with a real audio file.

The audio file is streamed in real time through a pipecat pipeline:

    audio file -> KrispSTTService -> reply -> KrispTTSService -> output file

Two reply modes:

- echo (default): each user turn's transcript is spoken back by Krisp TTS.
  Needs only a Krisp API key.
- --llm: a full voice-agent pipeline (STT -> user aggregator -> OpenAI LLM ->
  TTS -> assistant aggregator). Exercises the turn strategies KrispSTTService
  recommends: Silero VAD starts user turns, Krisp end-of-turn ends them.

The script prints a timeline (transcripts, turn events, TTS words, latency) and
writes a stereo file with the user's audio on the left and the bot's audio on
the right. Bot audio is placed where it was received; since nothing plays it
back in real time, it can arrive faster than it would be heard.

Usage:
    python test_krisp_stt_tts_audiofile.py input.wav conversation.wav
    python test_krisp_stt_tts_audiofile.py input.wav conversation.wav --voice Clara -v
    python test_krisp_stt_tts_audiofile.py input.wav conversation.wav --llm
    python test_krisp_stt_tts_audiofile.py input.wav conversation.wav --report report.json

Requirements:
    uv add soundfile numpy
    For --llm: uv add "pipecat-ai[openai,silero]"
    Set KRISP_API_KEY (Krisp master API key); for --llm also OPENAI_API_KEY
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf  # noqa: F401
    from audio_file_utils import read_audio_file, write_audio_file
except ImportError as e:
    print(f"Error: Missing required dependencies: {e}")
    print("Install with: uv add soundfile numpy")
    sys.exit(1)

# Add src directory to Python path for development environment
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from loguru import logger  # noqa: E402

from pipecat.audio.utils import create_stream_resampler  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.worker import PipelineParams, PipelineWorker  # noqa: E402
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402
from pipecat.services.krisp.stt import KrispSTTService  # noqa: E402
from pipecat.services.krisp.tts import KRISP_TTS_SAMPLE_RATE, KrispTTSService  # noqa: E402
from pipecat.workers.runner import WorkerRunner  # noqa: E402

CHUNK_MS = 20
OUTPUT_SAMPLE_RATE = KRISP_TTS_SAMPLE_RATE


class Timeline:
    """Collects timestamped events and bot audio relative to the stream start."""

    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.t0 = time.monotonic()
        self.events: list[dict] = []
        self.bot_audio: list[tuple[float, bytes]] = []
        self.turn_end_t: float | None = None
        self.reply_latencies: list[float] = []

    def now(self) -> float:
        return time.monotonic() - self.t0

    def log(self, kind: str, text: str = "", show: bool = True, **extra):
        t = self.now()
        self.events.append({"t": round(t, 3), "event": kind, "text": text, **extra})
        if show:
            print(f"[{t:7.2f}s] {kind:<22} {text}")


class Probe(FrameProcessor):
    """Records the frames passing through it on the timeline.

    One probe sits at the head of the pipeline (``errors``: errors travel
    upstream), one right after STT (``stt``: transcripts and turn events, before
    the user aggregator consumes them) and one after TTS (``tts``).
    """

    def __init__(
        self,
        timeline: Timeline,
        *,
        errors: bool = False,
        stt: bool = False,
        tts: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._tl = timeline
        self._errors = errors
        self._stt = stt
        self._tts = tts
        self._words: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.UPSTREAM:
            if self._errors and isinstance(frame, ErrorFrame):
                self._tl.log("ERROR", str(frame.error))
        else:
            if self._stt:
                self._record_stt(frame)
            if self._tts:
                self._record_tts(frame)
        await self.push_frame(frame, direction)

    def _record_stt(self, frame: Frame):
        tl = self._tl
        if isinstance(frame, InterimTranscriptionFrame):
            tl.log("stt interim", frame.text, show=tl.verbose)
        elif isinstance(frame, TranscriptionFrame):
            tl.log("stt final", frame.text, language=str(frame.language or ""))
        elif isinstance(frame, ProposedUserStoppedSpeakingFrame):
            tl.log("krisp end of turn")
            tl.turn_end_t = tl.now()

    def _record_tts(self, frame: Frame):
        tl = self._tl
        if isinstance(frame, UserStartedSpeakingFrame):
            tl.log("user turn started")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            tl.log("user turn stopped")
        elif isinstance(frame, TTSStartedFrame):
            tl.log("tts started", show=tl.verbose)
            self._words = []
        elif isinstance(frame, TTSAudioRawFrame):
            if tl.turn_end_t is not None:
                latency = tl.now() - tl.turn_end_t
                tl.reply_latencies.append(latency)
                tl.log("first bot audio", f"{latency * 1000:.0f} ms after end of turn")
                tl.turn_end_t = None
            tl.bot_audio.append((tl.now(), frame.audio))
        elif isinstance(frame, TTSTextFrame):
            self._words.append(frame.text)
            tl.log("tts word", frame.text, show=tl.verbose)
        elif isinstance(frame, TTSStoppedFrame):
            tl.log("tts spoken", " ".join(self._words))


class EchoReply(FrameProcessor):
    """Speaks each final user transcript back through TTS."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            await self.push_frame(TTSSpeakFrame(text=frame.text))


def build_llm_processors():
    """Build the user aggregator, LLM and assistant aggregator for --llm mode."""
    try:
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.processors.aggregators.llm_context import LLMContext
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContextAggregatorPair,
            LLMUserAggregatorParams,
        )
        from pipecat.services.openai.llm import OpenAILLMService
    except ImportError as e:
        print(f"Error: --llm needs extra dependencies: {e}")
        print('Install with: uv add "pipecat-ai[openai,silero]"')
        sys.exit(1)

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("Error: OPENAI_API_KEY environment variable not set (required for --llm)")
        sys.exit(1)

    llm = OpenAILLMService(
        api_key=openai_key,
        settings=OpenAILLMService.Settings(
            system_instruction=(
                "You are a helpful assistant in a voice conversation. Your responses "
                "will be spoken aloud, so avoid formatting. Reply briefly."
            ),
        ),
    )
    # No user_turn_strategies: the aggregator uses the ones KrispSTTService
    # recommends (default turn start, Krisp end of turn).
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        LLMContext(), user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )
    return user_aggregator, llm, assistant_aggregator


async def feed_audio(
    worker: PipelineWorker,
    started: asyncio.Event,
    audio: np.ndarray,
    sample_rate: int,
    tail_s: float,
    timeline: Timeline,
):
    """Stream the file in real time, then silence, then end the pipeline."""
    await started.wait()
    timeline.t0 = time.monotonic()
    chunk = sample_rate * CHUNK_MS // 1000
    silence = np.zeros(int(sample_rate * tail_s), dtype=np.int16)
    samples = np.concatenate([audio, silence])
    timeline.log("audio start", f"{len(audio) / sample_rate:.2f}s + {tail_s:.1f}s silence")

    next_t = time.monotonic()
    for i in range(0, len(samples), chunk):
        if i == len(audio) - len(audio) % chunk:
            timeline.log("audio file ended", show=timeline.verbose)
        frame = InputAudioRawFrame(
            audio=samples[i : i + chunk].tobytes(), sample_rate=sample_rate, num_channels=1
        )
        await worker.queue_frame(frame)
        next_t += CHUNK_MS / 1000
        await asyncio.sleep(max(0.0, next_t - time.monotonic()))

    timeline.log("ending pipeline")
    await worker.queue_frame(EndFrame())


async def build_conversation(audio: np.ndarray, sample_rate: int, timeline: Timeline) -> np.ndarray:
    """Stereo track: user audio on the left, bot audio on the right (24 kHz)."""
    resampler = create_stream_resampler()
    user = np.frombuffer(
        await resampler.resample(audio.tobytes(), sample_rate, OUTPUT_SAMPLE_RATE), dtype=np.int16
    )
    total = len(user)
    placed: list[tuple[int, np.ndarray]] = []
    cursor = 0
    for t, chunk in timeline.bot_audio:
        samples = np.frombuffer(chunk, dtype=np.int16)
        start = max(int(t * OUTPUT_SAMPLE_RATE), cursor)
        placed.append((start, samples))
        cursor = start + len(samples)
        total = max(total, cursor)

    stereo = np.zeros((total, 2), dtype=np.int16)
    stereo[: len(user), 0] = user
    for start, samples in placed:
        stereo[start : start + len(samples), 1] = samples
    return stereo


async def run(args):
    api_key = args.api_key or os.getenv("KRISP_API_KEY")
    if not api_key:
        print("Error: KRISP_API_KEY environment variable not set (or pass --api-key)")
        sys.exit(1)

    audio, sample_rate = read_audio_file(args.input, verbose=True)
    timeline = Timeline(verbose=args.verbose)

    stt_settings = KrispSTTService.Settings()
    if args.language_hint:
        stt_settings.language = args.language_hint
    if args.vocabulary:
        stt_settings.vocabulary = args.vocabulary
    if args.turn_sensitivity is not None:
        stt_settings.turn_sensitivity = args.turn_sensitivity
    if args.turn_max_silence_ms is not None:
        stt_settings.turn_max_silence_ms = args.turn_max_silence_ms
    endpoints = {"rest_base_url": args.rest_base_url} if args.rest_base_url else {}
    stt = KrispSTTService(api_key=api_key, settings=stt_settings, **endpoints, **_url(args.stt_url))

    tts_settings = KrispTTSService.Settings(voice=args.voice, language=args.tts_language)
    if args.speed is not None:
        tts_settings.speed = args.speed
    tts = KrispTTSService(api_key=api_key, settings=tts_settings, **endpoints, **_url(args.tts_url))

    error_probe = Probe(timeline, errors=True)
    stt_probe = Probe(timeline, stt=True)
    tts_probe = Probe(timeline, tts=True)
    if args.llm:
        user_aggregator, llm, assistant_aggregator = build_llm_processors()
        processors = [
            error_probe,
            stt,
            stt_probe,
            user_aggregator,
            llm,
            tts,
            tts_probe,
            assistant_aggregator,
        ]
    else:
        processors = [error_probe, stt, stt_probe, EchoReply(), tts, tts_probe]

    print("\nPipeline:")
    print(f"  - Mode: {'LLM (OpenAI)' if args.llm else 'echo'}")
    print(f"  - Input: {args.input} ({sample_rate} Hz, {len(audio) / sample_rate:.2f}s)")
    print(f"  - TTS voice: {args.voice} ({args.tts_language})")
    print()

    worker = PipelineWorker(
        Pipeline(processors),
        params=PipelineParams(
            audio_in_sample_rate=sample_rate,
            audio_out_sample_rate=OUTPUT_SAMPLE_RATE,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def on_pipeline_started(worker, frame):
        started.set()

    runner = WorkerRunner(handle_sigint=True)
    await runner.add_workers(worker)
    feeder = asyncio.create_task(
        feed_audio(worker, started, audio, sample_rate, args.tail, timeline)
    )
    try:
        await runner.run()
    finally:
        feeder.cancel()

    finals = [e["text"] for e in timeline.events if e["event"] == "stt final"]
    spoken = [e["text"] for e in timeline.events if e["event"] == "tts spoken"]
    errors = [e["text"] for e in timeline.events if e["event"] == "ERROR"]
    latencies = timeline.reply_latencies

    print("\nSummary:")
    print(f"  - User turns transcribed: {len(finals)}")
    for text in finals:
        print(f"      > {text}")
    print(f"  - Bot replies spoken: {len(spoken)}")
    for text in spoken:
        print(f"      < {text}")
    if latencies:
        print(
            f"  - End of turn -> first bot audio: avg {np.mean(latencies) * 1000:.0f} ms, "
            f"max {np.max(latencies) * 1000:.0f} ms"
        )
    if errors:
        print(f"  - Errors: {len(errors)}")
        for text in errors:
            print(f"      ! {text}")

    if timeline.bot_audio or len(audio):
        stereo = await build_conversation(audio, sample_rate, timeline)
        write_audio_file(args.output, stereo, OUTPUT_SAMPLE_RATE, verbose=True)
        print("  (left channel: user input, right channel: bot)")

    if args.report:
        report = {
            "input": args.input,
            "mode": "llm" if args.llm else "echo",
            "events": timeline.events,
            "reply_latencies_ms": [round(x * 1000) for x in latencies],
        }
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"Report written to: {args.report}")

    if errors or not finals:
        print("\n⚠️  Check the output: errors occurred or nothing was transcribed.")
        sys.exit(1)
    print("\n✅ Processing complete!")


def _url(url: str | None) -> dict:
    return {"url": url} if url else {}


def main():
    parser = argparse.ArgumentParser(
        description="Test Krisp STT and TTS end to end with a real audio file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_krisp_stt_tts_audiofile.py question.wav conversation.wav
  python test_krisp_stt_tts_audiofile.py question.wav conversation.wav --voice Gero --tts-language de-DE
  python test_krisp_stt_tts_audiofile.py question.wav conversation.wav --llm --report report.json

Required env var:  KRISP_API_KEY  — Krisp master API key (session tokens are created from it)
For --llm:         OPENAI_API_KEY
        """,
    )
    parser.add_argument("input", help="Input audio file path (speech)")
    parser.add_argument("output", help="Output stereo audio file (.wav/.flac/.ogg)")
    parser.add_argument("--api-key", default="", help="Krisp API key (default: KRISP_API_KEY)")
    parser.add_argument("--llm", action="store_true", help="Reply with an OpenAI LLM")
    parser.add_argument("--voice", default="Amy", help="Krisp TTS voice name (default: Amy)")
    parser.add_argument("--tts-language", default="en-US", help="TTS language (default: en-US)")
    parser.add_argument("--speed", type=float, default=None, help="TTS speed: 0.9-1.1")
    parser.add_argument("--language-hint", default=None, help="STT language hint, e.g. hy")
    parser.add_argument("--vocabulary", nargs="*", default=None, help="STT custom vocabulary")
    parser.add_argument("--turn-sensitivity", type=float, default=None, help="0-1")
    parser.add_argument("--turn-max-silence-ms", type=int, default=None, help="400-4000")
    parser.add_argument(
        "--tail",
        type=float,
        default=6.0,
        help="Seconds of silence streamed after the file so turns end and TTS finishes "
        "(default: 6.0)",
    )
    parser.add_argument(
        "--stt-url",
        default=os.getenv("KRISP_STT_URL"),
        help="Override the STT WebSocket URL (default: KRISP_STT_URL env var)",
    )
    parser.add_argument(
        "--tts-url",
        default=os.getenv("KRISP_TTS_URL"),
        help="Override the TTS WebSocket URL (default: KRISP_TTS_URL env var)",
    )
    parser.add_argument(
        "--rest-base-url",
        default=os.getenv("KRISP_REST_BASE_URL"),
        help="Override the Krisp REST base URL (default: KRISP_REST_BASE_URL env var)",
    )
    parser.add_argument("--report", default=None, help="Write a JSON timeline report here")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show interims, words and pipecat logs"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if args.verbose else "WARNING")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()

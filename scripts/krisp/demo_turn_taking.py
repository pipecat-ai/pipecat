#!/usr/bin/env python3
"""Turn-taking demonstration tool for comparing turn analyzers.

Processes an audio file through one or more turn analyzers side by side,
producing:
  - Annotated WAV files with beeps at detected turn points
  - ASCII timeline visualization in the terminal
  - Per-turn comparison table with verdicts
  - Self-contained HTML report with interactive timeline
  - Text timeline reports

Replicates the real Pipecat production pipeline: uses the same VAD engine
(Silero or Krisp VIVA) with the same default parameters (confidence=0.7,
start_secs=0.2, stop_secs=0.2, min_volume=0.6), and optionally applies
the Krisp VIVA noise filter before processing -- exactly as the transport
does in BaseInputTransport._audio_task_handler.

Usage:
    python demo_turn_taking.py input.wav
    python demo_turn_taking.py input.wav --analyzer krisp --analyzer smart-turn-v3
    python demo_turn_taking.py input.wav --viva-filter --threshold 0.7 -v

Requirements:
    pip install soundfile numpy pipecat-ai[krisp]
    Set KRISP_VIVA_TURN_MODEL_PATH environment variable for Krisp analyzer
    Set KRISP_VIVA_FILTER_MODEL_PATH for --viva-filter option
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Load .env file from the script directory if present
script_dir = Path(__file__).parent
_env_file = script_dir / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _key, _val = _key.strip(), _val.strip()
                if _key and _key not in os.environ:
                    os.environ[_key] = _val

try:
    import numpy as np
    import soundfile as sf  # noqa: F401
    from audio_file_utils import read_audio_file, resample_audio, write_audio_file
except ImportError as e:
    print(f"Error: Missing required dependencies: {e}")
    print("Install with: pip install soundfile numpy")
    sys.exit(1)

# Add src directory to Python path for development environment
project_root = script_dir.parent.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState

from demo_formatting import (
    format_ascii_timeline,
    format_comparison_table,
    format_summary,
    format_timeline,
    method_symbol,
)
from demo_html_report import generate_html_report
from demo_types import (
    METHOD_ON_DEMAND,
    METHOD_STREAMING,
    METHOD_TIMEOUT,
    AnalyzerResult,
    TurnEvent,
)


AVAILABLE_ANALYZERS = ["krisp", "smart-turn-v3"]
AVAILABLE_VADS = ["silero", "krisp"]


def create_vad(
    vad_type: str,
    params: VADParams,
    sample_rate: int,
    frame_duration_ms: int = 10,
) -> Tuple[VADAnalyzer, str]:
    """Create and configure a VAD analyzer by name.

    Args:
        vad_type: VAD engine ("silero" or "krisp").
        params: VAD detection parameters.
        sample_rate: Audio sample rate in Hz.
        frame_duration_ms: Frame duration for Krisp VAD (default: 10).

    Returns:
        Tuple of (VADAnalyzer instance, actual vad_type used).
    """
    if vad_type == "silero":
        from pipecat.audio.vad.silero import SileroVADAnalyzer

        vad = SileroVADAnalyzer(params=params)
        vad.set_sample_rate(sample_rate)
        return vad, "silero"

    elif vad_type == "krisp":
        from pipecat.audio.vad.krisp_viva_vad import KrispVivaVadAnalyzer

        vad = KrispVivaVadAnalyzer(frame_duration=frame_duration_ms, params=params)
        vad.set_sample_rate(sample_rate)
        return vad, "krisp"

    else:
        raise ValueError(f"Unknown VAD type '{vad_type}'. Available: {AVAILABLE_VADS}")


def create_analyzer(
    name: str, threshold: float, frame_duration_ms: int, sample_rate: int
) -> BaseTurnAnalyzer:
    """Create and configure a turn analyzer by name.

    Args:
        name: Analyzer identifier ("krisp" or "smart-turn-v3").
        threshold: Probability threshold for turn completion.
        frame_duration_ms: Frame duration in ms (used by Krisp).
        sample_rate: Audio sample rate in Hz.

    Returns:
        Configured BaseTurnAnalyzer instance.
    """
    if name == "krisp":
        from pipecat.audio.turn.krisp_viva_turn import KrispTurnParams, KrispVivaTurn

        model_path = os.getenv("KRISP_VIVA_TURN_MODEL_PATH")
        if not model_path:
            raise ValueError("KRISP_VIVA_TURN_MODEL_PATH environment variable not set")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Krisp turn model not found: {model_path}")

        params = KrispTurnParams(threshold=threshold, frame_duration_ms=frame_duration_ms)
        analyzer = KrispVivaTurn(model_path=model_path, params=params)
        analyzer.set_sample_rate(sample_rate)
        return analyzer

    elif name == "smart-turn-v3":
        from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
        from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3

        params = SmartTurnParams(stop_secs=3.0)
        analyzer = LocalSmartTurnAnalyzerV3(params=params)
        analyzer.set_sample_rate(sample_rate)
        return analyzer

    else:
        raise ValueError(f"Unknown analyzer '{name}'. Available: {AVAILABLE_ANALYZERS}")


def is_smart_turn(analyzer: BaseTurnAnalyzer) -> bool:
    """Check if an analyzer is a BaseSmartTurn subclass (needs analyze_end_of_turn)."""
    from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn

    return isinstance(analyzer, BaseSmartTurn)


def apply_viva_filter(
    audio_data: np.ndarray, sample_rate: int, frame_duration_ms: int = 10
) -> np.ndarray:
    """Apply Krisp VIVA noise filter to audio data.

    Args:
        audio_data: Audio samples as int16 numpy array.
        sample_rate: Audio sample rate in Hz.
        frame_duration_ms: Filter frame duration in ms.

    Returns:
        Filtered audio as int16 numpy array.
    """
    import krisp_audio

    from pipecat.audio.krisp_instance import (
        KrispVivaSDKManager,
        int_to_krisp_frame_duration,
        int_to_krisp_sample_rate,
    )

    model_path = os.getenv("KRISP_VIVA_FILTER_MODEL_PATH") or os.getenv("KRISP_VIVA_MODEL_PATH")
    if not model_path:
        raise ValueError("KRISP_VIVA_FILTER_MODEL_PATH environment variable not set")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Krisp filter model not found: {model_path}")

    KrispVivaSDKManager.acquire()
    try:
        model_info = krisp_audio.ModelInfo()
        model_info.path = model_path

        nc_cfg = krisp_audio.NcSessionConfig()
        nc_cfg.inputSampleRate = int_to_krisp_sample_rate(sample_rate)
        nc_cfg.inputFrameDuration = int_to_krisp_frame_duration(frame_duration_ms)
        nc_cfg.outputSampleRate = nc_cfg.inputSampleRate
        nc_cfg.modelInfo = model_info

        samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
        session = krisp_audio.NcInt16.create(nc_cfg)

        output = np.empty_like(audio_data)
        num_complete = len(audio_data) // samples_per_frame

        for i in range(num_complete):
            start = i * samples_per_frame
            end = start + samples_per_frame
            cleaned = session.process(audio_data[start:end], 100)
            output[start:end] = cleaned

        remainder_start = num_complete * samples_per_frame
        output[remainder_start:] = audio_data[remainder_start:]

        return output
    finally:
        KrispVivaSDKManager.release()


def generate_beep(
    sample_rate: int, freq: int = 1000, duration_ms: int = 50, amplitude: float = 0.3
) -> np.ndarray:
    """Generate a sine-wave beep tone as int16 samples."""
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
    beep = np.sin(2 * np.pi * freq * t)
    fade_samples = min(num_samples // 10, int(sample_rate * 0.005))
    if fade_samples > 0:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        beep[:fade_samples] *= fade_in
        beep[-fade_samples:] *= fade_out
    return (amplitude * 32767 * beep).astype(np.int16)


def mix_beeps(
    audio_data: np.ndarray,
    sample_rate: int,
    turn_timestamps: List[float],
    beep: np.ndarray,
) -> np.ndarray:
    """Mix beep tones into audio at the given timestamps."""
    output = audio_data.copy()
    for ts in turn_timestamps:
        start = int(ts * sample_rate)
        end = min(start + len(beep), len(output))
        if start < len(output):
            segment_len = end - start
            mixed = output[start:end].astype(np.int32) + beep[:segment_len].astype(np.int32)
            output[start:end] = np.clip(mixed, -32768, 32767).astype(np.int16)
    return output


async def process_audio(
    input_path: str,
    analyzer_names: List[str],
    threshold: float = 0.5,
    frame_duration_ms: int = 20,
    beep_freq: int = 1000,
    beep_duration_ms: int = 50,
    output_dir: str = "./demo_output",
    verbose: bool = False,
    use_viva_filter: bool = False,
    vad_type: str = "silero",
) -> None:
    """Process an audio file through turn analyzers and produce outputs.

    Uses the same VAD parameters as the production Pipecat pipeline
    to replicate real-life behavior.
    """
    audio_data, sample_rate = read_audio_file(input_path, verbose=True)
    duration_secs = len(audio_data) / sample_rate

    print(f"\nAudio: {duration_secs:.2f}s, {sample_rate} Hz")

    # Resample to 16 kHz if needed (Silero VAD only supports 8/16 kHz)
    audio_data, sample_rate = resample_audio(audio_data, sample_rate, 16000, verbose=True)
    duration_secs = len(audio_data) / sample_rate

    # Optionally apply VIVA noise filter (same as pipeline's KrispVivaFilter)
    if use_viva_filter:
        print("\nApplying VIVA noise filter...")
        t0 = time.time()
        try:
            audio_data = apply_viva_filter(audio_data, sample_rate, frame_duration_ms=10)
            filter_ms = (time.time() - t0) * 1000
            print(f"  Filter applied in {filter_ms:.1f}ms")
        except Exception as e:
            print(f"  Error applying VIVA filter: {e}")
            print("  Continuing without noise filtering")
            use_viva_filter = False

    # Production VAD defaults (same as Pipecat pipeline)
    print(f"\nInitializing {vad_type} VAD...")
    vad_params = VADParams(
        confidence=0.7,
        start_secs=0.2,
        stop_secs=0.2,
        min_volume=0.6,
    )
    vad, vad_type = create_vad(vad_type, vad_params, sample_rate, frame_duration_ms=10)
    print(
        f"  {vad_type} VAD ready (confidence={vad_params.confidence}, "
        f"start={vad_params.start_secs}s, stop={vad_params.stop_secs}s)"
    )

    # Create turn analyzers
    analyzers: Dict[str, BaseTurnAnalyzer] = {}
    results: Dict[str, AnalyzerResult] = {}

    for name in analyzer_names:
        print(f"\nInitializing analyzer: {name}...")
        t0 = time.time()
        try:
            analyzer = create_analyzer(name, threshold, frame_duration_ms, sample_rate)
            analyzers[name] = analyzer
            init_ms = (time.time() - t0) * 1000
            timeout_s = analyzer.params.stop_secs if is_smart_turn(analyzer) else None
            results[name] = AnalyzerResult(
                name=name, init_time_ms=init_ms, timeout_secs=timeout_s
            )
            print(f"  {name} initialized in {init_ms:.1f}ms")
        except Exception as e:
            print(f"  Error initializing {name}: {e}")
            print(f"  Skipping {name}")

    if not analyzers:
        print("Error: No analyzers initialized successfully.")
        sys.exit(1)

    frame_size_samples = int(sample_rate * frame_duration_ms / 1000)

    print("\nProcessing audio...")
    print(f"  Frame size: {frame_size_samples} samples ({frame_duration_ms}ms)")

    # State tracking
    speech_segments: List[Tuple[float, float]] = []
    current_speech_start: Optional[float] = None
    prev_vad_state: VADState = VADState.QUIET
    vad_speaking = False

    # Per-analyzer state
    silence_start: Dict[str, Optional[float]] = {name: None for name in analyzers}
    eot_called: Dict[str, bool] = {name: False for name in analyzers}

    audio_buffer = np.array([], dtype=np.int16)
    frames_processed = 0

    for i in range(0, len(audio_data), frame_size_samples):
        chunk = audio_data[i : i + frame_size_samples]
        if len(chunk) == 0:
            break

        audio_buffer = np.concatenate([audio_buffer, chunk])

        while len(audio_buffer) >= frame_size_samples:
            frame_samples = audio_buffer[:frame_size_samples].copy()
            audio_buffer = audio_buffer[frame_size_samples:]

            timestamp = frames_processed * frame_duration_ms / 1000.0
            frames_processed += 1

            frame_bytes = frame_samples.tobytes()

            # Run VAD
            vad_state = vad._run_analyzer(frame_bytes)

            # Detect VAD state transitions (same as pipeline frame events)
            vad_just_started = (
                prev_vad_state != VADState.SPEAKING and vad_state == VADState.SPEAKING
            )
            vad_just_stopped = (
                prev_vad_state != VADState.QUIET
                and vad_state == VADState.QUIET
                and vad_speaking
            )

            if vad_just_started:
                vad_speaking = True
                current_speech_start = timestamp

            if vad_just_stopped:
                vad_speaking = False
                if current_speech_start is not None:
                    # VAD confirmed silence after stop_secs; actual speech
                    # likely ended stop_secs earlier.  Adjust the segment
                    # end so the visual gap between speech and turn markers
                    # reflects real latency for all analyzers.
                    estimated_speech_end = max(
                        current_speech_start, timestamp - vad_params.stop_secs
                    )
                    speech_segments.append((current_speech_start, estimated_speech_end))
                    current_speech_start = None

            is_speech = vad_speaking

            # Feed all analyzers
            for name, analyzer in analyzers.items():
                state = analyzer.append_audio(frame_bytes, is_speech)

                # Track silence start for detection delay
                if vad_just_started:
                    silence_start[name] = None
                    eot_called[name] = False
                elif vad_just_stopped:
                    silence_start[name] = timestamp

                if state == EndOfTurnState.COMPLETE:
                    method = METHOD_TIMEOUT if is_smart_turn(analyzer) else METHOD_STREAMING
                    vad_wait = vad_params.stop_secs if is_smart_turn(analyzer) else None
                    event = TurnEvent(
                        timestamp=timestamp,
                        silence_start=silence_start[name],
                        method=method,
                        vad_stop_secs=vad_wait,
                    )
                    results[name].turn_events.append(event)
                    silence_start[name] = None
                    eot_called[name] = True
                    if verbose:
                        d = event.detection_delay
                        sym = method_symbol(method)
                        d_str = f" (delay: {d:.3f}s [{sym}])" if d is not None else ""
                        print(f"  [{name}] Turn at {timestamp:.2f}s{d_str}")

                # SmartTurn: call analyze_end_of_turn when VAD stops
                if (
                    vad_just_stopped
                    and is_smart_turn(analyzer)
                    and analyzer.speech_triggered
                    and not eot_called[name]
                ):
                    eot_state, _ = await analyzer.analyze_end_of_turn()
                    eot_called[name] = True
                    if eot_state == EndOfTurnState.COMPLETE:
                        event = TurnEvent(
                            timestamp=timestamp,
                            silence_start=silence_start[name],
                            method=METHOD_ON_DEMAND,
                            vad_stop_secs=vad_params.stop_secs,
                        )
                        results[name].turn_events.append(event)
                        silence_start[name] = None
                        if verbose:
                            d = event.detection_delay
                            d_str = f" (delay: {d:.3f}s [D])" if d is not None else ""
                            print(f"  [{name}] Turn at {timestamp:.2f}s{d_str}")

            prev_vad_state = vad_state

        # Progress
        if i % (frame_size_samples * 100) == 0:
            progress = (i / len(audio_data)) * 100
            print(f"  Progress: {progress:.1f}%", end="\r")

    print("  Progress: 100.0%")

    # Final speech segment
    if current_speech_start is not None:
        speech_segments.append((current_speech_start, duration_secs))

    # ---- Generate outputs ----
    os.makedirs(output_dir, exist_ok=True)
    input_stem = Path(input_path).stem
    beep = generate_beep(sample_rate, freq=beep_freq, duration_ms=beep_duration_ms)

    # ASCII timeline (printed to terminal)
    ascii_tl = format_ascii_timeline(duration_secs, speech_segments, results)
    print(ascii_tl)

    print(f"Generating outputs in: {output_dir}/")

    annotated_audio_map: Dict[str, np.ndarray] = {}

    for name, result in results.items():
        timeline = format_timeline(
            analyzer_name=name,
            input_path=input_path,
            sample_rate=sample_rate,
            duration_secs=duration_secs,
            threshold=threshold,
            frame_duration_ms=frame_duration_ms,
            speech_segments=speech_segments,
            result=result,
            viva_filter_used=use_viva_filter,
        )

        print(f"\n{timeline}")

        timeline_path = os.path.join(output_dir, f"{input_stem}_{name}_timeline.txt")
        with open(timeline_path, "w", encoding="utf-8") as f:
            f.write(timeline)
        print(f"  Timeline saved: {timeline_path}")

        turn_timestamps = [e.timestamp for e in result.turn_events]
        annotated = mix_beeps(audio_data, sample_rate, turn_timestamps, beep)
        annotated_audio_map[name] = annotated
        wav_path = os.path.join(output_dir, f"{input_stem}_{name}_annotated.wav")
        write_audio_file(wav_path, annotated, sample_rate, verbose=True)
        print(f"  Annotated WAV: {wav_path}")

    # Comparison outputs (multiple analyzers)
    if len(results) > 1:
        cmp_table = format_comparison_table(speech_segments, results)
        print(cmp_table)

        summary = format_summary(results)
        print(summary)

    # HTML report (with embedded audio players)
    vad_label = "Krisp VIVA" if vad_type == "krisp" else "Silero"
    vad_info = (
        f"{vad_label} (confidence={vad_params.confidence}, "
        f"start={vad_params.start_secs}s, stop={vad_params.stop_secs}s)"
    )
    html_path = os.path.join(output_dir, f"{input_stem}_report.html")
    generate_html_report(
        input_path=input_path,
        duration_secs=duration_secs,
        sample_rate=sample_rate,
        speech_segments=speech_segments,
        results=results,
        viva_filter_used=use_viva_filter,
        output_path=html_path,
        annotated_audio=annotated_audio_map,
        vad_info=vad_info,
    )
    print(f"\n  HTML report: {html_path}")

    # Cleanup
    for name, analyzer in analyzers.items():
        analyzer.clear()
        await analyzer.cleanup()
    if hasattr(vad, "cleanup"):
        await vad.cleanup()

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Turn-taking demo: compare turn analyzers on audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_turn_taking.py conversation.wav
  python demo_turn_taking.py input.wav --analyzer krisp
  python demo_turn_taking.py input.wav --analyzer krisp --analyzer smart-turn-v3
  python demo_turn_taking.py input.wav --vad krisp --viva-filter
  python demo_turn_taking.py input.wav --viva-filter --threshold 0.7 -v

Environment variables:
  KRISP_VIVA_TURN_MODEL_PATH    Path to Krisp turn detection model (.kef)
  KRISP_VIVA_FILTER_MODEL_PATH  Path to Krisp noise filter model (.kef)
  KRISP_VIVA_VAD_MODEL_PATH     Path to Krisp VAD model (.kef, for --vad krisp)
        """,
    )

    parser.add_argument("input", help="Input audio file path")
    parser.add_argument(
        "--analyzer",
        action="append",
        choices=AVAILABLE_ANALYZERS,
        dest="analyzers",
        help=f"Analyzer(s) to run (repeatable, default: all). Choices: {AVAILABLE_ANALYZERS}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Turn probability threshold (0.0 to 1.0, default: 0.5)",
    )
    parser.add_argument(
        "--frame-duration",
        type=int,
        default=20,
        choices=[10, 15, 20, 30, 32],
        help="Frame duration in ms for Krisp analyzer (default: 20)",
    )
    parser.add_argument(
        "--viva-filter",
        action="store_true",
        help="Apply Krisp VIVA noise filter before processing "
        "(requires KRISP_VIVA_FILTER_MODEL_PATH)",
    )
    parser.add_argument(
        "--beep-freq",
        type=int,
        default=1000,
        help="Beep frequency in Hz (default: 1000)",
    )
    parser.add_argument(
        "--beep-duration",
        type=int,
        default=50,
        help="Beep duration in ms (default: 50)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./demo_output",
        help="Output directory (default: ./demo_output)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-frame turn detection events",
    )

    parser.add_argument(
        "--vad",
        type=str,
        default="silero",
        choices=AVAILABLE_VADS,
        dest="vad_type",
        help="VAD engine: silero (default) or krisp (requires KRISP_VIVA_VAD_MODEL_PATH). "
        "Uses production Pipecat VAD parameters.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if not 0.0 <= args.threshold <= 1.0:
        print(f"Error: Threshold must be between 0.0 and 1.0, got {args.threshold}")
        sys.exit(1)

    analyzer_names = args.analyzers or AVAILABLE_ANALYZERS

    asyncio.run(
        process_audio(
            input_path=args.input,
            analyzer_names=analyzer_names,
            threshold=args.threshold,
            frame_duration_ms=args.frame_duration,
            beep_freq=args.beep_freq,
            beep_duration_ms=args.beep_duration,
            output_dir=args.output_dir,
            verbose=args.verbose,
            use_viva_filter=args.viva_filter,
            vad_type=args.vad_type,
        )
    )


if __name__ == "__main__":
    main()

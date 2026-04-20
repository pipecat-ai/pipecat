#!/usr/bin/env python3
"""Interrupt prediction demonstration tool.

Compares interruption handling strategies side by side on a pre-recorded
audio file, similar to how demo_turn_taking.py compares turn analyzers.

Available strategies (simulated offline):

  krisp-ip   Krisp VIVA Interruption Prediction model. Distinguishes
             genuine interruptions from backchannels -- only triggers
             when the IP probability exceeds the threshold.

  vad        VAD-only (equivalent to VADUserTurnStartStrategy). Every
             VAD speech start triggers an interruption immediately.
             This is the most aggressive strategy -- it interrupts the
             bot on any user speech, including backchannels.

For each strategy the tool produces an annotated WAV file where:
  - A beep marks each detected interruption point
  - The original user audio is preserved for listening

Outputs:
  - Per-strategy annotated WAV files for A/B listening
  - ASCII timeline in the terminal
  - Self-contained HTML report with interactive timeline and audio playback

Usage:
    python demo_interrupt_prediction.py input.wav
    python demo_interrupt_prediction.py input.wav --strategy krisp-ip --strategy vad
    python demo_interrupt_prediction.py input.wav --vad krisp
    python demo_interrupt_prediction.py input.wav --threshold 0.6 -v

Requirements:
    pip install soundfile numpy pipecat-ai[krisp]
    Set KRISP_VIVA_IP_MODEL_PATH environment variable for krisp-ip strategy
    Set KRISP_VIVA_VAD_MODEL_PATH for --vad krisp option
"""

import argparse
import asyncio
import base64
import io
import os
import sys
import time
from dataclasses import dataclass, field
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
    import soundfile as sf
    from audio_file_utils import read_audio_file, resample_audio, write_audio_file
except ImportError as e:
    print(f"Error: Missing required dependencies: {e}")
    print("Install with: pip install soundfile numpy")
    sys.exit(1)

project_root = script_dir.parent.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState
from pipecat.frames.frames import InputAudioRawFrame
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.krisp_viva_ip_user_turn_start_strategy import (
    KrispVivaIPUserTurnStartStrategy,
)


AVAILABLE_STRATEGIES = ["krisp-ip", "vad"]
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


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class StrategyResult:
    """Results for one interruption strategy."""

    name: str
    interruption_times: List[float] = field(default_factory=list)
    init_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _encode_wav_base64(audio: np.ndarray, sample_rate: int) -> str:
    """Encode int16 audio as a base64 WAV data URI."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _generate_beep(
    sample_rate: int, freq: int = 1200, duration_ms: int = 80, amplitude: float = 0.4
) -> np.ndarray:
    """Generate a short marker beep."""
    num_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, num_samples, endpoint=False)
    beep = np.sin(2 * np.pi * freq * t)
    fade = min(num_samples // 10, int(sample_rate * 0.005))
    if fade > 0:
        beep[:fade] *= np.linspace(0, 1, fade)
        beep[-fade:] *= np.linspace(1, 0, fade)
    return (amplitude * 32767 * beep).astype(np.int16)


def _mix_audio(base: np.ndarray, overlay: np.ndarray, start: int = 0) -> np.ndarray:
    """Mix overlay into base at a given sample offset, clipping to int16."""
    end = min(start + len(overlay), len(base))
    seg_len = end - start
    if seg_len <= 0:
        return base
    mixed = base[start:end].astype(np.int32) + overlay[:seg_len].astype(np.int32)
    base[start:end] = np.clip(mixed, -32768, 32767).astype(np.int16)
    return base


def _build_annotated_audio(
    user_audio: np.ndarray,
    sample_rate: int,
    interruption_times: List[float],
    beep: np.ndarray,
) -> np.ndarray:
    """Build annotated audio: user audio with beep markers at interruption points."""
    output = user_audio.copy()
    for t_int in interruption_times:
        output = _mix_audio(output, beep, int(t_int * sample_rate))
    return output


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


async def process_audio(
    input_path: str,
    strategy_names: List[str],
    model_path: Optional[str] = None,
    threshold: float = 0.5,
    frame_duration_ms: int = 20,
    output_dir: str = "./demo_output",
    verbose: bool = False,
    vad_type: str = "silero",
) -> None:
    """Process audio through multiple interruption strategies and compare.

    Uses ``KrispVivaIPUserTurnStartStrategy`` from the pipecat integration
    rather than calling the Krisp SDK directly, so the demo always stays in
    sync with the pipeline behaviour.

    Args:
        input_path: Path to user audio file.
        strategy_names: List of strategy names to compare.
        model_path: Path to Krisp IP model (.kef), required for krisp-ip.
        threshold: IP probability threshold for krisp-ip strategy.
        output_dir: Output directory for generated files.
        verbose: Print per-frame details.
        vad_type: VAD engine ("silero" or "krisp").
    """
    user_audio, sample_rate = read_audio_file(input_path, verbose=True)
    duration_secs = len(user_audio) / sample_rate

    print(f"\nAudio: {duration_secs:.2f}s, {sample_rate} Hz")

    # Resample to 16 kHz if needed (Silero VAD only supports 8/16 kHz;
    # Krisp VAD supports more rates but 16 kHz is a safe common default)
    if vad_type == "silero":
        user_audio, sample_rate = resample_audio(user_audio, sample_rate, 16000, verbose=True)
    elif sample_rate not in (8000, 16000, 32000, 44100, 48000):
        user_audio, sample_rate = resample_audio(user_audio, sample_rate, 16000, verbose=True)
    duration_secs = len(user_audio) / sample_rate
    num_samples = len(user_audio)

    # Initialize VAD (shared across all strategies)
    vad_params = VADParams(stop_secs=0.2)
    print(f"\nInitializing {vad_type} VAD...")
    vad, vad_type = create_vad(vad_type, vad_params, sample_rate, frame_duration_ms=10)
    print(
        f"  {vad_type} VAD ready (confidence={vad_params.confidence}, "
        f"stop_secs={vad_params.stop_secs})"
    )

    # Initialize the pipecat IP strategy if krisp-ip is requested
    ip_strategy: Optional[KrispVivaIPUserTurnStartStrategy] = None
    ip_init_ms = 0.0
    use_ip = "krisp-ip" in strategy_names

    if use_ip:
        if not model_path:
            print("Error: krisp-ip strategy requires KRISP_VIVA_IP_MODEL_PATH")
            sys.exit(1)
        print("\nInitializing Krisp IP strategy (pipecat integration)...")
        t0 = time.time()
        ip_strategy = KrispVivaIPUserTurnStartStrategy(
            model_path=model_path,
            threshold=threshold,
            frame_duration_ms=frame_duration_ms,
        )
        ip_init_ms = (time.time() - t0) * 1000
        print(f"  IP strategy ready in {ip_init_ms:.1f}ms (threshold={threshold})")

    # Prepare per-strategy results
    results: Dict[str, StrategyResult] = {}
    for name in strategy_names:
        init_ms = ip_init_ms if name == "krisp-ip" else 0.0
        results[name] = StrategyResult(name=name, init_time_ms=init_ms)

    frame_size = int(sample_rate * frame_duration_ms / 1000)

    print(f"\nProcessing audio...")
    print(f"  Frame size: {frame_size} samples ({frame_duration_ms}ms)")
    print(f"  Strategies: {', '.join(strategy_names)}")

    # Shared VAD state
    speech_segments: List[Tuple[float, float]] = []
    current_speech_start: Optional[float] = None
    prev_vad_state: VADState = VADState.QUIET
    vad_speaking = False

    # Per-strategy state: tracks whether a strategy already fired during the
    # current speech segment.  Reset on each VAD stop so every segment is
    # evaluated independently (regression-test mode).
    strategy_fired: Dict[str, bool] = {name: False for name in strategy_names}

    frames_processed = 0

    for i in range(0, num_samples, frame_size):
        chunk = user_audio[i : i + frame_size]
        if len(chunk) == 0:
            break
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))

        timestamp = frames_processed * frame_duration_ms / 1000.0
        frames_processed += 1
        frame_bytes = chunk.tobytes()

        # VAD (shared) — smoothed state with start/stop debouncing
        vad_state = vad._run_analyzer(frame_bytes)
        vad_just_started = prev_vad_state != VADState.SPEAKING and vad_state == VADState.SPEAKING
        vad_just_stopped = (
            prev_vad_state != VADState.QUIET and vad_state == VADState.QUIET and vad_speaking
        )

        # Raw per-frame VAD: True when the last raw confidence check passed,
        # before the state machine applies start/stop debouncing.
        # STARTING/SPEAKING → raw voice detected; QUIET/STOPPING → raw silence.
        raw_speaking = vad_state in (VADState.STARTING, VADState.SPEAKING)

        # Smoothed transitions — used for speech segment tracking and the
        # "vad" strategy which mirrors pipeline behaviour.
        if vad_just_started:
            vad_speaking = True
            current_speech_start = timestamp

            # VAD strategy: every speech start = interruption
            if "vad" in results and not strategy_fired.get("vad", False):
                results["vad"].interruption_times.append(timestamp)
                strategy_fired["vad"] = True
                if verbose:
                    print(f"  [vad] Interruption at {timestamp:.2f}s (VAD speech start)")

        if vad_just_stopped:
            vad_speaking = False
            if current_speech_start is not None:
                speech_segments.append((current_speech_start, timestamp))
                current_speech_start = None
            # Reset per-segment flags so the next speech segment is evaluated
            for k in strategy_fired:
                strategy_fired[k] = False
            if ip_strategy:
                ip_strategy._speech_active = False
                ip_strategy._decision_made = False

        # Feed audio to the IP strategy with two separate flags:
        #   _vad_flag      = raw per-frame VAD (for ip_session.process())
        #   _speech_active = smoothed detection window (for threshold guard)
        # The smoothed window stays open slightly after raw VAD goes silent,
        # which is needed because the IP model may output high probability
        # one frame *after* the raw VAD transitions to False.
        if ip_strategy:
            ip_strategy._vad_flag = raw_speaking
            ip_strategy._speech_active = vad_speaking
            audio_frame = InputAudioRawFrame(
                audio=frame_bytes,
                sample_rate=sample_rate,
                num_channels=1,
            )
            result = await ip_strategy.process_frame(audio_frame)
            if result == ProcessFrameResult.STOP and not strategy_fired.get("krisp-ip", False):
                results["krisp-ip"].interruption_times.append(timestamp)
                strategy_fired["krisp-ip"] = True

        prev_vad_state = vad_state

        if i % (frame_size * 100) == 0:
            progress = (i / num_samples) * 100
            print(f"  Progress: {progress:.1f}%", end="\r")

    print("  Progress: 100.0%")

    # Final speech segment
    if current_speech_start is not None:
        speech_segments.append((current_speech_start, duration_secs))

    # ---- Generate outputs ----
    os.makedirs(output_dir, exist_ok=True)
    input_stem = Path(input_path).stem
    beep = _generate_beep(sample_rate)

    annotated_audio_map: Dict[str, np.ndarray] = {}

    for name, result in results.items():
        annotated = _build_annotated_audio(
            user_audio,
            sample_rate,
            result.interruption_times,
            beep,
        )
        annotated_audio_map[name] = annotated

        wav_path = os.path.join(output_dir, f"{input_stem}_{name.replace('-', '_')}.wav")
        write_audio_file(wav_path, annotated, sample_rate, verbose=True)
        n_int = len(result.interruption_times)
        print(f"  {name}: {n_int} interruption(s) -> {wav_path}")

    # ---- Terminal output ----
    print(_format_ascii_timeline(duration_secs, speech_segments, results))
    print(
        _format_summary(
            input_path,
            sample_rate,
            duration_secs,
            threshold,
            frame_duration_ms,
            speech_segments,
            results,
            vad_type,
        )
    )

    # HTML report
    html_path = os.path.join(output_dir, f"{input_stem}_ip_report.html")
    _generate_html_report(
        input_path=input_path,
        duration_secs=duration_secs,
        sample_rate=sample_rate,
        threshold=threshold,
        speech_segments=speech_segments,
        results=results,
        output_path=html_path,
        annotated_audio=annotated_audio_map,
        vad_type=vad_type,
    )
    print(f"\n  HTML report: {html_path}")

    print(f"\n  Listen to the annotated WAV files -- beeps mark detected interruptions.")

    # Cleanup
    if ip_strategy:
        await ip_strategy.cleanup()
    if hasattr(vad, "cleanup"):
        await vad.cleanup()
    print("\nDone.")


# ---------------------------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------------------------

STRATEGY_COLORS_TERM = {
    "krisp-ip": ("█", "!"),
    "vad": ("▓", "X"),
}


def _format_ascii_timeline(
    duration_secs: float,
    speech_segments: List[Tuple[float, float]],
    results: Dict[str, StrategyResult],
    width: int = 80,
) -> str:
    """Render ASCII timeline comparing strategies."""
    label_width = 16
    bar_width = width - label_width - 1

    def time_to_col(t: float) -> int:
        return min(int(t / duration_secs * bar_width), bar_width - 1)

    lines: List[str] = []
    lines.append("")
    lines.append("Visual Timeline:")

    # Ruler
    ruler_label = " " * (label_width + bar_width + 2)
    ruler_ticks = " " * (label_width + bar_width + 2)
    for sec in range(0, int(duration_secs) + 1, 5):
        col = time_to_col(sec)
        tag = f"{sec}"
        pos = label_width + col
        if pos + len(tag) < len(ruler_label):
            ruler_label = ruler_label[:pos] + tag + ruler_label[pos + len(tag) :]
        if pos < len(ruler_ticks):
            ruler_ticks = ruler_ticks[:pos] + "|" + ruler_ticks[pos + 1 :]
    lines.append(f"  {'Time(s)':>{label_width - 2}}  {ruler_label[label_width:]}")
    lines.append(f"  {'':>{label_width - 2}}  {ruler_ticks[label_width:]}")

    # User speech
    speech_bar = list("." * bar_width)
    for start, end in speech_segments:
        for c in range(time_to_col(start), min(time_to_col(end) + 1, bar_width)):
            speech_bar[c] = "#"
    lines.append(f"  {'User Speech':>{label_width - 2}}  {''.join(speech_bar)}")

    # Per-strategy interruption rows
    for name, result in results.items():
        row = list("." * bar_width)
        for t_int in result.interruption_times:
            col = time_to_col(t_int)
            if 0 <= col < bar_width:
                row[col] = "!"
        lines.append(f"  {name:>{label_width - 2}}  {''.join(row)}")

    lines.append("")
    lines.append(
        f"  {'Legend:':>{label_width - 2}}  #=user speech  !=interruption detected  .=no detection"
    )
    lines.append("")

    return "\n".join(lines)


def _format_summary(
    input_path: str,
    sample_rate: int,
    duration_secs: float,
    threshold: float,
    frame_duration_ms: int,
    speech_segments: List[Tuple[float, float]],
    results: Dict[str, StrategyResult],
    vad_type: str = "silero",
) -> str:
    """Format text summary comparing strategies."""
    lines: List[str] = []
    sep = "=" * 60

    vad_label = "Krisp VIVA" if vad_type == "krisp" else "Silero"

    lines.append(sep)
    lines.append("Interrupt Prediction: Strategy Comparison")
    lines.append(sep)
    lines.append(f"Audio: {os.path.basename(input_path)} ({sample_rate} Hz, {duration_secs:.2f}s)")
    lines.append(f"IP threshold: {threshold}, Frame duration: {frame_duration_ms}ms")
    lines.append(f"VAD: {vad_label} (confidence=0.7, stop_secs=0.2)")
    lines.append(f"Speech segments: {len(speech_segments)}")
    lines.append("")

    for name, result in results.items():
        lines.append(f"--- {name} ---")
        n_int = len(result.interruption_times)
        lines.append(f"  Interruptions detected: {n_int}")
        if result.init_time_ms > 0:
            lines.append(f"  Init time: {result.init_time_ms:.1f}ms")

        if result.interruption_times:
            for idx, t in enumerate(result.interruption_times, 1):
                lines.append(f"  Interruption {idx}: {t:.2f}s")
        else:
            lines.append(f"  No interruptions detected")

        lines.append("")

    # Per-segment comparison
    if len(results) >= 2 and speech_segments:
        lines.append("Per-segment comparison:")
        for seg_idx, (seg_start, seg_end) in enumerate(speech_segments, 1):
            lines.append(f"  Segment {seg_idx} [{seg_start:.2f}s - {seg_end:.2f}s]:")
            for name, result in results.items():
                seg_ints = [t for t in result.interruption_times if seg_start <= t <= seg_end + 0.5]
                if seg_ints:
                    lines.append(f"    {name}: interrupted at {seg_ints[0]:.2f}s")
                else:
                    lines.append(f"    {name}: no interruption (backchannel)")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

STRATEGY_COLORS = {
    "krisp-ip": "#4ecca3",
    "vad": "#FF9800",
}


def _generate_html_report(
    input_path: str,
    duration_secs: float,
    sample_rate: int,
    threshold: float,
    speech_segments: List[Tuple[float, float]],
    results: Dict[str, StrategyResult],
    output_path: str,
    annotated_audio: Optional[Dict[str, np.ndarray]] = None,
    vad_type: str = "silero",
) -> None:
    """Generate a self-contained HTML report comparing strategies."""
    basename = os.path.basename(input_path)
    seg_json = ", ".join(f"[{s:.3f},{e:.3f}]" for s, e in speech_segments)

    # Build per-strategy JSON
    strategy_blocks: List[str] = []
    for name, result in results.items():
        color = STRATEGY_COLORS.get(name, "#2196F3")
        ints_json = ", ".join(f"{t:.3f}" for t in result.interruption_times)

        audio_uri = ""
        if annotated_audio and name in annotated_audio:
            audio_uri = _encode_wav_base64(annotated_audio[name], sample_rate)

        strategy_blocks.append(
            f'{{"name":"{name}","color":"{color}",'
            f'"init_ms":{result.init_time_ms:.1f},'
            f'"n_int":{len(result.interruption_times)},'
            f'"audio":"{audio_uri}",'
            f'"ints":[{ints_json}]}}'
        )
    strategies_json = ",\n      ".join(strategy_blocks)

    vad_label = "Krisp VIVA" if vad_type == "krisp" else "Silero"

    html = _HTML_TEMPLATE.format(
        basename=basename,
        duration_secs=duration_secs,
        sample_rate=sample_rate,
        threshold=threshold,
        vad_label=vad_label,
        n_speech=len(speech_segments),
        seg_json=seg_json,
        strategies_json=strategies_json,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Interrupt Prediction: {basename}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #1a1a2e; color: #eee; padding: 24px; }}
  h1 {{ font-size: 1.4em; margin-bottom: 4px; color: #fff; }}
  .meta {{ color: #888; font-size: 0.85em; margin-bottom: 20px; }}
  .container {{ background: #16213e; border-radius: 8px; padding: 16px 20px;
                margin-bottom: 20px; overflow-x: auto; }}
  .transport {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px;
                padding: 8px 12px; background: #0f3460; border-radius: 6px; }}
  .transport-btn {{ width: 32px; height: 32px; border-radius: 50%; border: none;
                    background: #4ecca3; color: #1a1a2e; font-size: 13px; cursor: pointer;
                    display: flex; align-items: center; justify-content: center;
                    flex-shrink: 0; }}
  .transport-btn:hover {{ filter: brightness(1.2); }}
  .transport-time {{ font-size: 0.85em; color: #ccc; min-width: 100px;
                     font-variant-numeric: tabular-nums; }}
  .transport-select {{ background: #16213e; color: #eee; border: 1px solid #333;
                       border-radius: 4px; padding: 4px 8px; font-size: 0.82em;
                       font-family: inherit; cursor: pointer; }}
  .transport-label {{ font-size: 0.78em; color: #888; }}
  .transport-spacer {{ flex: 1; }}
  .tl-wrap {{ position: relative; }}
  .timeline-row {{ display: flex; align-items: center; margin: 4px 0; height: 28px; }}
  .timeline-label {{ width: 140px; min-width: 140px; text-align: right; padding-right: 12px;
                     font-size: 0.85em; color: #aaa; white-space: nowrap; }}
  .timeline-track {{ position: relative; flex: 1; height: 100%; background: #0f3460;
                     border-radius: 4px; cursor: pointer; }}
  .speech-bar {{ position: absolute; height: 100%; background: #4ecca3; border-radius: 3px;
                 opacity: 0.8; pointer-events: none; min-width: 2px; }}
  .interrupt-marker {{ position: absolute; width: 3px; height: 100%;
                       border-radius: 2px; cursor: pointer; z-index: 2; }}
  .interrupt-marker:hover {{ filter: brightness(1.4); }}
  .interrupt-marker .tooltip {{ display: none; position: absolute; bottom: 110%; left: 50%;
    transform: translateX(-50%); background: #222; color: #fff; padding: 6px 10px;
    border-radius: 4px; font-size: 0.75em; white-space: nowrap; z-index: 10;
    pointer-events: none; box-shadow: 0 2px 8px rgba(0,0,0,.5); }}
  .interrupt-marker:hover .tooltip {{ display: block; }}
  .playhead {{ position: absolute; top: 0; bottom: 0; width: 2px; background: #fff;
               pointer-events: none; z-index: 5; opacity: 0; transition: opacity 0.15s;
               box-shadow: 0 0 6px rgba(255,255,255,.4); }}
  .playhead.visible {{ opacity: 1; }}
  .playhead-dot {{ position: absolute; top: -4px; left: -3px; width: 8px; height: 8px;
                   border-radius: 50%; background: #fff; }}
  .ruler {{ position: relative; height: 20px; cursor: pointer; }}
  .ruler-tick {{ position: absolute; top: 0; font-size: 0.7em; color: #666;
                 transform: translateX(-50%); }}
  .ruler-line {{ position: absolute; top: 14px; width: 1px; height: 6px; background: #444; }}
  .summary {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 16px; }}
  .card {{ background: #16213e; border-radius: 8px; padding: 16px 20px; flex: 1;
           min-width: 260px; }}
  .card h3 {{ font-size: 1em; margin-bottom: 10px; }}
  .card .stat {{ display: flex; justify-content: space-between; padding: 3px 0;
                 font-size: 0.85em; }}
  .card .stat .label {{ color: #888; }}
  .card .note {{ color: #ffb74d; font-size: 0.8em; margin-top: 8px; }}
  footer {{ margin-top: 24px; color: #555; font-size: 0.75em; text-align: center; }}
</style>
</head>
<body>
<h1>Interrupt Prediction: Strategy Comparison</h1>
<p class="meta">{basename} &mdash; {duration_secs:.2f}s, {sample_rate} Hz &mdash;
VAD: {vad_label} &mdash; IP threshold: {threshold} &mdash; {n_speech} speech segments</p>

<div class="container">
  <div class="transport" id="transport" style="display:none">
    <button class="transport-btn" id="play-btn" title="Play / Pause">&#9654;</button>
    <span class="transport-time" id="play-time">0:00.0 / 0:00.0</span>
    <div class="transport-spacer"></div>
    <span class="transport-label">Audio track</span>
    <select class="transport-select" id="track-select"></select>
  </div>
  <div class="tl-wrap" id="tl-wrap">
    <div id="tl"></div>
    <div class="playhead" id="playhead"><div class="playhead-dot"></div></div>
  </div>
</div>
<div class="summary" id="cards"></div>
<footer>Generated by demo_interrupt_prediction.py</footer>

<script>
const DUR = {duration_secs};
const THRESHOLD = {threshold};
const SPEECH = [{seg_json}];
const STRATEGIES = [
      {strategies_json}
];

const tl = document.getElementById('tl');
const cards = document.getElementById('cards');
const tlWrap = document.getElementById('tl-wrap');
const playhead = document.getElementById('playhead');
const playBtn = document.getElementById('play-btn');
const playTime = document.getElementById('play-time');
const trackSelect = document.getElementById('track-select');
const transport = document.getElementById('transport');

function pct(t) {{ return (t / DUR * 100).toFixed(4) + '%'; }}
function fmt(t) {{
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return m + ':' + (s < 10 ? '0' : '') + s.toFixed(1);
}}

// ---- Audio player ----
const audioEls = {{}};
let activeAudio = null;
const hasAudio = STRATEGIES.some(s => s.audio && s.audio.length > 0);

if (hasAudio) {{
  transport.style.display = 'flex';
  STRATEGIES.forEach(s => {{
    if (!s.audio || s.audio.length === 0) return;
    const el = new Audio(s.audio);
    el.preload = 'auto';
    audioEls[s.name] = el;
    const opt = document.createElement('option');
    opt.value = s.name;
    opt.textContent = s.name;
    opt.style.color = s.color;
    trackSelect.appendChild(opt);
  }});
  const firstName = Object.keys(audioEls)[0];
  if (firstName) {{ activeAudio = audioEls[firstName]; trackSelect.value = firstName; }}
  trackSelect.addEventListener('change', () => {{
    const wasPlaying = activeAudio && !activeAudio.paused;
    const curTime = activeAudio ? activeAudio.currentTime : 0;
    if (activeAudio) activeAudio.pause();
    activeAudio = audioEls[trackSelect.value] || null;
    if (activeAudio) {{
      activeAudio.currentTime = Math.min(curTime, activeAudio.duration || DUR);
      if (wasPlaying) activeAudio.play();
    }}
  }});
}}

function togglePlay() {{
  if (!activeAudio) return;
  if (activeAudio.paused) {{
    activeAudio.play();
    playBtn.innerHTML = '&#9646;&#9646;';
    playhead.classList.add('visible');
  }} else {{
    activeAudio.pause();
    playBtn.innerHTML = '&#9654;';
  }}
}}
playBtn.addEventListener('click', togglePlay);
document.addEventListener('keydown', e => {{
  if (e.code === 'Space' && hasAudio && e.target === document.body) {{
    e.preventDefault(); togglePlay();
  }}
}});

function tickPlayhead() {{
  if (activeAudio && !activeAudio.paused) {{
    const t = activeAudio.currentTime;
    const pctVal = Math.min(t / DUR * 100, 100);
    const firstTrack = tl.querySelector('.timeline-track');
    if (firstTrack) {{
      const tlRect = tlWrap.getBoundingClientRect();
      const trackRect = firstTrack.getBoundingClientRect();
      playhead.style.left = ((trackRect.left - tlRect.left) + (trackRect.width * pctVal / 100)) + 'px';
    }}
    playTime.textContent = fmt(t) + ' / ' + fmt(DUR);
  }}
  requestAnimationFrame(tickPlayhead);
}}
requestAnimationFrame(tickPlayhead);

if (hasAudio) {{
  Object.values(audioEls).forEach(el => {{
    el.addEventListener('ended', () => {{
      playBtn.innerHTML = '&#9654;';
      playhead.classList.remove('visible');
    }});
  }});
}}

function seekFromClick(e) {{
  if (!activeAudio) return;
  const track = e.target.closest('.timeline-track');
  if (!track) return;
  const rect = track.getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  activeAudio.currentTime = frac * DUR;
  playhead.classList.add('visible');
  const tlRect = tlWrap.getBoundingClientRect();
  playhead.style.left = ((rect.left - tlRect.left) + (rect.width * frac)) + 'px';
  playTime.textContent = fmt(activeAudio.currentTime) + ' / ' + fmt(DUR);
}}

// ---- Build timeline ----

// Ruler
let ruler = '<div class="timeline-row"><div class="timeline-label"></div>' +
  '<div class="timeline-track ruler">';
for (let s = 0; s <= DUR; s += 5)
  ruler += '<span class="ruler-tick" style="left:' + pct(s) + '">' + s + 's</span>' +
    '<span class="ruler-line" style="left:' + pct(s) + '"></span>';
ruler += '</div></div>';
tl.innerHTML = ruler;

// User speech
let sr = '<div class="timeline-row"><div class="timeline-label">User Speech</div>' +
  '<div class="timeline-track">';
SPEECH.forEach(s => {{
  sr += '<div class="speech-bar" style="left:' + pct(s[0]) + ';width:' + pct(s[1]-s[0]) + '"></div>';
}});
sr += '</div></div>';
tl.innerHTML += sr;

// Per-strategy interruption rows
STRATEGIES.forEach(s => {{
  let row = '<div class="timeline-row"><div class="timeline-label" style="color:' +
    s.color + '">' + s.name + '</div><div class="timeline-track">';
  s.ints.forEach((t, i) => {{
    row += '<div class="interrupt-marker" style="left:' + pct(t) + ';background:' + s.color + '">' +
      '<div class="tooltip">' + s.name + ': interruption at ' + t.toFixed(2) + 's</div></div>';
  }});
  row += '</div></div>';
  tl.innerHTML += row;
}});

tl.addEventListener('click', seekFromClick);

// ---- Summary cards ----
STRATEGIES.forEach(s => {{
  let note = '';
  if (s.name === 'vad' && s.n_int > 0)
    note = '<div class="note">Fires on first speech start (may be a backchannel)</div>';
  else if (s.n_int === 0)
    note = '<div class="note">No interruptions -- all speech classified as backchannel</div>';

  let intsStr = s.n_int > 0 ? s.ints.map(t => t.toFixed(2) + 's').join(', ') : 'N/A';
  cards.innerHTML += '<div class="card"><h3 style="color:' + s.color + '">' + s.name + '</h3>'
    + '<div class="stat"><span class="label">Interruptions detected</span><span>'
    + s.n_int + '</span></div>'
    + '<div class="stat"><span class="label">Interrupt at</span><span>' + intsStr + '</span></div>'
    + (s.init_ms > 0 ? '<div class="stat"><span class="label">Init time</span><span>' +
        s.init_ms.toFixed(0) + 'ms</span></div>' : '')
    + note + '</div>';
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Interrupt prediction demo: compare interruption strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_interrupt_prediction.py input.wav
  python demo_interrupt_prediction.py input.wav --strategy krisp-ip
  python demo_interrupt_prediction.py input.wav --strategy krisp-ip --strategy vad
  python demo_interrupt_prediction.py input.wav --vad krisp
  python demo_interrupt_prediction.py input.wav --threshold 0.6 -v

Strategies:
  krisp-ip    Krisp IP model (genuine interruptions only)
  vad         VAD-only (any speech = interruption)

Environment variables:
  KRISP_VIVA_IP_MODEL_PATH   Path to Krisp IP model (.kef)
  KRISP_VIVA_VAD_MODEL_PATH  Path to Krisp VAD model (.kef, for --vad krisp)
        """,
    )

    parser.add_argument("input", help="Input audio file with user speech/interruptions")
    parser.add_argument(
        "--strategy",
        action="append",
        choices=AVAILABLE_STRATEGIES,
        dest="strategies",
        help=f"Strategy to run (repeatable, default: all). Choices: {AVAILABLE_STRATEGIES}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="IP probability threshold (0.0 to 1.0, default: 0.5)",
    )
    parser.add_argument(
        "--frame-duration",
        type=int,
        default=20,
        choices=[10, 15, 20, 30, 32],
        help="Frame duration in ms (default: 20)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./demo_output",
        help="Output directory (default: ./demo_output)",
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
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-frame details",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if not 0.0 <= args.threshold <= 1.0:
        print(f"Error: Threshold must be between 0.0 and 1.0, got {args.threshold}")
        sys.exit(1)

    strategy_names = args.strategies or AVAILABLE_STRATEGIES

    model_path = os.getenv("KRISP_VIVA_IP_MODEL_PATH")
    if "krisp-ip" in strategy_names:
        if not model_path:
            print("Error: KRISP_VIVA_IP_MODEL_PATH environment variable not set")
            sys.exit(1)
        if not os.path.isfile(model_path):
            print(f"Error: IP model file not found: {model_path}")
            sys.exit(1)

    asyncio.run(
        process_audio(
            input_path=args.input,
            strategy_names=strategy_names,
            model_path=model_path,
            threshold=args.threshold,
            frame_duration_ms=args.frame_duration,
            output_dir=args.output_dir,
            verbose=args.verbose,
            vad_type=args.vad_type,
        )
    )


if __name__ == "__main__":
    main()

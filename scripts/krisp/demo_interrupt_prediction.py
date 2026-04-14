#!/usr/bin/env python3
"""Interrupt prediction demonstration tool.

Demonstrates how Krisp's Interruption Prediction (IP) model distinguishes
genuine user interruptions from backchannels during assistant speech.

The tool takes a user audio recording and simulates two scenarios:

  IP ENABLED:  When the IP model detects a genuine interruption, the
               simulated assistant audio stops immediately -- demonstrating
               responsive, natural interaction.

  IP DISABLED: The assistant audio plays through regardless of user speech --
               showing what happens without interrupt handling.

Outputs:
  - Two annotated WAV files (IP enabled vs disabled) for A/B listening
  - ASCII timeline in the terminal
  - Self-contained HTML report with interactive timeline
  - Per-event analysis with IP probabilities

The simulated assistant audio is either a provided WAV file (--bot-audio)
or a generated low hum tone, mixed with the user recording so listeners
can hear when the bot stops vs continues.

Usage:
    python demo_interrupt_prediction.py user_speech.wav
    python demo_interrupt_prediction.py user_speech.wav --bot-audio assistant.wav
    python demo_interrupt_prediction.py user_speech.wav --threshold 0.6 -v

Requirements:
    pip install soundfile numpy pipecat-ai[krisp]
    Set KRISP_VIVA_IP_MODEL_PATH environment variable
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

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
    from audio_file_utils import read_audio_file, write_audio_file
except ImportError as e:
    print(f"Error: Missing required dependencies: {e}")
    print("Install with: pip install soundfile numpy")
    sys.exit(1)

project_root = script_dir.parent.parent
src_dir = project_root / "src"
if src_dir.exists() and str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class IPEvent:
    """A single IP model evaluation at a point in time."""

    timestamp: float
    probability: float
    is_interruption: bool
    vad_speaking: bool


@dataclass
class IPResult:
    """Collected results for one processing run."""

    events: List[IPEvent] = field(default_factory=list)
    speech_segments: List[Tuple[float, float]] = field(default_factory=list)
    interruption_times: List[float] = field(default_factory=list)
    init_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# IP session helpers
# ---------------------------------------------------------------------------

def create_ip_session(
    model_path: str,
    sample_rate: int,
    frame_duration_ms: int,
):
    """Create a Krisp IP session.

    Args:
        model_path: Path to the IP model .kef file.
        sample_rate: Audio sample rate in Hz.
        frame_duration_ms: Frame duration in ms.

    Returns:
        Tuple of (ip_session, samples_per_frame).
    """
    import krisp_audio

    from pipecat.audio.krisp_instance import (
        KrispVivaSDKManager,
        int_to_krisp_frame_duration,
        int_to_krisp_sample_rate,
    )

    KrispVivaSDKManager.acquire()

    model_info = krisp_audio.ModelInfo()
    model_info.path = model_path

    ip_cfg = krisp_audio.IpSessionConfig()
    ip_cfg.inputSampleRate = int_to_krisp_sample_rate(sample_rate)
    ip_cfg.inputFrameDuration = int_to_krisp_frame_duration(frame_duration_ms)
    ip_cfg.modelInfo = model_info

    session = krisp_audio.IpFloat.create(ip_cfg)
    samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
    return session, samples_per_frame


# ---------------------------------------------------------------------------
# Audio generation helpers
# ---------------------------------------------------------------------------

def generate_bot_tone(
    num_samples: int,
    sample_rate: int,
    freq: float = 180.0,
    amplitude: float = 0.15,
) -> np.ndarray:
    """Generate a low-frequency hum simulating assistant speech.

    Uses a composite of harmonics for a more natural "speaking" feel.
    """
    t = np.arange(num_samples) / sample_rate
    tone = np.zeros(num_samples, dtype=np.float64)
    for harmonic, amp_scale in [(1, 1.0), (2, 0.5), (3, 0.25)]:
        tone += amp_scale * np.sin(2 * np.pi * freq * harmonic * t)
    # Slight amplitude modulation for naturalness
    mod = 1.0 + 0.3 * np.sin(2 * np.pi * 3.0 * t)
    tone *= mod
    # Normalize and scale
    peak = np.max(np.abs(tone))
    if peak > 0:
        tone = tone / peak * amplitude
    return (tone * 32767).astype(np.int16)


def generate_beep(
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


def mix_audio(base: np.ndarray, overlay: np.ndarray, start: int = 0) -> np.ndarray:
    """Mix overlay into base at a given sample offset, clipping to int16."""
    end = min(start + len(overlay), len(base))
    seg_len = end - start
    if seg_len <= 0:
        return base
    mixed = base[start:end].astype(np.int32) + overlay[:seg_len].astype(np.int32)
    base[start:end] = np.clip(mixed, -32768, 32767).astype(np.int16)
    return base


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_audio(
    input_path: str,
    model_path: str,
    threshold: float = 0.5,
    frame_duration_ms: int = 20,
    bot_audio_path: Optional[str] = None,
    output_dir: str = "./demo_output",
    verbose: bool = False,
) -> None:
    """Process user audio through the IP model and produce comparison outputs.

    Args:
        input_path: Path to user audio file (contains speech/interruptions).
        model_path: Path to Krisp IP model (.kef).
        threshold: IP probability threshold for genuine interruption.
        bot_audio_path: Optional path to assistant audio WAV. If None, a
            synthetic tone is generated.
        output_dir: Output directory for generated files.
        verbose: Print per-frame IP probabilities.
    """
    user_audio, sample_rate = read_audio_file(input_path, verbose=True)
    duration_secs = len(user_audio) / sample_rate
    num_samples = len(user_audio)

    print(f"\nAudio: {duration_secs:.2f}s, {sample_rate} Hz")

    # Bot audio: load or generate
    if bot_audio_path:
        print(f"\nLoading bot audio from: {bot_audio_path}")
        bot_audio, bot_sr = read_audio_file(bot_audio_path, verbose=True)
        if bot_sr != sample_rate:
            print(f"  Warning: bot audio sample rate ({bot_sr}) differs from user ({sample_rate})")
            print(f"  Results may be inaccurate. Please provide matching sample rates.")
        if len(bot_audio) < num_samples:
            bot_audio = np.pad(bot_audio, (0, num_samples - len(bot_audio)))
        else:
            bot_audio = bot_audio[:num_samples]
    else:
        print("\nGenerating synthetic bot audio (low hum)...")
        bot_audio = generate_bot_tone(num_samples, sample_rate)

    # Initialize Silero VAD
    print("\nInitializing Silero VAD...")
    vad = SileroVADAnalyzer(params=VADParams(stop_secs=0.2))
    vad.set_sample_rate(sample_rate)
    print(f"  Silero VAD ready (stop_secs=0.2, confidence={vad.params.confidence})")

    # Initialize IP session
    print(f"\nInitializing Krisp IP model...")
    t0 = time.time()
    ip_session, samples_per_frame = create_ip_session(
        model_path, sample_rate, frame_duration_ms
    )
    init_ms = (time.time() - t0) * 1000
    print(f"  IP model ready in {init_ms:.1f}ms (threshold={threshold})")

    result = IPResult(init_time_ms=init_ms)
    frame_size = samples_per_frame

    print(f"\nProcessing audio...")
    print(f"  Frame size: {frame_size} samples ({frame_duration_ms}ms)")

    # State
    speech_segments: List[Tuple[float, float]] = []
    current_speech_start: Optional[float] = None
    prev_vad_state: VADState = VADState.QUIET
    vad_speaking = False
    decision_made = False
    audio_buffer = bytearray()
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

        # VAD
        vad_state = vad._run_analyzer(frame_bytes)
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
            decision_made = False
            audio_buffer.clear()

        if vad_just_stopped:
            vad_speaking = False
            if current_speech_start is not None:
                speech_segments.append((current_speech_start, timestamp))
                current_speech_start = None
            decision_made = False
            audio_buffer.clear()

        # Feed IP model during speech
        if vad_speaking and not decision_made:
            audio_buffer.extend(frame_bytes)

            total_ip_samples = len(audio_buffer) // 2
            num_complete = total_ip_samples // samples_per_frame

            if num_complete > 0:
                bytes_to_process = num_complete * samples_per_frame * 2
                audio_to_process = bytes(audio_buffer[:bytes_to_process])
                audio_buffer = audio_buffer[bytes_to_process:]

                audio_int16 = np.frombuffer(audio_to_process, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                ip_frames = audio_float32.reshape(-1, samples_per_frame)

                for ip_frame in ip_frames:
                    ip_prob = ip_session.process(ip_frame.tolist(), True)
                    is_interruption = ip_prob >= threshold

                    event = IPEvent(
                        timestamp=timestamp,
                        probability=ip_prob,
                        is_interruption=is_interruption,
                        vad_speaking=True,
                    )
                    result.events.append(event)

                    if verbose:
                        marker = " *** INTERRUPTION ***" if is_interruption else ""
                        print(
                            f"  [{timestamp:.2f}s] IP prob={ip_prob:.3f}"
                            f" (threshold={threshold}){marker}"
                        )

                    if is_interruption:
                        result.interruption_times.append(timestamp)
                        decision_made = True
                        break

        prev_vad_state = vad_state

        if i % (frame_size * 100) == 0:
            progress = (i / num_samples) * 100
            print(f"  Progress: {progress:.1f}%", end="\r")

    print("  Progress: 100.0%")

    # Final speech segment
    if current_speech_start is not None:
        speech_segments.append((current_speech_start, duration_secs))

    result.speech_segments = speech_segments

    # ---- Generate outputs ----
    os.makedirs(output_dir, exist_ok=True)
    input_stem = Path(input_path).stem
    beep = generate_beep(sample_rate)

    # Build IP-ENABLED output: bot audio silenced after first interruption
    ip_enabled_audio = bot_audio.copy()

    if result.interruption_times:
        first_interrupt = result.interruption_times[0]
        cut_sample = int(first_interrupt * sample_rate)
        if cut_sample < num_samples:
            # Fade out bot audio over 50ms at interrupt point
            fade_len = min(int(sample_rate * 0.05), num_samples - cut_sample)
            if fade_len > 0:
                fade_out = np.linspace(1, 0, fade_len)
                ip_enabled_audio[cut_sample : cut_sample + fade_len] = (
                    ip_enabled_audio[cut_sample : cut_sample + fade_len]
                    .astype(np.float64) * fade_out
                ).astype(np.int16)
            ip_enabled_audio[cut_sample + fade_len :] = 0

    # Mix user audio into both versions
    ip_enabled_mixed = ip_enabled_audio.copy()
    ip_enabled_mixed = mix_audio(ip_enabled_mixed, user_audio)
    # Add beep markers at interruption points
    for t_int in result.interruption_times:
        sample_pos = int(t_int * sample_rate)
        ip_enabled_mixed = mix_audio(ip_enabled_mixed, beep, sample_pos)

    ip_disabled_mixed = bot_audio.copy()
    ip_disabled_mixed = mix_audio(ip_disabled_mixed, user_audio)

    # Save outputs
    enabled_path = os.path.join(output_dir, f"{input_stem}_ip_enabled.wav")
    disabled_path = os.path.join(output_dir, f"{input_stem}_ip_disabled.wav")
    write_audio_file(enabled_path, ip_enabled_mixed, sample_rate, verbose=True)
    write_audio_file(disabled_path, ip_disabled_mixed, sample_rate, verbose=True)

    # ---- Terminal output ----
    print(_format_ascii_timeline(duration_secs, speech_segments, result))
    print(_format_summary(
        input_path, sample_rate, duration_secs, threshold,
        frame_duration_ms, result,
    ))

    # HTML report
    html_path = os.path.join(output_dir, f"{input_stem}_ip_report.html")
    _generate_ip_html_report(
        input_path=input_path,
        duration_secs=duration_secs,
        sample_rate=sample_rate,
        threshold=threshold,
        speech_segments=speech_segments,
        result=result,
        output_path=html_path,
    )
    print(f"  HTML report: {html_path}")

    print(f"\nOutput files:")
    print(f"  IP enabled (bot stops):     {enabled_path}")
    print(f"  IP disabled (bot continues): {disabled_path}")
    print(f"\n  Listen to both files to hear the difference!")

    # Cleanup
    from pipecat.audio.krisp_instance import KrispVivaSDKManager

    KrispVivaSDKManager.release()
    print("\nDone.")


# ---------------------------------------------------------------------------
# Terminal formatting
# ---------------------------------------------------------------------------

def _format_ascii_timeline(
    duration_secs: float,
    speech_segments: List[Tuple[float, float]],
    result: IPResult,
    width: int = 80,
) -> str:
    """Render ASCII timeline showing speech, IP probabilities, and interruptions."""
    label_width = 16
    bar_width = width - label_width - 1

    def time_to_col(t: float) -> int:
        return min(int(t / duration_secs * bar_width), bar_width - 1)

    lines: List[str] = []
    lines.append("")
    lines.append("Visual Timeline:")

    # Time ruler
    ruler_label = " " * (label_width + bar_width + 2)
    ruler_ticks = " " * (label_width + bar_width + 2)
    for sec in range(0, int(duration_secs) + 1, 5):
        col = time_to_col(sec)
        tag = f"{sec}"
        pos = label_width + col
        if pos + len(tag) < len(ruler_label):
            ruler_label = ruler_label[:pos] + tag + ruler_label[pos + len(tag):]
        if pos < len(ruler_ticks):
            ruler_ticks = ruler_ticks[:pos] + "|" + ruler_ticks[pos + 1:]
    lines.append(f"  {'Time(s)':>{label_width - 2}}  {ruler_label[label_width:]}")
    lines.append(f"  {'':>{label_width - 2}}  {ruler_ticks[label_width:]}")

    # Speech row
    speech_bar = list("·" * bar_width)
    for start, end in speech_segments:
        c0 = time_to_col(start)
        c1 = time_to_col(end)
        for c in range(c0, min(c1 + 1, bar_width)):
            speech_bar[c] = "█"
    lines.append(f"  {'User Speech':>{label_width - 2}}  {''.join(speech_bar)}")

    # Bot audio row (full bar, cut at first interruption)
    bot_bar = list("░" * bar_width)
    if result.interruption_times:
        cut_col = time_to_col(result.interruption_times[0])
        for c in range(cut_col, bar_width):
            bot_bar[c] = "·"
    lines.append(f"  {'Bot (enabled)':>{label_width - 2}}  {''.join(bot_bar)}")

    bot_bar_full = list("░" * bar_width)
    lines.append(f"  {'Bot (disabled)':>{label_width - 2}}  {''.join(bot_bar_full)}")

    # IP events row
    ip_bar = list("·" * bar_width)
    for event in result.events:
        col = time_to_col(event.timestamp)
        if 0 <= col < bar_width:
            if event.is_interruption:
                ip_bar[col] = "!"
            elif event.probability > 0.3:
                ip_bar[col] = "~"
    for t_int in result.interruption_times:
        col = time_to_col(t_int)
        if 0 <= col < bar_width:
            ip_bar[col] = "!"
    lines.append(f"  {'IP Detection':>{label_width - 2}}  {''.join(ip_bar)}")

    lines.append("")
    lines.append(
        f"  {'Legend:':>{label_width - 2}}  "
        "█=user speech  ░=bot playing  !=interruption  ~=elevated prob  ·=silence"
    )
    lines.append("")

    return "\n".join(lines)


def _format_summary(
    input_path: str,
    sample_rate: int,
    duration_secs: float,
    threshold: float,
    frame_duration_ms: int,
    result: IPResult,
) -> str:
    """Format a text summary of the IP analysis."""
    lines: List[str] = []
    sep = "=" * 60

    lines.append(sep)
    lines.append("Interrupt Prediction Analysis")
    lines.append(sep)
    lines.append(
        f"Audio: {os.path.basename(input_path)} ({sample_rate} Hz, {duration_secs:.2f}s)"
    )
    lines.append(f"IP threshold: {threshold}, Frame duration: {frame_duration_ms}ms")
    lines.append(f"VAD: Silero (stop_secs=0.2)")
    lines.append(f"IP model init time: {result.init_time_ms:.1f}ms")
    lines.append("")

    lines.append("Speech Segments:")
    if result.speech_segments:
        for start, end in result.speech_segments:
            dur = end - start
            lines.append(f"  [{start:7.2f}s - {end:7.2f}s ] speech ({dur:.2f}s)")
    else:
        lines.append("  (none detected)")
    lines.append("")

    lines.append("Interruption Events:")
    if result.interruption_times:
        for i, t_int in enumerate(result.interruption_times, 1):
            matching = [e for e in result.events if e.is_interruption and abs(e.timestamp - t_int) < 0.05]
            prob_str = f" (prob={matching[0].probability:.3f})" if matching else ""
            lines.append(f"  Interruption {i}: {t_int:7.2f}s{prob_str}")
    else:
        lines.append("  (none detected -- all speech classified as backchannel)")
    lines.append("")

    all_probs = [e.probability for e in result.events]
    if all_probs:
        lines.append("IP Probability Stats:")
        lines.append(f"  Evaluations:  {len(all_probs)}")
        lines.append(f"  Mean prob:    {sum(all_probs) / len(all_probs):.3f}")
        lines.append(f"  Max prob:     {max(all_probs):.3f}")
        lines.append(f"  Min prob:     {min(all_probs):.3f}")
        lines.append("")

    lines.append("Behavioral Difference:")
    if result.interruption_times:
        first = result.interruption_times[0]
        remaining = duration_secs - first
        lines.append(
            f"  IP ENABLED:  Bot stops at {first:.2f}s"
            f" ({remaining:.2f}s of bot audio silenced)"
        )
        lines.append(f"  IP DISABLED: Bot plays full {duration_secs:.2f}s uninterrupted")
        lines.append(f"  Difference:  {remaining:.2f}s of overlap removed")
    else:
        lines.append(f"  IP ENABLED:  Bot plays full {duration_secs:.2f}s (no interruptions)")
        lines.append(f"  IP DISABLED: Same -- no difference for this audio")
    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _generate_ip_html_report(
    input_path: str,
    duration_secs: float,
    sample_rate: int,
    threshold: float,
    speech_segments: List[Tuple[float, float]],
    result: IPResult,
    output_path: str,
) -> None:
    """Generate a self-contained HTML report for the IP demo."""
    basename = os.path.basename(input_path)

    seg_json = ", ".join(f"[{s:.3f},{e:.3f}]" for s, e in speech_segments)
    events_json = ", ".join(
        f'{{"t":{e.timestamp:.3f},"p":{e.probability:.4f},"int":{str(e.is_interruption).lower()}}}'
        for e in result.events
    )
    interrupts_json = ", ".join(f"{t:.3f}" for t in result.interruption_times)

    all_probs = [e.probability for e in result.events]
    mean_prob = sum(all_probs) / len(all_probs) if all_probs else 0
    max_prob = max(all_probs) if all_probs else 0

    html = _IP_HTML_TEMPLATE.format(
        basename=basename,
        duration_secs=duration_secs,
        sample_rate=sample_rate,
        threshold=threshold,
        init_ms=result.init_time_ms,
        n_interruptions=len(result.interruption_times),
        n_evaluations=len(result.events),
        mean_prob=mean_prob,
        max_prob=max_prob,
        seg_json=seg_json,
        events_json=events_json,
        interrupts_json=interrupts_json,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


_IP_HTML_TEMPLATE = """<!DOCTYPE html>
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
  .timeline-row {{ display: flex; align-items: center; margin: 4px 0; height: 28px; }}
  .timeline-label {{ width: 140px; min-width: 140px; text-align: right; padding-right: 12px;
                     font-size: 0.85em; color: #aaa; white-space: nowrap; }}
  .timeline-track {{ position: relative; flex: 1; height: 100%; background: #0f3460;
                     border-radius: 4px; }}
  .speech-bar {{ position: absolute; height: 100%; background: #4ecca3; border-radius: 3px;
                 opacity: 0.8; }}
  .bot-bar {{ position: absolute; height: 100%; background: #e65100; border-radius: 3px;
              opacity: 0.6; }}
  .bot-bar-disabled {{ background: #e65100; }}
  .bot-bar-silenced {{ background: #333; opacity: 0.3; }}
  .interrupt-marker {{ position: absolute; width: 3px; height: 100%;
                       background: #ff1744; border-radius: 2px; cursor: pointer; z-index: 2; }}
  .interrupt-marker:hover {{ filter: brightness(1.4); }}
  .interrupt-marker .tooltip {{ display: none; position: absolute; bottom: 110%; left: 50%;
    transform: translateX(-50%); background: #222; color: #fff; padding: 6px 10px;
    border-radius: 4px; font-size: 0.75em; white-space: nowrap; z-index: 10;
    pointer-events: none; box-shadow: 0 2px 8px rgba(0,0,0,.5); }}
  .interrupt-marker:hover .tooltip {{ display: block; }}
  .prob-dot {{ position: absolute; width: 4px; height: 4px; border-radius: 50%;
               bottom: 0; transform: translateX(-50%); }}
  .ruler {{ position: relative; height: 20px; }}
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
  .enabled {{ color: #4ecca3; }}
  .disabled {{ color: #ff6b6b; }}
  .prob-track {{ position: relative; flex: 1; height: 40px; background: #0f3460;
                 border-radius: 4px; }}
  .threshold-line {{ position: absolute; width: 100%; border-top: 1px dashed #ff9800;
                     z-index: 1; }}
  footer {{ margin-top: 24px; color: #555; font-size: 0.75em; text-align: center; }}
</style>
</head>
<body>
<h1>Interrupt Prediction Analysis</h1>
<p class="meta">{basename} &mdash; {duration_secs:.2f}s, {sample_rate} Hz &mdash;
IP threshold: {threshold}</p>

<div class="container" id="tl"></div>
<div class="summary" id="cards"></div>
<div class="container" style="margin-top:20px">
  <h3 style="margin-bottom:12px;font-size:1em;">IP Probability Over Time</h3>
  <div id="prob-chart" style="position:relative;height:60px;background:#0f3460;border-radius:4px">
  </div>
  <div style="display:flex;justify-content:space-between;font-size:0.7em;color:#666;margin-top:4px">
    <span>0.0</span><span>threshold={threshold}</span><span>1.0</span>
  </div>
</div>
<footer>Generated by demo_interrupt_prediction.py</footer>

<script>
const DUR = {duration_secs};
const THRESHOLD = {threshold};
const SPEECH = [{seg_json}];
const EVENTS = [{events_json}];
const INTERRUPTS = [{interrupts_json}];

const tl = document.getElementById('tl');
const cards = document.getElementById('cards');

function pct(t) {{ return (t / DUR * 100).toFixed(4) + '%'; }}

// Ruler
let ruler = '<div class="timeline-row"><div class="timeline-label"></div>' +
  '<div class="timeline-track ruler">';
for (let s = 0; s <= DUR; s += 5) {{
  ruler += '<span class="ruler-tick" style="left:' + pct(s) + '">' + s + 's</span>';
  ruler += '<span class="ruler-line" style="left:' + pct(s) + '"></span>';
}}
ruler += '</div></div>';
tl.innerHTML = ruler;

// User speech row
let sr = '<div class="timeline-row"><div class="timeline-label">User Speech</div>' +
  '<div class="timeline-track">';
SPEECH.forEach(s => {{
  sr += '<div class="speech-bar" style="left:' + pct(s[0]) + ';width:' + pct(s[1]-s[0]) + '"></div>';
}});
sr += '</div></div>';
tl.innerHTML += sr;

// Bot (IP enabled) row
let br = '<div class="timeline-row"><div class="timeline-label" style="color:#4ecca3">' +
  'Bot (IP enabled)</div><div class="timeline-track">';
if (INTERRUPTS.length > 0) {{
  const cutT = INTERRUPTS[0];
  br += '<div class="bot-bar" style="left:0;width:' + pct(cutT) + '"></div>';
  br += '<div class="bot-bar bot-bar-silenced" style="left:' + pct(cutT) + ';width:' +
    pct(DUR - cutT) + '"></div>';
  INTERRUPTS.forEach(t => {{
    br += '<div class="interrupt-marker" style="left:' + pct(t) + '">' +
      '<div class="tooltip">Interruption at ' + t.toFixed(2) + 's</div></div>';
  }});
}} else {{
  br += '<div class="bot-bar" style="left:0;width:100%"></div>';
}}
br += '</div></div>';
tl.innerHTML += br;

// Bot (IP disabled) row
let dr = '<div class="timeline-row"><div class="timeline-label" style="color:#ff6b6b">' +
  'Bot (IP disabled)</div><div class="timeline-track">';
dr += '<div class="bot-bar bot-bar-disabled" style="left:0;width:100%"></div>';
INTERRUPTS.forEach(t => {{
  dr += '<div class="interrupt-marker" style="left:' + pct(t) + ';background:#ff9800;opacity:0.5">' +
    '<div class="tooltip">User interrupts at ' + t.toFixed(2) + 's (ignored)</div></div>';
}});
dr += '</div></div>';
tl.innerHTML += dr;

// Summary cards
const first = INTERRUPTS.length > 0 ? INTERRUPTS[0] : null;
const remaining = first !== null ? (DUR - first).toFixed(2) : '0.00';

cards.innerHTML = '<div class="card"><h3 class="enabled">IP Enabled</h3>'
  + '<div class="stat"><span class="label">Interruptions detected</span><span>'
  + {n_interruptions} + '</span></div>'
  + '<div class="stat"><span class="label">First interrupt at</span><span>'
  + (first !== null ? first.toFixed(2) + 's' : 'N/A') + '</span></div>'
  + '<div class="stat"><span class="label">Bot audio silenced</span><span>'
  + remaining + 's</span></div>'
  + '<div class="stat"><span class="label">Init time</span><span>'
  + {init_ms:.0f} + 'ms</span></div>'
  + '</div>'
  + '<div class="card"><h3 class="disabled">IP Disabled</h3>'
  + '<div class="stat"><span class="label">Interruptions detected</span><span>0</span></div>'
  + '<div class="stat"><span class="label">Bot audio silenced</span><span>0.00s</span></div>'
  + '<div class="stat"><span class="label">User-bot overlap</span><span>'
  + remaining + 's</span></div>'
  + '</div>'
  + '<div class="card"><h3 style="color:#fff">IP Model Stats</h3>'
  + '<div class="stat"><span class="label">Evaluations</span><span>'
  + {n_evaluations} + '</span></div>'
  + '<div class="stat"><span class="label">Mean probability</span><span>'
  + {mean_prob:.3f} + '</span></div>'
  + '<div class="stat"><span class="label">Max probability</span><span>'
  + {max_prob:.3f} + '</span></div>'
  + '<div class="stat"><span class="label">Threshold</span><span>'
  + THRESHOLD + '</span></div>'
  + '</div>';

// Probability chart
const chart = document.getElementById('prob-chart');
const thresholdPct = (1 - THRESHOLD) * 100;
chart.innerHTML = '<div class="threshold-line" style="top:' + thresholdPct + '%"></div>';
EVENTS.forEach(e => {{
  const left = pct(e.t);
  const h = Math.max(2, e.p * 100);
  const color = e.int ? '#ff1744' : (e.p > 0.3 ? '#ff9800' : '#4ecca3');
  chart.innerHTML += '<div style="position:absolute;bottom:0;left:' + left +
    ';width:3px;height:' + h + '%;background:' + color +
    ';border-radius:1px;opacity:0.8"></div>';
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interrupt prediction demo: compare IP enabled vs disabled",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_interrupt_prediction.py user_speech.wav
  python demo_interrupt_prediction.py user_speech.wav --bot-audio assistant.wav
  python demo_interrupt_prediction.py user_speech.wav --threshold 0.6 -v

Environment variables:
  KRISP_VIVA_IP_MODEL_PATH  Path to Krisp IP model (.kef)
        """,
    )

    parser.add_argument("input", help="Input audio file with user speech/interruptions")
    parser.add_argument(
        "--bot-audio",
        type=str,
        default=None,
        help="Optional assistant audio WAV file. If not provided, a synthetic tone is generated.",
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
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-frame IP probabilities",
    )

    args = parser.parse_args()

    model_path = os.getenv("KRISP_VIVA_IP_MODEL_PATH")
    if not model_path:
        print("Error: KRISP_VIVA_IP_MODEL_PATH environment variable not set")
        sys.exit(1)
    if not os.path.isfile(model_path):
        print(f"Error: IP model file not found: {model_path}")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if args.bot_audio and not os.path.exists(args.bot_audio):
        print(f"Error: Bot audio file not found: {args.bot_audio}")
        sys.exit(1)

    if not 0.0 <= args.threshold <= 1.0:
        print(f"Error: Threshold must be between 0.0 and 1.0, got {args.threshold}")
        sys.exit(1)

    process_audio(
        input_path=args.input,
        model_path=model_path,
        threshold=args.threshold,
        frame_duration_ms=args.frame_duration,
        bot_audio_path=args.bot_audio,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

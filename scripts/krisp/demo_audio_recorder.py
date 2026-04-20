#!/usr/bin/env python3
"""Record microphone audio for use with the turn-taking and IP demo tools.

Captures from the default (or specified) input device and saves a mono
16 kHz int16 WAV file -- the format expected by demo_turn_taking.py and
demo_interrupt_prediction.py.

Usage:
    python demo_audio_recorder.py                        # record to recording.wav
    python demo_audio_recorder.py -o my_session.wav      # custom output name
    python demo_audio_recorder.py -d 30                  # stop after 30 seconds
    python demo_audio_recorder.py --list-devices         # show available inputs
    python demo_audio_recorder.py --device 3             # use input device #3

After recording, run:
    python demo_turn_taking.py recording.wav --analyzer krisp --analyzer smart-turn-v3
    python demo_interrupt_prediction.py recording.wav

Requirements:
    pip install sounddevice numpy soundfile
"""

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
FRAME_MS = 20


class AudioRecorder:
    """Records microphone audio to a WAV file.

    Args:
        output_path: Destination WAV file path.
        sample_rate: Recording sample rate in Hz.
        max_duration: Optional maximum recording duration in seconds.
        device: Optional input device index.
    """

    def __init__(
        self,
        output_path: str = "recording.wav",
        sample_rate: int = SAMPLE_RATE,
        max_duration: float | None = None,
        device: int | None = None,
    ):
        self.output_path = output_path
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.device = device

        self._running = False
        self._stream: sd.InputStream | None = None
        self._buffers: list[np.ndarray] = []
        self._peak: int = 0
        self._start_time: float = 0.0
        self._actual_sample_rate: int = sample_rate

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status:
            print(f"  ⚠ {status}", file=sys.stderr)
        self._buffers.append(indata.copy())
        self._peak = max(self._peak, int(np.abs(indata).max()))

    def record(self) -> str | None:
        """Record until Ctrl-C or max_duration, then save.

        Returns:
            Path to the saved WAV file, or None if nothing was recorded.
        """
        self._running = True
        self._buffers.clear()
        self._peak = 0

        block_size = int(self.sample_rate * FRAME_MS / 1000)

        print(f"Recording to: {self.output_path}")
        print(f"  Requested sample rate: {self.sample_rate} Hz, frame: {FRAME_MS} ms, mono int16")
        if self.device is not None:
            dev_info = sd.query_devices(self.device)
            print(f"  Input device: [{self.device}] {dev_info['name']}")
        if self.max_duration:
            print(f"  Max duration: {self.max_duration:.0f}s")
        print("  Press Ctrl+C to stop.\n")

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=block_size,
            device=self.device,
            callback=self._callback,
        )

        # Read back the actual sample rate the device is using
        self._actual_sample_rate = int(self._stream.samplerate)
        if self._actual_sample_rate != self.sample_rate:
            print(
                f"  ⚠ Device does not support {self.sample_rate} Hz;"
                f" recording at {self._actual_sample_rate} Hz instead."
            )

        self._start_time = time.time()
        self._stream.start()

        try:
            while self._running:
                elapsed = time.time() - self._start_time
                level = self._level_bar(self._peak)
                print(
                    f"\r  ● REC  {elapsed:6.1f}s  {level}  peak={self._peak:5d}",
                    end="",
                    flush=True,
                )
                self._peak = 0

                if self.max_duration and elapsed >= self.max_duration:
                    print()
                    break

                sd.sleep(200)
        except KeyboardInterrupt:
            print()

        self._stream.stop()
        self._stream.close()
        self._stream = None

        return self._save()

    def _save(self) -> str | None:
        if not self._buffers:
            print("  Nothing recorded.")
            return None

        audio = np.concatenate(self._buffers, axis=0).squeeze()

        save_rate = self._actual_sample_rate

        if self._actual_sample_rate != self.sample_rate:
            try:
                import samplerate as src

                ratio = self.sample_rate / self._actual_sample_rate
                audio_float = audio.astype(np.float32) / 32768.0
                resampled = src.resample(audio_float, ratio, "sinc_best")
                audio = (resampled * 32768.0).clip(-32768, 32767).astype(np.int16)
                save_rate = self.sample_rate
                print(f"\n  Resampled {self._actual_sample_rate} Hz → {self.sample_rate} Hz")
            except ImportError:
                print(
                    f"\n  ⚠ Saving at {self._actual_sample_rate} Hz (install 'samplerate'"
                    f" package to auto-resample to {self.sample_rate} Hz)"
                )

        duration = len(audio) / save_rate
        sf.write(self.output_path, audio, save_rate, subtype="PCM_16")

        print(f"\n  Saved: {self.output_path}")
        print(f"  Sample rate: {save_rate} Hz")
        print(f"  Duration: {duration:.2f}s  ({len(audio)} samples)")
        print(f"  Range: [{audio.min()}, {audio.max()}]")
        self._print_next_steps()
        return self.output_path

    def _print_next_steps(self):
        name = Path(self.output_path).name
        print("\nNext steps:")
        print(f"  python demo_turn_taking.py {name} --analyzer krisp --analyzer smart-turn-v3")
        print(f"  python demo_interrupt_prediction.py {name}")

    @staticmethod
    def _level_bar(peak: int, width: int = 30) -> str:
        frac = min(peak / 32767, 1.0)
        filled = int(frac * width)
        bar = "█" * filled + "░" * (width - filled)
        if frac > 0.95:
            return f"|{bar}| CLIP"
        return f"|{bar}|"


def list_devices():
    """Print available audio input devices."""
    print("\nAudio Input Devices:")
    sep = "=" * 75
    print(sep)
    print(f"  {'#':<5} {'Name':<50} {'Ch':>4}  {'Rate':>6}")
    print(f"  {'-' * 5} {'-' * 50} {'-' * 4}  {'-' * 6}")

    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            marker = " *" if i == sd.default.device[0] else "  "
            print(
                f"{marker}{i:<5} {dev['name']:<50} {dev['max_input_channels']:>4}"
                f"  {int(dev['default_samplerate']):>5}Hz"
            )

    default = sd.query_devices(kind="input")
    print(f"\n  * Default input: {default['name']}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Record microphone audio for the turn-taking / IP demo tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_audio_recorder.py
  python demo_audio_recorder.py -o conversation.wav
  python demo_audio_recorder.py -d 60
  python demo_audio_recorder.py --list-devices
  python demo_audio_recorder.py --device 3 -o test.wav
        """,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="recording.wav",
        help="Output WAV file path (default: recording.wav)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=float,
        default=None,
        help="Max recording duration in seconds (default: unlimited, Ctrl+C to stop)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Input device index (use --list-devices to see options)",
    )
    parser.add_argument(
        "-r",
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        choices=[8000, 16000, 24000, 32000, 48000],
        help="Sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--list-devices",
        "-l",
        action="store_true",
        help="List available audio input devices and exit",
    )

    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal.default_int_handler)

    if args.list_devices:
        list_devices()
        return

    recorder = AudioRecorder(
        output_path=args.output,
        sample_rate=args.sample_rate,
        max_duration=args.duration,
        device=args.device,
    )
    saved = recorder.record()
    if not saved:
        sys.exit(1)


if __name__ == "__main__":
    main()

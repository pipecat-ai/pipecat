"""Probe signal generation and onset detection for the transport latency benchmark.

The probe is silence with a short Hann-windowed linear chirp every ``period_ms``.
Chirps survive lossy Opus round-trips; a matched filter (normalized
cross-correlation against the known template) recovers each onset with
sub-millisecond resolution. A lost chirp is a counted drop, not a corrupted
measurement.

Pure numpy, no I/O. Run as a script for the self-test (clean + Opus round-trip).
"""

import numpy as np

SAMPLE_RATE = 48000


def chirp_template(sample_rate: int = SAMPLE_RATE, chirp_ms: int = 20) -> np.ndarray:
    """Hann-windowed 300 Hz -> 3 kHz linear chirp, float32 in [-1, 1]."""
    n = int(sample_rate * chirp_ms / 1000)
    t = np.arange(n) / sample_rate
    f0, f1 = 300.0, 3000.0
    # Linear chirp: phase = 2*pi*(f0*t + (f1-f0)/(2*T)*t^2)
    T = n / sample_rate
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * T) * t * t)
    return (np.sin(phase) * np.hanning(n)).astype(np.float32)


def gen_probe(
    duration_s: float,
    sample_rate: int = SAMPLE_RATE,
    chirp_ms: int = 20,
    period_ms: int = 250,
    amplitude: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the probe signal.

    Returns (signal_int16, marker_onset_positions). Marker k's chirp starts at
    ``marker_onset_positions[k]`` (samples). The first marker starts one full
    period in, so no chirp straddles the stream start.
    """
    total = int(duration_s * sample_rate)
    template = chirp_template(sample_rate, chirp_ms) * amplitude
    period = int(sample_rate * period_ms / 1000)
    signal = np.zeros(total, dtype=np.float32)
    positions = []
    pos = period
    while pos + len(template) <= total:
        signal[pos : pos + len(template)] += template
        positions.append(pos)
        pos += period
    return (signal * 32767).astype(np.int16), np.array(positions, dtype=np.int64)


def detect_onsets(
    signal_i16: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    chirp_ms: int = 20,
    period_ms: int = 250,
    threshold: float = 0.35,
) -> np.ndarray:
    """Find chirp onset positions (samples) in a received int16 signal.

    Normalized matched filter: FFT cross-correlation with the template,
    normalized by the sliding RMS of the signal under the template window, so
    detection is robust to codec level changes. Peaks above ``threshold`` with
    at least half a period of separation are onsets.
    """
    if len(signal_i16) == 0:
        return np.array([], dtype=np.int64)
    x = signal_i16.astype(np.float32) / 32768.0
    template = chirp_template(sample_rate, chirp_ms)
    m = len(template)

    n = len(x) + m - 1
    nfft = 1 << (n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(x, nfft) * np.conj(np.fft.rfft(template, nfft)), nfft)[
        : len(x) - m + 1
    ]

    # Sliding L2 norm of the signal under the template window.
    csum = np.concatenate(([0.0], np.cumsum(x.astype(np.float64) ** 2)))
    window_energy = csum[m:] - csum[: len(x) - m + 1]
    denom = np.sqrt(window_energy * float(np.sum(template**2))) + 1e-9
    ncc = corr / denom
    # Near-silent windows make the normalization a noise amplifier
    # (FFT numerical noise / epsilon) — mask them out.
    ncc[window_energy < 1e-6 * m] = 0.0

    min_dist = int(sample_rate * period_ms / 1000) // 2
    onsets = []
    i = 0
    while i < len(ncc):
        if ncc[i] > threshold:
            j = min(i + min_dist, len(ncc))
            peak = i + int(np.argmax(ncc[i:j]))
            onsets.append(peak)
            i = peak + min_dist
        else:
            i += 1
    return np.array(onsets, dtype=np.int64)


def _self_test() -> None:
    """Verify sub-millisecond onset recovery on clean and Opus-round-tripped audio."""
    sr = SAMPLE_RATE
    signal, positions = gen_probe(10.0, sr)

    detected = detect_onsets(signal, sr)
    assert len(detected) == len(positions), (len(detected), len(positions))
    err = np.abs(detected - positions)
    print(f"clean: {len(detected)} onsets, max error {err.max() / sr * 1000:.3f} ms")
    assert err.max() < sr / 1000, "clean onset error >= 1 ms"

    # Opus round-trip via PyAV (libopus), 48 kHz mono 20 ms frames.
    import av

    def opus_roundtrip(pcm: np.ndarray) -> np.ndarray:
        enc = av.CodecContext.create("libopus", "w")
        enc.sample_rate = sr
        enc.layout = "mono"
        enc.format = "s16"
        dec = av.CodecContext.create("opus", "r")
        dec.sample_rate = sr
        dec.layout = "mono"
        out = []
        frame_samples = sr // 50  # 20 ms
        for start in range(0, len(pcm) - frame_samples + 1, frame_samples):
            frame = av.AudioFrame(format="s16", layout="mono", samples=frame_samples)
            frame.sample_rate = sr
            frame.pts = start
            frame.planes[0].update(pcm[start : start + frame_samples].tobytes())
            for pkt in enc.encode(frame):
                for dfr in dec.decode(pkt):
                    arr = dfr.to_ndarray().reshape(-1)
                    if arr.dtype.kind == "f":  # decoder may emit float planar
                        arr = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
                    out.append(arr.astype(np.int16))
        return np.concatenate(out) if out else np.array([], dtype=np.int16)

    decoded = opus_roundtrip(signal)
    detected2 = detect_onsets(decoded, sr)
    # Opus adds a constant codec delay; compare spacing-aligned errors.
    assert len(detected2) == len(positions), (len(detected2), len(positions))
    delays = detected2 - positions
    spread = delays.max() - delays.min()
    print(
        f"opus:  {len(detected2)} onsets, codec delay ~{np.median(delays) / sr * 1000:.2f} ms, "
        f"spread {spread / sr * 1000:.3f} ms"
    )
    assert spread < sr / 1000, "opus onset jitter >= 1 ms"
    print("probe self-test OK")


if __name__ == "__main__":
    _self_test()

"""Shared helpers for the smart-turn ONNX shared-session benchmarks.

All three benchmark scripts (correctness / latency_throughput / memory) import
from here so they build sessions identically and locate the model the same way.

Dependencies: numpy, onnxruntime. pipecat is optional and only used to locate
the bundled smart-turn-v3.2 model; pass --model to point at the file directly.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys

import numpy as np
import onnxruntime as ort

# Model input is [batch, n_mels, frames]; smart-turn-v3.2 uses 80 mel bins and
# 800 frames (8 s @ 16 kHz, hop 10 ms). Values are irrelevant to compute cost,
# so synthetic input is representative for latency/throughput and lets the
# correctness test feed every thread the exact same tensor.
INPUT_NAME = "input_features"
INPUT_SHAPE = (1, 80, 800)
MODEL_RESOURCE = ("pipecat.audio.turn.smart_turn.data", "smart-turn-v3.2-cpu.onnx")


def resolve_model_path(explicit: str | None) -> str:
    """Return a filesystem path to the ONNX model.

    Order: --model argument, then the model bundled with an installed pipecat.
    """
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"--model path does not exist: {explicit}")
        return explicit
    try:
        from importlib import resources

        pkg, name = MODEL_RESOURCE
        path = str(resources.files(pkg).joinpath(name))
        if os.path.exists(path):
            return path
    except Exception:  # noqa: BLE001 - fall through to a helpful error
        pass
    sys.exit(
        "Could not locate the model. Install pipecat-ai (which bundles "
        f"{MODEL_RESOURCE[1]}) or pass --model /path/to/{MODEL_RESOURCE[1]}"
    )


def make_session(model_path: str, intra_op: int = 1) -> ort.InferenceSession:
    """Build an InferenceSession with the SAME options pipecat's
    LocalSmartTurnAnalyzerV3 uses, so the benchmark reflects production.

    Reference: pipecat/audio/turn/smart_turn/local_smart_turn_v3.py
        execution_mode = ORT_SEQUENTIAL
        inter_op_num_threads = 1
        intra_op_num_threads = cpu_count   (pipecat default cpu_count=1)
        graph_optimization_level = ORT_ENABLE_ALL
    """
    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = intra_op
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=so)


def make_input(seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(INPUT_SHAPE, dtype=np.float32)


def current_rss_mb() -> float:
    """Current resident set size of THIS process, in MiB. Cross-platform.

    Uses `ps -o rss=` (KiB on both Linux and macOS). Falls back to
    resource.ru_maxrss (a high-water mark, not current) if ps is unavailable.
    """
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
        ).strip()
        return int(out) / 1024.0
    except Exception:  # noqa: BLE001
        import resource

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss units: bytes on macOS, KiB on Linux. Convert to MiB.
        if sys.platform == "darwin":
            return maxrss / (1024.0 * 1024.0)
        return maxrss / 1024.0


def print_environment(model_path: str) -> None:
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print("=" * 72)
    print("Environment")
    print("-" * 72)
    print(f"  platform            : {platform.platform()}")
    print(f"  python              : {platform.python_version()}")
    print(f"  onnxruntime         : {ort.__version__}")
    print(f"  logical CPUs        : {os.cpu_count()}")
    print(f"  ORT providers       : {ort.get_available_providers()}")
    print(f"  model               : {model_path}")
    print(f"  model size on disk  : {size_mb:.2f} MB")
    print("=" * 72)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=None,
        help="Path to smart-turn-v3.2-cpu.onnx (default: locate via pipecat install)",
    )

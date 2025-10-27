#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Test runner for frame processors from JSON test files."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameProcessor

from .serialization import dict_to_frame, frame_to_dict, load_frames_from_json


async def run_test_from_file(
    processor: FrameProcessor,
    test_file: str,
) -> Tuple[List[Frame], Optional[List[Dict[str, Any]]], bool]:
    """Run a processor test from a JSON test file.

    Args:
        processor: The frame processor to test
        test_file: Path to JSON test file

    Returns:
        Tuple of (output_frames, expected_output, passed)
        - output_frames: List of Frame objects that were output
        - expected_output: List of expected frame dicts (None if not specified)
        - passed: True if test passed, False if failed, None if no validation

    Raises:
        FileNotFoundError: If test file doesn't exist
        ValueError: If test file is invalid

    Example test file format:
        {
          "input_frames": [
            {"type": "TextFrame", "text": "hello"}
          ],
          "expected_output": [
            {"type": "TextFrame"},
            {"type": "EndFrame"}
          ]
        }
    """
    path = Path(test_file)
    if not path.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")

    with open(path, "r") as f:
        test_data = json.load(f)

    # Load input frames
    if "input_frames" not in test_data:
        raise ValueError("Test file must contain 'input_frames'")

    input_frames = [dict_to_frame(frame_dict) for frame_dict in test_data["input_frames"]]

    # Load expected output (optional)
    expected_output = test_data.get("expected_output", None)

    # Run the test
    # Note: run_test() only collects frames if expected_down_frames is provided,
    # so we need to manually collect from the pipeline ourselves
    import asyncio
    from pipecat.frames.frames import EndFrame
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat.tests.utils import QueuedFrameProcessor
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineTask, PipelineParams
    from pipecat.pipeline.runner import PipelineRunner

    # Set up the test pipeline manually
    received_down = asyncio.Queue()
    received_up = asyncio.Queue()
    source = QueuedFrameProcessor(
        queue=received_up,
        queue_direction=FrameDirection.UPSTREAM,
        ignore_start=True,
    )
    sink = QueuedFrameProcessor(
        queue=received_down,
        queue_direction=FrameDirection.DOWNSTREAM,
        ignore_start=True,
    )

    pipeline = Pipeline([source, processor, sink])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(),
        observers=[],
        cancel_on_idle_timeout=False,
    )

    async def push_frames():
        await asyncio.sleep(0.01)
        for frame in input_frames:
            await task.queue_frame(frame)
        await task.queue_frame(EndFrame())

    runner = PipelineRunner()
    await asyncio.gather(runner.run(task), push_frames())

    # Collect all frames from the downstream queue
    downstream_frames = []
    while not received_down.empty():
        frame = await received_down.get()
        if not isinstance(frame, EndFrame):
            downstream_frames.append(frame)

    # Validate if expected_output is provided
    passed = None
    if expected_output is not None:
        passed = _validate_output(downstream_frames, expected_output)

    return downstream_frames, expected_output, passed


def _validate_output(actual_frames: List[Frame], expected_output: List[Dict[str, Any]]) -> bool:
    """Validate actual output frames against expected output.

    Args:
        actual_frames: List of frames that were actually output
        expected_output: List of expected frame specifications

    Returns:
        True if validation passed, False otherwise
    """
    if len(actual_frames) != len(expected_output):
        return False

    for actual, expected in zip(actual_frames, expected_output):
        # Check frame type
        if "type" not in expected:
            return False

        expected_type = expected["type"]
        if actual.__class__.__name__ != expected_type:
            return False

        # Check specific fields if provided
        for field_name, expected_value in expected.items():
            if field_name == "type":
                continue

            if not hasattr(actual, field_name):
                return False

            actual_value = getattr(actual, field_name)

            # Special handling for different types
            if isinstance(expected_value, str) and isinstance(actual_value, str):
                # For string fields, support partial matching with "contains"
                if field_name.endswith("_contains"):
                    base_field = field_name.replace("_contains", "")
                    if hasattr(actual, base_field):
                        actual_text = getattr(actual, base_field)
                        if expected_value not in actual_text:
                            return False
                elif actual_value != expected_value:
                    return False
            elif actual_value != expected_value:
                return False

    return True

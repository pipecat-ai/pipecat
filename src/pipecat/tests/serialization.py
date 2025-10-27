#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Frame serialization and deserialization for testing."""

import base64
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List

from pipecat.frames import frames


def _get_frame_class(frame_type: str):
    """Get a frame class by name from the frames module.
    
    Args:
        frame_type: The name of the frame class (e.g., "TextFrame")
        
    Returns:
        The frame class object
        
    Raises:
        ValueError: If the frame type is not found
    """
    if not hasattr(frames, frame_type):
        raise ValueError(f"Unknown frame type: {frame_type}")
    
    cls = getattr(frames, frame_type)
    if not inspect.isclass(cls) or not issubclass(cls, frames.Frame):
        raise ValueError(f"{frame_type} is not a valid Frame class")
    
    return cls


def dict_to_frame(data: Dict[str, Any]) -> frames.Frame:
    """Convert a dictionary to a Frame object.
    
    Args:
        data: Dictionary containing frame data with a "type" key
        
    Returns:
        A Frame instance
        
    Raises:
        ValueError: If frame type is missing or invalid
        
    Example:
        >>> dict_to_frame({"type": "TextFrame", "text": "hello"})
        TextFrame(text="hello")
    """
    if "type" not in data:
        raise ValueError("Frame dictionary must contain a 'type' field")
    
    frame_type = data["type"]
    frame_cls = _get_frame_class(frame_type)
    
    # Build kwargs from data, excluding 'type'
    kwargs = {k: v for k, v in data.items() if k != "type"}
    
    # Special handling for audio frames with base64 encoded audio
    if "audio" in kwargs and isinstance(kwargs["audio"], str):
        kwargs["audio"] = base64.b64decode(kwargs["audio"])
    
    # Special handling for image frames with base64 encoded images
    if "image" in kwargs and isinstance(kwargs["image"], str):
        kwargs["image"] = base64.b64decode(kwargs["image"])
    
    try:
        return frame_cls(**kwargs)
    except TypeError as e:
        raise ValueError(f"Failed to create {frame_type}: {e}")


def load_frames_from_json(filepath: str) -> List[frames.Frame]:
    """Load frames from a JSON file.
    
    Args:
        filepath: Path to JSON file containing frame data
        
    Returns:
        List of Frame objects
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If JSON is invalid or frames cannot be deserialized
        
    Example JSON format:
        {
          "input_frames": [
            {"type": "TextFrame", "text": "hello"},
            {"type": "EndFrame"}
          ]
        }
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Frame file not found: {filepath}")
    
    with open(path, "r") as f:
        data = json.load(f)
    
    if not isinstance(data, dict):
        raise ValueError("JSON must contain a dictionary")
    
    if "input_frames" not in data:
        raise ValueError("JSON must contain an 'input_frames' key")
    
    frame_dicts = data["input_frames"]
    if not isinstance(frame_dicts, list):
        raise ValueError("'input_frames' must be a list")
    
    return [dict_to_frame(frame_dict) for frame_dict in frame_dicts]


def frame_to_dict(frame: frames.Frame) -> Dict[str, Any]:
    """Convert a Frame object to a dictionary.
    
    Args:
        frame: Frame object to serialize
        
    Returns:
        Dictionary representation of the frame
        
    Example:
        >>> frame_to_dict(TextFrame(text="hello"))
        {"type": "TextFrame", "text": "hello"}
    """
    result = {"type": frame.__class__.__name__}
    
    # Get all fields from the dataclass
    if hasattr(frame, "__dataclass_fields__"):
        for field_name in frame.__dataclass_fields__:
            # Skip internal fields from base Frame class
            if field_name in ("id", "name", "pts", "metadata", "transport_source", "transport_destination"):
                continue
            
            value = getattr(frame, field_name, None)
            if value is not None:
                # Special handling for bytes (audio/image data)
                if isinstance(value, bytes):
                    result[field_name] = base64.b64encode(value).decode("utf-8")
                else:
                    result[field_name] = value
    
    return result

#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""JSON-based bus message serializer with pluggable type adapters."""

import base64
import dataclasses
import importlib
import json
from enum import Enum
from functools import cache
from typing import Any

from loguru import logger
from pydantic import BaseModel

from pipecat.bus.adapters.base import TypeAdapter
from pipecat.bus.messages import BusMessage
from pipecat.bus.serializers.base import MessageSerializer

# JSON-native types that don't need an adapter.
_JSON_NATIVE = (str, int, float, bool, type(None))


class JSONMessageSerializer(MessageSerializer):
    """Serialize bus messages as JSON with pluggable type adapters.

    Handles JSON-native types, enums, bytes, dataclasses, and any type
    with a registered ``TypeAdapter`` (e.g. ``LLMContext``, ``ToolsSchema``).
    Adapters for common Pipecat types are registered by default.
    Additional type adapters can be registered via ``register_adapter()``.

    Example::

        serializer = JSONMessageSerializer()

        data = serializer.serialize(message)
        restored = serializer.deserialize(data)
    """

    def __init__(self):
        """Create a serializer with default adapters for `LLMContext` and `ToolsSchema`."""
        from pipecat.adapters.schemas.tools_schema import ToolsSchema
        from pipecat.bus.adapters import LLMContextAdapter, ToolsSchemaAdapter
        from pipecat.processors.aggregators.llm_context import LLMContext

        self._adapters: dict[type, TypeAdapter] = {
            LLMContext: LLMContextAdapter(),
            ToolsSchema: ToolsSchemaAdapter(),
        }

    def register_adapter(self, type_: type, adapter: TypeAdapter) -> None:
        """Register a type adapter.

        Args:
            type_: The type to handle.
            adapter: The adapter that serializes/deserializes instances of this type.
        """
        self._adapters[type_] = adapter

    def serialize(self, message: BusMessage) -> bytes:
        """Convert a bus message to JSON bytes.

        Args:
            message: The bus message to serialize.

        Returns:
            UTF-8 encoded JSON bytes.
        """
        data = self._serialize_value(message)
        return json.dumps(data, separators=(",", ":")).encode("utf-8")

    def deserialize(self, data: bytes) -> BusMessage | None:
        """Reconstruct a bus message from JSON bytes.

        Args:
            data: The JSON bytes produced by `serialize()`.

        Returns:
            The reconstructed `BusMessage`, or None if deserialization fails.
        """
        payload = json.loads(data)
        return self._deserialize_value(payload)

    def _serialize_value(self, value: Any) -> Any:
        """Recursively serialize a value to a JSON-compatible representation."""
        if isinstance(value, _JSON_NATIVE):
            return value
        if isinstance(value, Enum):
            return {
                "__type__": f"{type(value).__module__}.{type(value).__name__}",
                "__data__": value.name,
            }
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        if isinstance(value, bytes):
            return {"__type__": "bytes", "__data__": base64.b64encode(value).decode("ascii")}
        if isinstance(value, BaseModel):
            return {
                "__type__": f"{type(value).__module__}.{type(value).__name__}",
                "__data__": {
                    k: self._serialize_value(v) for k, v in value.__dict__.items() if v is not None
                },
            }
        if callable(value):
            return None
        adapter = self._find_adapter(type(value))
        if adapter is not None:
            return {
                "__type__": f"{type(value).__module__}.{type(value).__name__}",
                "__data__": adapter.serialize(value, self._serialize_value),
            }
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            fields = {}
            for f in dataclasses.fields(value):
                v = getattr(value, f.name)
                if v is None:
                    continue
                serialized = self._serialize_value(v)
                if serialized is not None:
                    fields[f.name] = serialized
            return {
                "__type__": f"{type(value).__module__}.{type(value).__name__}",
                "__data__": fields,
            }
        logger.warning(
            f"JSONMessageSerializer: skipping field with unserializable type {type(value).__name__}"
        )
        return None

    def _deserialize_value(self, value: Any) -> Any:
        """Recursively deserialize a value from its JSON representation."""
        if isinstance(value, _JSON_NATIVE):
            return value
        if isinstance(value, list):
            return [self._deserialize_value(v) for v in value]
        if isinstance(value, dict):
            if "__type__" in value and "__data__" in value:
                return self._deserialize_typed(value["__type__"], value["__data__"])
            return {k: self._deserialize_value(v) for k, v in value.items()}
        return value

    def _deserialize_typed(self, type_name: str, data: Any) -> Any:
        """Deserialize a tagged value using its fully qualified type name."""
        if type_name == "bytes":
            return base64.b64decode(data)
        cls = _resolve_type(type_name)
        if cls is None:
            logger.warning(f"JSONMessageSerializer: could not resolve type {type_name}")
            return None
        if issubclass(cls, Enum):
            return cls[data]
        adapter = self._find_adapter(cls)
        if adapter is not None:
            return adapter.deserialize(data, self._deserialize_value, target_type=cls)
        if isinstance(data, dict) and issubclass(cls, BaseModel):
            return cls.model_validate({k: self._deserialize_value(v) for k, v in data.items()})
        if dataclasses.is_dataclass(cls) and isinstance(data, dict):
            init_fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
            init_kwargs = {}
            post_init = {}
            for key, value in data.items():
                deserialized = self._deserialize_value(value)
                if key in init_fields:
                    init_kwargs[key] = deserialized
                else:
                    post_init[key] = deserialized
            for name, f in init_fields.items():
                if name not in init_kwargs:
                    if (
                        f.default is dataclasses.MISSING
                        and f.default_factory is dataclasses.MISSING
                    ):
                        init_kwargs[name] = None
            obj = cls(**init_kwargs)
            for key, value in post_init.items():
                setattr(obj, key, value)
            return obj
        logger.warning(f"JSONMessageSerializer: no adapter registered for type {type_name}")
        return None

    def _find_adapter(self, type_: type) -> TypeAdapter | None:
        """Find an adapter for a type, checking parent classes via MRO."""
        for cls in type_.__mro__:
            if cls in self._adapters:
                return self._adapters[cls]
        return None


@cache
def _resolve_type(qualified_name: str) -> type | None:
    """Resolve a fully qualified type name to its class.

    Args:
        qualified_name: Dotted path like ``"pipecat.frames.frames.TextFrame"``.

    Returns:
        The resolved class, or None if it cannot be found.
    """
    module_path, _, class_name = qualified_name.rpartition(".")
    if not module_path:
        return None
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name, None)
    except ImportError:
        return None

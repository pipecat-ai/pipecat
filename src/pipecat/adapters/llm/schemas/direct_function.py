#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Direct function wrapper utilities for LLM function calling.

This module provides utilities for wrapping "direct" functions that handle LLM
function calls. Direct functions have their metadata automatically extracted
from function signatures and docstrings, allowing them to be used without
accompanying configurations (as FunctionSchemas or in provider-specific
formats).
"""

import dataclasses
import enum
import inspect
import types
from collections.abc import Awaitable, Callable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import docstring_parser
from pydantic import BaseModel, TypeAdapter

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams


DirectFunction: TypeAlias = Callable[..., Awaitable[Any]]
"""A "direct" function that handles LLM function calls.

A direct function is an async function whose first parameter is ``params`` (a
``FunctionCallParams``), followed by the tool's arguments. Its metadata (name,
description, and parameter schemas) is extracted automatically from its
signature and docstring, so it can be used without an accompanying
``FunctionSchema`` or provider-specific configuration.

Typed as a permissive ``Callable`` so that concrete function signatures (which a
stricter ``Protocol`` form would reject) type-check when passed to public APIs.
The precise contract — async, with a first parameter named ``params`` — is
validated at runtime by ``DirectFunctionWrapper.validate_function``.
"""


class BaseDirectFunctionWrapper:
    """Base class for a wrapper around a DirectFunction.

    Provides functionality to:

    - extract metadata from the function signature and docstring
    - use that metadata to generate a corresponding FunctionSchema
    """

    def __init__(self, function: Callable):
        """Initialize the direct function wrapper.

        Args:
            function: The function to wrap and extract metadata from.
        """
        self.__class__.validate_function(function)
        self.function = function
        self._initialize_metadata()

    @classmethod
    def special_first_param_name(cls) -> str:
        """Get the name of the special first function parameter.

        The special first parameter is ignored by metadata extraction as it's
        not relevant to the LLM (e.g., 'params' for FunctionCallParams).

        Returns:
            The name of the special first parameter.
        """
        raise NotImplementedError("Subclasses must define the special first parameter name.")

    @classmethod
    def validate_function(cls, function: Callable) -> None:
        """Validate that the function meets direct function requirements.

        Args:
            function: The function to validate.

        Raises:
            Exception: If function doesn't meet requirements (not async, missing
                parameters, incorrect first parameter name).
        """
        if not inspect.iscoroutinefunction(function):
            raise Exception(f"Direct function {function.__name__} must be async")
        params = list(inspect.signature(function).parameters.items())
        special_first_param_name = cls.special_first_param_name()
        if len(params) == 0:
            raise Exception(
                f"Direct function {function.__name__} must have at least one parameter ({special_first_param_name})"
            )
        first_param_name = params[0][0]
        if first_param_name != special_first_param_name:
            raise Exception(
                f"Direct function {function.__name__} first parameter must be named '{special_first_param_name}'"
            )

    def to_function_schema(self) -> FunctionSchema:
        """Convert the wrapped function to a FunctionSchema.

        Returns:
            A FunctionSchema instance with extracted metadata.
        """
        return FunctionSchema(
            name=self.name,
            description=self.description,
            properties=self.properties,
            required=self.required,
        )

    def _initialize_metadata(self):
        """Initialize metadata from function signature and docstring."""
        # Get function name
        self.name = self.function.__name__

        # Parse docstring for description and parameters
        docstring = docstring_parser.parse(inspect.getdoc(self.function) or "")

        # Get function description
        self.description = (docstring.description or "").strip()

        # Get function parameters as JSON schemas, and the list of required parameters
        self.properties, self.required = self._get_parameters_as_jsonschema(
            self.function, docstring.params
        )

    def _get_parameters_as_jsonschema(
        self, func: Callable, docstring_params: list[docstring_parser.DocstringParam]
    ) -> tuple[dict[str, Any], list[str]]:
        """Get function parameters as a dictionary of JSON schemas and a list of required parameters.

        Ignore the first parameter, as it's expected to be the "special" one.

        Args:
            func: Function to get parameters from.
            docstring_params: List of parameters extracted from the function's docstring.

        Returns:
            A tuple of a dictionary mapping each function parameter to its JSON
            schema, and a list of required parameter names.
        """
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        properties = {}
        required = []
        # Adapters that turn JSON arguments back into the declared types (enums,
        # Pydantic models, dataclasses) before the function is called.
        self._argument_adapters: dict[str, TypeAdapter[Any]] = {}

        for name, param in sig.parameters.items():
            # Ignore 'self' parameter
            if name == "self":
                continue

            # Ignore the first parameter, which is expected to be the "special" one
            # (We have already validated that this is the case in validate_function())
            is_first_param = name == next(iter(sig.parameters))
            if is_first_param:
                continue

            type_hint = hints.get(name)

            # Convert type hint to JSON schema
            properties[name] = self._typehint_to_jsonschema(type_hint)
            if _needs_conversion(type_hint):
                self._argument_adapters[name] = TypeAdapter(type_hint)

            # Add whether the parameter is required
            # If the parameter has no default value, it's required
            if param.default is inspect.Parameter.empty:
                required.append(name)

            # Add parameter description from docstring
            for doc_param in docstring_params:
                if doc_param.arg_name == name:
                    properties[name]["description"] = doc_param.description or ""

        return properties, required

    def _convert_arguments(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """Convert JSON arguments from the LLM to the function's declared types.

        Arguments declared as an ``Enum``, a Pydantic model or a dataclass (also
        inside ``Optional``, lists and dicts) arrive as plain JSON values. They are
        validated and converted, so the function receives the types it declares.
        Other arguments are passed through unchanged.

        Args:
            args: Arguments from the LLM.

        Returns:
            The arguments to call the function with.

        Raises:
            pydantic.ValidationError: If an argument doesn't match its declared type.
        """
        return {
            name: (
                self._argument_adapters[name].validate_python(value)
                if name in self._argument_adapters
                else value
            )
            for name, value in args.items()
        }

    def _typehint_to_jsonschema(self, type_hint: Any) -> dict[str, Any]:
        """Convert a Python type hint to a JSON Schema.

        Args:
            type_hint: A Python type hint

        Returns:
            A dictionary representing the JSON Schema
        """
        if type_hint is None or type_hint is Any:
            return {}

        # Handle enums and literals as a list of allowed values. Checked before the
        # basic types, since a str or int enum is also a str or int subclass.
        if isinstance(type_hint, type) and issubclass(type_hint, enum.Enum):
            return _enum_schema([member.value for member in type_hint])
        if get_origin(type_hint) is Literal:
            return _enum_schema(list(get_args(type_hint)))

        # Handle basic types
        if type_hint is type(None):
            return {"type": "null"}
        if type_hint is str:
            return {"type": "string"}
        elif type_hint is int:
            return {"type": "integer"}
        elif type_hint is float:
            return {"type": "number"}
        elif type_hint is bool:
            return {"type": "boolean"}
        elif type_hint is dict:
            return {"type": "object"}
        elif type_hint is list:
            return {"type": "array"}

        # Get origin and arguments for complex types
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        # Handle Optional/Union types
        if origin is Union or origin is types.UnionType:
            return {"anyOf": [self._typehint_to_jsonschema(arg) for arg in args]}

        # Handle fixed-length tuples such as tuple[int, str]: every position must be
        # allowed, not only the first one
        if origin is tuple and args and args[-1] is not Ellipsis:
            item_schemas: list[dict[str, Any]] = []
            for arg in args:
                item_schema = self._typehint_to_jsonschema(arg)
                if item_schema not in item_schemas:
                    item_schemas.append(item_schema)
            items = item_schemas[0] if len(item_schemas) == 1 else {"anyOf": item_schemas}
            return {"type": "array", "items": items, "minItems": len(args), "maxItems": len(args)}

        # Handle List, Tuple, Set with specific item types
        if origin in (list, tuple, set) and args:
            return {"type": "array", "items": self._typehint_to_jsonschema(args[0])}

        # Handle Dict with specific key/value types
        if origin is dict and len(args) == 2:
            # For JSON Schema, keys must be strings
            return {"type": "object", "additionalProperties": self._typehint_to_jsonschema(args[1])}

        # Handle Pydantic models: fields with defaults are optional, and fields
        # are keyed by alias, which is what validation expects
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            properties = {}
            required = []
            for field_name, field in type_hint.model_fields.items():
                key = field.alias or field_name
                properties[key] = self._typehint_to_jsonschema(field.annotation)
                if field.description:
                    properties[key]["description"] = field.description
                if field.is_required():
                    required.append(key)
            return _object_schema(properties, required)

        # Handle dataclasses: fields with a default or default factory are optional
        if isinstance(type_hint, type) and dataclasses.is_dataclass(type_hint):
            field_types = get_type_hints(type_hint)
            properties = {}
            required = []
            for field in dataclasses.fields(type_hint):
                if not field.init:
                    continue
                properties[field.name] = self._typehint_to_jsonschema(field_types[field.name])
                if (
                    field.default is dataclasses.MISSING
                    and field.default_factory is dataclasses.MISSING
                ):
                    required.append(field.name)
            return _object_schema(properties, required)

        # Handle TypedDict
        if hasattr(type_hint, "__annotations__"):
            properties = {}
            required = []

            # NOTE: this does not yet support some fields being required and others not, which could happen when:
            # - the base class is a TypedDict with required fields (total=True or not specified) and the derived class has optional fields (total=False)
            # - Python 3.11+ NotRequired is used
            all_fields_required = getattr(type_hint, "__total__", True)

            for field_name, field_type in get_type_hints(type_hint).items():
                properties[field_name] = self._typehint_to_jsonschema(field_type)
                if all_fields_required:
                    required.append(field_name)

            schema = {"type": "object", "properties": properties}

            if required:
                schema["required"] = required

            return schema

        # Default to any type if we can't determine the specific schema
        return {}


_ENUM_VALUE_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def _enum_schema(values: list[Any]) -> dict[str, Any]:
    """JSON schema for a fixed set of allowed values.

    Adds the JSON type when every value has the same one, since some providers
    require a type next to ``enum``.
    """
    schema: dict[str, Any] = {"enum": values}
    value_types = {_ENUM_VALUE_TYPES.get(type(value)) for value in values}
    if len(value_types) == 1 and None not in value_types:
        schema = {"type": value_types.pop(), **schema}
    return schema


def _object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """JSON schema for an object with the given properties."""
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _needs_conversion(type_hint: Any) -> bool:
    """Whether a JSON value must be converted to match this type hint.

    True for hints that are, or contain, an ``Enum``, a Pydantic model or a
    dataclass, since JSON has no such values.
    """
    if isinstance(type_hint, type):
        return issubclass(type_hint, (enum.Enum, BaseModel)) or dataclasses.is_dataclass(type_hint)
    return any(_needs_conversion(arg) for arg in get_args(type_hint))


class DirectFunctionWrapper(BaseDirectFunctionWrapper):
    """Wrapper around a DirectFunction for LLM function calling.

    This class:

    - Extracts metadata from the function signature and docstring
    - Generates a corresponding FunctionSchema
    - Helps with function invocation
    """

    @classmethod
    def special_first_param_name(cls) -> str:
        """Get the special first parameter name for direct functions.

        Returns:
            The string "params" which is expected as the first parameter.
        """
        return "params"

    async def invoke(self, args: Mapping[str, Any], params: "FunctionCallParams"):
        """Invoke the wrapped function with the provided arguments.

        Args:
            args: Arguments to pass to the function.
            params: Function call parameters from the LLM service.

        Returns:
            The result of the function call.
        """
        return await self.function(params=params, **self._convert_arguments(args))


def tool_options(
    fn=None,
    *,
    cancel_on_interruption: bool = True,
    timeout_secs: float | None = None,
    cancellable_by_llm: bool = False,
):
    """Configure a handler's call options.

    This decorator is optional. A handler advertised in an ``LLMContext``'s
    tools — a direct function, or the ``handler`` a ``FunctionSchema`` carries —
    is registered automatically with default call options, so the decorator is
    only needed when you want to override those defaults. It attaches the options
    to the handler and returns it unchanged — it does not register anything, so
    decorated handlers can stay at module level.

    On an ``LLMWorker``, use ``@tool`` instead, which applies these same options
    and additionally marks the method for collection as one of the worker's tools.

    Can be used with or without arguments::

        @tool_options
        async def get_weather(params, location: str):
            ...

        @tool_options(cancel_on_interruption=False, timeout_secs=60)
        async def end_call(params, reason: str):
            ...

    Args:
        fn: The function to decorate (when used without arguments).
        cancel_on_interruption: Whether to cancel this function call when an
            interruption occurs. Defaults to True.
        timeout_secs: Optional per-tool timeout in seconds. A call that runs past
            it is cancelled. Defaults to None (uses the LLM service default).
        cancellable_by_llm: Whether the LLM may cancel this call once it is running,
            by calling the ``cancel_<name>`` tool advertised alongside it. Only
            meaningful together with ``cancel_on_interruption=False``: a
            synchronous call blocks the LLM until it returns, so there is no
            moment at which it could ask for the call to stop. What decides it is
            duration: the LLM reaches the cancel tool a few seconds in at best, so
            work that finishes before then can't be stopped whatever it costs.
            Every tool that opts in also adds a tool for the LLM to weigh.
            Defaults to False.

    Returns:
        The decorated function, unchanged except for pipecat call-option metadata
        attached under private ``_pipecat_*`` attributes (read internally; not
        part of the public API).
    """

    def decorator(fn):
        fn._pipecat_cancel_on_interruption = cancel_on_interruption
        fn._pipecat_timeout_secs = timeout_secs
        fn._pipecat_cancellable_by_llm = cancellable_by_llm
        return fn

    if fn is not None:
        return decorator(fn)
    return decorator

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
import inspect
import types
from collections.abc import Awaitable, Callable, Mapping
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import docstring_parser
from pydantic import BaseModel

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

    # TODO: maybe to better support things like enums, check if each type is a pydantic type and use its convert-to-jsonschema function
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

            # Add whether the parameter is required
            # If the parameter has no default value, it's required
            if param.default is inspect.Parameter.empty:
                required.append(name)

            # Add parameter description from docstring
            for doc_param in docstring_params:
                if doc_param.arg_name == name:
                    properties[name]["description"] = doc_param.description or ""

        return properties, required

    def _typehint_to_jsonschema(self, type_hint: Any) -> dict[str, Any]:
        """Convert a Python type hint to a JSON Schema.

        Args:
            type_hint: A Python type hint

        Returns:
            A dictionary representing the JSON Schema
        """
        if type_hint is None or type_hint is Any:
            return {}

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

        # Handle List and Set with specific item types
        if origin in (list, set) and args:
            return {"type": "array", "items": self._typehint_to_jsonschema(args[0])}

        # A tuple's items can differ from each other, and a fixed-length tuple
        # says so in the schema (JSON Schema 2020-12 prefixItems).
        if origin is tuple and args:
            if len(args) == 2 and args[1] is Ellipsis:  # tuple[int, ...]
                return {"type": "array", "items": self._typehint_to_jsonschema(args[0])}
            return {
                "type": "array",
                "prefixItems": [self._typehint_to_jsonschema(arg) for arg in args],
                "minItems": len(args),
                "maxItems": len(args),
            }

        # Handle Dict with specific key/value types
        if origin is dict and len(args) == 2:
            # For JSON Schema, keys must be strings
            return {"type": "object", "additionalProperties": self._typehint_to_jsonschema(args[1])}

        # Handle Pydantic models and dataclasses: their fields' defaults decide
        # what is required, and they carry their own (nested) field schemas.
        model_schema = self._model_to_jsonschema(type_hint)
        if model_schema is not None:
            return model_schema

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

    def _model_to_jsonschema(self, type_hint: Any) -> dict[str, Any] | None:
        """Build a JSON Schema for a Pydantic model or a dataclass.

        Both expose ``__annotations__``, which used to make them fall into the
        TypedDict branch. That marked every field required, even the ones with a
        default, so the model was told to send values it could leave out.

        Returns:
            The schema, or None if the type hint is neither of the two.
        """
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            # Pydantic already knows which fields are required, nested models
            # included; it also drops the keys we can't express in a tool schema.
            return type_hint.model_json_schema()

        if not dataclasses.is_dataclass(type_hint) or not isinstance(type_hint, type):
            return None

        properties = {}
        required = []
        # A field without a default still has to be sent. Anything else
        # (including default_factory) can be left out.
        hints = get_type_hints(type_hint)
        for field in dataclasses.fields(type_hint):
            properties[field.name] = self._typehint_to_jsonschema(hints.get(field.name))
            if (
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            ):
                required.append(field.name)

        schema = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    def _convert_arg(self, value: Any, type_hint: Any) -> Any:
        """Turn a value the LLM sent back into the type the function declares.

        The LLM only ever produces JSON, so a ``Unit`` parameter arrives as
        ``"celsius"`` and a ``Window`` one as a plain dict. Converting upfront
        means the function body can use its parameters as annotated.
        """
        if type_hint is None or type_hint is Any or value is None:
            return value

        origin = get_origin(type_hint)
        args = get_args(type_hint)

        if origin is Union or origin is types.UnionType:
            for arg in args:
                converted = self._convert_arg(value, arg)
                if converted is not value:
                    return converted
            return value

        if isinstance(type_hint, type) and issubclass(type_hint, Enum):
            try:
                return type_hint(value)
            except ValueError:
                return value

        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            return value if isinstance(value, type_hint) else type_hint.model_validate(value)

        if dataclasses.is_dataclass(type_hint) and isinstance(type_hint, type):
            if isinstance(value, type_hint):
                return value
            if isinstance(value, Mapping):
                hints = get_type_hints(type_hint)
                known = {f.name for f in dataclasses.fields(type_hint)}
                kwargs = {
                    key: self._convert_arg(item, hints.get(key))
                    for key, item in value.items()
                    if key in known
                }
                return type_hint(**kwargs)
            return value

        if origin is tuple and isinstance(value, (list, tuple)):
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(self._convert_arg(item, args[0]) for item in value)
            if len(args) == len(value):
                return tuple(self._convert_arg(item, hint) for item, hint in zip(value, args))
            return value

        if origin is list and isinstance(value, (list, tuple)):
            item_hint = args[0] if args else None
            return [self._convert_arg(item, item_hint) for item in value]

        if origin is dict and isinstance(value, Mapping):
            value_hint = args[1] if len(args) == 2 else None
            return {key: self._convert_arg(item, value_hint) for key, item in value.items()}

        return value


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
        hints = get_type_hints(self.function)
        converted = {
            name: self._convert_arg(value, hints.get(name)) for name, value in args.items()
        }
        return await self.function(params=params, **converted)


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

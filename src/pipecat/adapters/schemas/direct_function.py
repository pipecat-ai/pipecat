#
# Copyright (c) 2024–2025, Daily
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

import inspect
import types
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Protocol,
    Set,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import docstring_parser

from pipecat.adapters.schemas.function_schema import FunctionSchema

if TYPE_CHECKING:
    from pipecat.services.llm_service import FunctionCallParams


class DirectFunction(Protocol):
    """Protocol for a "direct" function that handles LLM function calls.

    "Direct" functions' metadata is automatically extracted from their function signature and
    docstrings, allowing them to be used without accompanying function configurations (as
    FunctionSchemas or in provider-specific formats).
    """

    async def __call__(self, params: "FunctionCallParams", **kwargs: Any) -> None:
        """Execute the direct function.

        Args:
            params: Function call parameters from the LLM service.
            **kwargs: Additional keyword arguments passed to the function.
        """
        ...


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
        docstring = docstring_parser.parse(inspect.getdoc(self.function))

        # Get function description
        self.description = (docstring.description or "").strip()

        # Get function parameters as JSON schemas, and the list of required parameters
        self.properties, self.required = self._get_parameters_as_jsonschema(
            self.function, docstring.params
        )

    # TODO: maybe to better support things like enums, check if each type is a pydantic type and use its convert-to-jsonschema function
    def _get_parameters_as_jsonschema(
        self, func: Callable, docstring_params: List[docstring_parser.DocstringParam]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Get function parameters as a dictionary of JSON schemas and a list of required parameters.

        Ignore the first parameter, as it's expected to be the "special" one.

        Args:
            func: Function to get parameters from.
            docstring_params: List of parameters extracted from the function's docstring.

        Returns:
            A tuple containing:

            - A dictionary mapping each function parameter to its JSON schema
            - A list of required parameter names
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

    def _typehint_to_jsonschema(self, type_hint: Any) -> Dict[str, Any]:
        """Convert a Python type hint to a JSON Schema.

        Args:
            type_hint: A Python type hint

        Returns:
            A dictionary representing the JSON Schema
        """
        if type_hint is None:
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
        elif type_hint is dict or type_hint is Dict:
            return {"type": "object"}
        elif type_hint is list or type_hint is List:
            return {"type": "array"}

        # Get origin and arguments for complex types
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        # Handle Optional/Union types
        if origin is Union or origin is types.UnionType:
            return {"anyOf": [self._typehint_to_jsonschema(arg) for arg in args]}

        # Handle List, Tuple, Set with specific item types
        if origin in (list, List, tuple, Tuple, set, Set) and args:
            return {"type": "array", "items": self._typehint_to_jsonschema(args[0])}

        # Handle Dict with specific key/value types
        if origin in (dict, Dict) and len(args) == 2:
            # For JSON Schema, keys must be strings
            return {"type": "object", "additionalProperties": self._typehint_to_jsonschema(args[1])}

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
        return await self.function(params=params, **args)

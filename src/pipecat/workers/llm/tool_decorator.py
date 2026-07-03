#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Decorator for marking methods as LLM tools."""

import warnings

from pipecat.adapters.schemas.direct_function import tool_options


def tool(fn=None, *, cancel_on_interruption=True, timeout_secs=None, timeout=None):
    """Mark a method as a tool.

    On ``LLMWorker`` subclasses, decorated methods are automatically
    registered with the LLM via ``register_direct_function`` and
    included in ``build_tools()``.

    This is the worker-flavored variant of ``@tool_options``: it attaches the
    same ``cancel_on_interruption`` / ``timeout_secs`` call options and additionally
    marks the method (with ``_pipecat_is_llm_tool``) so the worker collects it from the MRO.

    Can be used with or without arguments::

        @tool
        async def my_tool(self, params, arg: str):
            ...

        @tool(cancel_on_interruption=False, timeout_secs=60)
        async def my_tool(self, params, arg: str):
            ...

    Args:
        fn: The function to decorate (when used without arguments).
        cancel_on_interruption: Whether to cancel this tool call when
            an interruption occurs. Defaults to True. Only applies to
            ``LLMWorker`` tools.
        timeout_secs: Optional timeout in seconds for this tool call.
            Defaults to None (uses the LLM service default). Only applies
            to ``LLMWorker`` tools.
        timeout: Deprecated alias for ``timeout_secs``.

            .. deprecated:: 1.4.0
                Use ``timeout_secs`` instead. Will be removed in 2.0.0.
    """
    if timeout is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "The `timeout` argument to `@tool` is deprecated since 1.4.0, "
                "use `timeout_secs` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        # An explicit timeout_secs wins over the deprecated alias.
        if timeout_secs is None:
            timeout_secs = timeout

    def decorator(fn):
        fn = tool_options(cancel_on_interruption=cancel_on_interruption, timeout_secs=timeout_secs)(
            fn
        )
        fn._pipecat_is_llm_tool = True
        return fn

    if fn is not None:
        return decorator(fn)
    return decorator


def _collect_tools(obj) -> list:
    """Collect all ``@tool`` decorated bound methods from an object.

    Walks the MRO so that overridden methods in subclasses take
    precedence over base-class definitions.
    """
    seen: set[str] = set()
    tools = []
    for cls in type(obj).__mro__:
        for name, val in cls.__dict__.items():
            if name in seen:
                continue
            seen.add(name)
            if callable(val) and getattr(val, "_pipecat_is_llm_tool", False):
                tools.append(getattr(obj, name))
    return tools

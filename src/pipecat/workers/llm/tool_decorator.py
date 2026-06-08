#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Decorator for marking methods as LLM tools."""

from pipecat.adapters.schemas.direct_function import direct_function


def tool(fn=None, *, cancel_on_interruption=True, timeout=None):
    """Mark a method as a tool.

    On ``LLMWorker`` subclasses, decorated methods are automatically
    registered with the LLM via ``register_direct_function`` and
    included in ``build_tools()``.

    This is the worker-flavored variant of ``@direct_function``: it attaches the
    same ``cancel_on_interruption`` / ``timeout`` call options and additionally
    marks the method with ``is_llm_tool`` so the worker collects it from the MRO.

    Can be used with or without arguments::

        @tool
        async def my_tool(self, params, arg: str):
            ...

        @tool(cancel_on_interruption=False, timeout=60)
        async def my_tool(self, params, arg: str):
            ...

    Args:
        fn: The function to decorate (when used without arguments).
        cancel_on_interruption: Whether to cancel this tool call when
            an interruption occurs. Defaults to True. Only applies to
            ``LLMWorker`` tools.
        timeout: Optional timeout in seconds for this tool call.
            Defaults to None (uses the LLM service default).
    """

    def decorator(fn):
        fn = direct_function(cancel_on_interruption=cancel_on_interruption, timeout=timeout)(fn)
        fn.is_llm_tool = True
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
            if callable(val) and getattr(val, "is_llm_tool", False):
                tools.append(getattr(obj, name))
    return tools

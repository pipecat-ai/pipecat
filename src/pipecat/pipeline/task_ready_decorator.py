#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Decorator for marking agent methods as agent-ready handlers."""


def task_ready(*, name: str):
    """Mark a method as a handler for a specific agent becoming ready.

    Decorated methods are automatically collected by ``BaseTask`` at
    initialization. When ``on_ready`` fires, the agent calls
    ``watch_task`` for each decorated handler. When the watched agent
    registers, the decorated method is called with the ready data.

    Example::

        @task_ready(name="greeter")
        async def on_greeter_ready(self, data: TaskReadyData) -> None:
            await self.activate_task("greeter", args=...)

    Args:
        name: The name of the agent to watch.
    """

    def decorator(fn):
        fn.agent_ready_name = name
        return fn

    return decorator


def _collect_task_ready_handlers(obj) -> dict:
    """Collect all ``@task_ready`` decorated bound methods from an object.

    Returns a dict mapping agent name to the bound method.

    Raises:
        ValueError: If two handlers watch the same agent name.
    """
    seen: set[str] = set()
    handlers: dict[str, object] = {}
    for cls in type(obj).__mro__:
        for attr_name, val in cls.__dict__.items():
            if attr_name in seen:
                continue
            seen.add(attr_name)
            if callable(val) and hasattr(val, "agent_ready_name"):
                agent_name = val.agent_ready_name
                if agent_name in handlers:
                    existing = handlers[agent_name].__name__
                    raise ValueError(
                        f"Duplicate @task_ready handler for '{agent_name}': "
                        f"'{attr_name}' conflicts with '{existing}'"
                    )
                handlers[agent_name] = getattr(obj, attr_name)
    return handlers

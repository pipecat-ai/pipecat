#
# Copyright (c) 2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Decorator for marking worker methods as UI event handlers."""


def ui_event(name: str):
    """Mark a worker method as a handler for a named UI event.

    On ``UIWorker`` subclasses, decorated methods are automatically
    dispatched when a ``BusUIEventMessage`` with a matching ``name``
    arrives.

    Example::

        class MyUIWorker(UIWorker):
            @ui_event("nav_click")
            async def on_nav(self, message):
                view = message.payload.get("view")
                ...

    Args:
        name: The UI event name to match.
    """

    def decorator(fn):
        fn.is_ui_event_handler = True
        fn.ui_event_name = name
        return fn

    return decorator


def _collect_ui_event_handlers(obj) -> dict:
    """Collect all ``@ui_event`` decorated bound methods from an object.

    Walks the MRO so that overridden methods in subclasses take
    precedence over base-class definitions.

    Returns:
        A dict mapping event name to the bound method.

    Raises:
        ValueError: If two handlers share the same event name on the
            same subclass level.
    """
    seen: set[str] = set()
    handlers: dict[str, object] = {}
    source_names: dict[str, str] = {}  # event name -> defining method name, for errors
    for cls in type(obj).__mro__:
        for attr_name, val in cls.__dict__.items():
            if attr_name in seen:
                continue
            seen.add(attr_name)
            if callable(val) and getattr(val, "is_ui_event_handler", False):
                event_name = val.ui_event_name
                if event_name in handlers:
                    raise ValueError(
                        f"Duplicate @ui_event handler for '{event_name}': "
                        f"'{attr_name}' conflicts with '{source_names[event_name]}'"
                    )
                handlers[event_name] = getattr(obj, attr_name)
                source_names[event_name] = attr_name
    return handlers

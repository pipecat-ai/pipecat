class EventEmitter:
    def __init__(self):
        self._events = {}

    def on(self, event_name):
        """Decorator to register an event handler."""

        def decorator(func):
            if event_name not in self._events:
                self._events[event_name] = []
            self._events[event_name].append(func)
            return func

        return decorator

    async def emit(self, event_name, *args, **kwargs):
        """Trigger all handlers for a given event."""
        if event_name in self._events:
            for handler in self._events[event_name]:
                await handler(*args, **kwargs)

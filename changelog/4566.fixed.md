- Fixed tracing context access in service decorators to avoid crashes when
  `_tracing_context` is missing (`None`) while handling function calls.

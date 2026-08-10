- Added Agent Client Protocol (ACP) support in `pipecat.services.acp`. `ACPService`
  runs a coding agent as a subprocess and bridges it to a pipeline: it sends user
  turns as `session/prompt`, converts the agent's `session/update` stream into ACP
  frames, and hands agent-initiated callbacks (permission requests, filesystem and
  terminal methods) to any processor willing to answer them.

  The package also provides `ACPClient`, a Pipecat-independent JSON-RPC client for
  the protocol; `ACPUserAggregator`, which collects transcriptions into one prompt
  per user turn; and `ACPAutoPermission`, which auto-approves the agent's
  permission requests. `ACPService` offers `on_session_started`, `on_turn_started`,
  `on_turn_ended`, and `on_agent_exited` event handlers. See `examples/acp/`.

- Added `ACPLogObserver` (`pipecat.observers.loggers.acp_log_observer`), which logs
  an ACP agent's full stream (turns, messages, reasoning, tool calls, plans) without
  speaking any of it.

- Added client functions: tools the client executes rather than the server, for state that lives in the client (a game, a native app, a page). Register them with `LLMService.register_client_function()`; the service pushes a `ClientFunctionCallRequestFrame` for the transport layer to deliver, and settles the call when the client's `ClientFunctionCallResultFrame` arrives. Client functions time out, get cancelled by interruptions, and ignore late results like any other function call.

  ```python
  llm.register_client_function("build_building", timeout_secs=30)
  ```

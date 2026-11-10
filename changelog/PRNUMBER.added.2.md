- Added `pipecat.extensions.deepgram_voice_agent`, which serves the Deepgram Voice Agent API (`/v1/agent/converse`) from Pipecat pipelines, so clients written for Deepgram's Voice Agent work unchanged. `DeepgramVoiceAgentServer` builds a pipeline per connection from the client's `Settings` (Deepgram listen and speak, the `open_ai`, `anthropic` and `mock` think providers) and runs the client's functions on the client, or requests their `endpoint` from the server. See `examples/deepgram-voice-agent/server.py`.

  ```python
  server = DeepgramVoiceAgentServer(ProviderCredentials(deepgram_api_key=...))
  app.add_api_websocket_route("/v1/agent/converse", server.handle_websocket)
  ```

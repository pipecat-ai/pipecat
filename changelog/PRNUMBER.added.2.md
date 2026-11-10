- Added `pipecat.extensions.deepgram_voice_agent`, which serves the Deepgram Voice Agent API (`/v1/agent/converse`) from Pipecat pipelines, so clients written for Deepgram's Voice Agent work unchanged. `DeepgramVoiceAgentServer` builds a pipeline per connection from the client's `Settings`: Deepgram Flux or Nova listen; Deepgram Aura or Flux, Cartesia and ElevenLabs speak, with each further `agent.speak` entry a fallback should the one before it fail; the `open_ai`, `anthropic`, `google` and `mock` think providers. It runs the client's functions on the client, or requests their `endpoint` from the server, and handles the API's mid-session messages (`UpdateListen`, `UpdateSpeak`, `UpdateThink`, `UpdatePrompt`, `InjectAgentMessage`, `InjectUserMessage`, `ForceEndTurn`). `add_routes()` also serves the settings validation, models and `/metrics` endpoints. See `examples/deepgram-voice-agent/server.py`.

  ```python
  server = DeepgramVoiceAgentServer(ProviderCredentials(deepgram_api_key=...))
  server.add_routes(app)
  ```

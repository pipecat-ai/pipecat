- Added `GrokRealtimeLLMService` for xAI's Grok Voice Agent API with real-time voice conversations:

  - Support for real-time audio streaming with WebSocket connection
  - Built-in server-side VAD (Voice Activity Detection)
  - Multiple voice options: Ara, Rex, Sal, Eve, Leo
  - Built-in tools support: web_search, x_search, file_search
  - Custom function calling with standard Pipecat tools schema
  - Configurable audio formats (PCM at 8kHz-48kHz)

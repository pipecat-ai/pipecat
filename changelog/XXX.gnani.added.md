- Added Gnani Vachana speech AI services via the `pipecat-gnani` package:

  **STT (Speech-to-Text):**
  - `GnaniHttpSTTService`: REST-based transcription with VAD-segmented audio
  - `GnaniSTTService`: WebSocket streaming real-time transcription

  **TTS (Text-to-Speech):**
  - `GnaniHttpTTSService`: REST-based single-request synthesis
  - `GnaniSSETTSService`: SSE streaming synthesis (lower latency)
  - `GnaniTTSService`: WebSocket streaming synthesis with interruption handling

  **Features:**
  - Support for 12 Indian languages (Assamese, Bengali, English-India, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu)
  - 10 voices: Karan, Simran, Nara, Riya, Viraj, Raju, Pranav, Kaveri, Shubhra, Deepak
  - Dynamic language switching via `set_language()`
  - Built-in metrics (TTFB and processing time)
  - Traced transcription (`@traced_stt`) and synthesis (`@traced_tts`)
  - Complete foundational example at `examples/foundational/07x-interruptible-gnani.py`

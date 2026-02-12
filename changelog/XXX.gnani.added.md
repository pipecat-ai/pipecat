- Added `GnaniSTTService` for speech-to-text transcription using Gnani's multilingual AI-powered transcription API:

  - Support for 10+ Indian languages (Hindi, Tamil, Telugu, Kannada, Gujarati, Marathi, Bengali, Malayalam, Punjabi, and English-India)
  - REST API-based transcription with segmented audio processing
  - Extends `SegmentedSTTService` for VAD-based audio buffering
  - Dynamic language switching support with `set_language()` method
  - Built-in metrics support for TTFB and processing time tracking
  - Proper error handling with `ErrorFrame` integration
  - Traced transcription using `@traced_stt` decorator
  - Complete foundational example at `examples/foundational/07x-interruptible-gnani.py`


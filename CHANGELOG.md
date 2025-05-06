# Changelog

All notable changes to **Pipecat** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added support to `RimeHttpTTSService` for the `arcana` model.

### Fixed

- Fixed a `PipelineTask` issue that would cause tasks to not be cancelled if
  task was cancelled from outside of Pipecat.

- Fixed a `TaskManager` that was causing dangling tasks to be reported.

- Fixed an issue that could cause data to be sent to the transports when they
  were still not ready.

- Remove custom audio tracks from `DailyTransport` before leaving.

## [0.0.66] - 2025-05-02

### Added

- Added two new input parameters to `RimeTTSService`: `pause_between_brackets`
  and `phonemize_between_brackets`.

- Added support for cross-platform local smart turn detection. You can use
  `LocalSmartTurnAnalyzer` for on-device inference using Torch.

- `BaseOutputTransport` now allows multiple destinations if the transport
  implementation supports it (e.g. Daily's custom tracks). With multiple
  destinations it is possible to send different audio or video tracks with a
  single transport simultaneously. To do that, you need to set the new
  `Frame.transport_destination` field with your desired transport destination
  (e.g. custom track name), tell the transport you want a new destination with
  `TransportParams.audio_out_destinations` or
  `TransportParams.video_out_destinations` and the transport should take care of
  the rest.

- Similar to the new `Frame.transport_destination`, there's a new
  `Frame.transport_source` field which is set by the `BaseInputTransport` if the
  incoming data comes from a non-default source (e.g. custom tracks).

- `TTSService` has a new `transport_destination` constructor parameter. This
  parameter will be used to update the `Frame.transport_destination` field for
  each generated `TTSAudioRawFrame`. This allows sending multiple bots' audio to
  multiple destinations in the same pipeline.

- Added `DailyTransportParams.camera_out_enabled` and
  `DailyTransportParams.microphone_out_enabled` which allows you to
  enable/disable the main output camera or microphone tracks. This is useful if
  you only want to use custom tracks and not send the main tracks. Note that you
  still need `audio_out_enabled=True` or `video_out_enabled`.

- Added `DailyTransport.capture_participant_audio()` which allows you to capture
  an audio source (e.g. "microphone", "screenAudio" or a custom track name) from
  a remote participant.

- Added `DailyTransport.update_publishing()` which allows you to update the call
  video and audio publishing settings (e.g. audio and video quality).

- Added `RTVIObserverParams` which allows you to configure what RTVI messages
  are sent to the clients.

- Added a `context_window_compression` InputParam to
  `GeminiMultimodalLiveLLMService` which allows you to enable a sliding context
  window for the session as well as set the token limit of the sliding window.

- Updated `SmallWebRTCConnection` to support `ice_servers` with credentials.

- Added `VADUserStartedSpeakingFrame` and `VADUserStoppedSpeakingFrame`,
  indicating when the VAD detected the user to start and stop speaking. These
  events are helpful when using smart turn detection, as the user's stop time
  can differ from when their turn ends (signified by UserStoppedSpeakingFrame).

- Added `TranslationFrame`, a new frame type that contains a translated
  transcription.

- Added `TransportParams.audio_in_passthrough`. If set (the default), incoming
  audio will be pushed downstream.

- Added `MCPClient`; a way to connect to MCP servers and use the MCP servers'
  tools.

- Added `Mem0 OSS`, along with Mem0 cloud support now the OSS version is also
  available.

### Changed

- `TransportParams.audio_mixer` now supports a string and also a dictionary to
  provide a mixer per destination. For example:

```python
  audio_out_mixer={
      "track-1": SoundfileMixer(...),
      "track-2": SoundfileMixer(...),
      "track-N": SoundfileMixer(...),
  },
```

- The `STTMuteFilter` now mutes `InterimTranscriptionFrame` and
  `TranscriptionFrame` which allows the `STTMuteFilter` to be used in
  conjunction with transports that generate transcripts, e.g. `DailyTransport`.

- Function calls now receive a single parameter `FunctionCallParams` instead of
  `(function_name, tool_call_id, args, llm, context, result_callback)` which is
  now deprecated.

- Changed the user aggregator timeout for late transcriptions from 1.0s to 0.5s
  (`LLMUserAggregatorParams.aggregation_timeout`). Sometimes, the STT services
  might give us more than one transcription which could come after the user
  stopped speaking. We still want to include these additional transcriptions
  with the first one because it's part of the user turn. This is what this
  timeout is helpful with.

- Short utterances not detected by VAD while the bot is speaking are now
  ignored. This reduces the amount of bot interruptions significantly providing
  a more natural conversation experience.

- Updated `GladiaSTTService` to output a `TranslationFrame` when specifying a
  `translation` and `translation_config`.

- STT services now passthrough audio frames by default. This allows you to add
  audio recording without worrying about what's wrong in your pipeline when it
  doesn't work the first time.

- Input transports now always push audio downstream unless disabled with
  `TransportParams.audio_in_passthrough`. After many Pipecat releases, we
  realized this is the common use case. There are use cases where the input
  transport already provides STT and you also don't want recordings, in which
  case there's no need to push audio to the rest of the pipeline, but this is
  not a very common case.

- Added `RivaSegmentedSTTService`, which allows Riva offline/batch models, such
  as to be "canary-1b-asr" used in Pipecat.

### Deprecated

- Function calls with parameters
  `(function_name, tool_call_id, args, llm, context, result_callback)` are
  deprectated, use a single `FunctionCallParams` parameter instead.

- `TransportParams.camera_*` parameters are now deprecated, use
  `TransportParams.video_*` instead.

- `TransportParams.vad_enabled` parameter is now deprecated, use
  `TransportParams.audio_in_enabled` and `TransportParams.vad_analyzer` instead.

- `TransportParams.vad_audio_passthrough` parameter is now deprecated, use
  `TransportParams.audio_in_passthrough` instead.

- `ParakeetSTTService` is now deprecated, use `RivaSTTService` instead, which uses
  the model "parakeet-ctc-1.1b-asr" by default.

- `FastPitchTTSService` is now deprecated, use `RivaTTSService` instead, which uses
  the model "magpie-tts-multilingual" by default.

### Fixed

- Fixed an issue with `SimliVideoService` where the bot was continuously outputting
  audio, which prevents the `BotStoppedSpeakingFrame` from being emitted.

- Fixed an issue where `OpenAIRealtimeBetaLLMService` would add two assistant
  messages to the context.

- Fixed an issue with `GeminiMultimodalLiveLLMService` where the context
  contained tokens instead of words.

- Fixed an issue with HTTP Smart Turn handling, where the service returns a 500
  error. Previously, this would cause an unhandled exception. Now, a 500 error
  is treated as an incomplete response.

- Fixed a TTS services issue that could cause assistant output not to be
  aggregated to the context when also using `TTSSpeakFrame`s.

- Fixed an issue where the `SmartTurnMetricsData` was reporting 0ms for
  inference and processing time when using the `FalSmartTurnAnalyzer`.

### Other

- Added `examples/daily-custom-tracks` to show how to send and receive Daily
  custom tracks.

- Added `examples/daily-multi-translation` to showcase how to send multiple
  simulataneous translations with the same transport.

- Added 04 foundational examples for client/server transports. Also, renamed
  `29-livekit-audio-chat.py` to `04b-transports-livekit.py`.

- Added foundational example `13c-gladia-translation.py` showing how to use
  `TranscriptionFrame` and `TranslationFrame`.

## [0.0.65] - 2025-04-23 "Sant Jordi's release" 🌹📕

https://en.wikipedia.org/wiki/Saint_George%27s_Day_in_Catalonia

### Added

- Added automatic hangup logic to the Telnyx serializer. This feature hangs up
  the Telnyx call when an `EndFrame` or `CancelFrame` is received. It is
  enabled by default and is configurable via the `auto_hang_up` `InputParam`.

- Added a keepalive task to `GladiaSTTService` to prevent the websocket from
  disconnecting after 30 seconds of no audio input.

### Changed

- The `InputParams` for `ElevenLabsTTSService` and `ElevenLabsHttpTTSService`
  no longer require that `stability` and `similarity_boost` be set. You can
  individually set each param.

- In `TwilioFrameSerializer`, `call_sid` is Optional so as to avoid a breaking
  changed. `call_sid` is required to automatically hang up.

### Fixed

- Fixed an issue where `TwilioFrameSerializer` would send two hang up commands:
  one for the `EndFrame` and one for the `CancelFrame`.

## [0.0.64] - 2025-04-22

### Added

- Added automatic hangup logic to the Twilio serializer. This feature hangs up
  the Twilio call when an `EndFrame` or `CancelFrame` is received. It is
  enabled by default and is configurable via the `auto_hang_up` `InputParam`.

- Added `SmartTurnMetricsData`, which contains end-of-turn prediction metrics,
  to the `MetricsFrame`. Using `MetricsFrame`, you can now retrieve prediction
  confidence scores and processing time metrics from the smart turn analyzers.

- Added support for Application Default Credentials in Google services,
  `GoogleSTTService`, `GoogleTTSService`, and `GoogleVertexLLMService`.

- Added support for Smart Turn Detection via the `turn_analyzer` transport
  parameter. You can now choose between `HttpSmartTurnAnalyzer()` or
  `FalSmartTurnAnalyzer()` for remote inference or
  `LocalCoreMLSmartTurnAnalyzer()` for on-device inference using Core ML.

- `DeepgramTTSService` accepts `base_url` argument again, allowing you to
  connect to an on-prem service.

- Added `LLMUserAggregatorParams` and `LLMAssistantAggregatorParams` which allow
  you to control aggregator settings. You can now pass these arguments when
  creating aggregator pairs with `create_context_aggregator()`.

- Added `previous_text` context support to ElevenLabsHttpTTSService, improving
  speech consistency across sentences within an LLM response.

- Added word/timestamp pairs to `ElevenLabsHttpTTSService`.

- It is now possible to disable `SoundfileMixer` when created. You can then use
  `MixerEnableFrame` to dynamically enable it when necessary.

- Added `on_client_connected` and `on_client_disconnected` event handlers to
  the `DailyTransport` class. These handlers map to the same underlying Daily
  events as `on_participant_joined` and `on_participant_left`, respectively.
  This makes it easier to write a single bot pipeline that can also use other
  transports like `SmallWebRTCTransport` and `FastAPIWebsocketTransport`.

### Changed

- `GrokLLMService` now uses `grok-3-beta` as its default model.

- Daily's REST helpers now include an `eject_at_token_exp` param, which ejects
  the user when their token expires. This new parameter defaults to False.
  Also, the default value for `enable_prejoin_ui` changed to False and
  `eject_at_room_exp` changed to False.

- `OpenAILLMService` and `OpenPipeLLMService` now use `gpt-4.1` as their
  default model.

- `SoundfileMixer` constructor arguments need to be keywords.

### Deprecated

- `DeepgramSTTService` parameter `url` is now deprecated, use `base_url`
  instead.

### Removed

- Parameters `user_kwargs` and `assistant_kwargs` when creating a context
  aggregator pair using `create_context_aggregator()` have been removed. Use
  `user_params` and `assistant_params` instead.

### Fixed

- Fixed an issue that would cause TTS websocket-based services to not cleanup
  resources properly when disconnecting.

- Fixed a `TavusVideoService` issue that was causing audio choppiness.

- Fixed an issue in `SmallWebRTCTransport` where an error was thrown if the
  client did not create a video transceiver.

- Fixed an issue where LLM input parameters were not working and applied
  correctly in `GoogleVertexLLMService`, causing unexpected behavior during
  inference.

### Other

- Updated the `twilio-chatbot` example to use the auto-hangup feature.

## [0.0.63] - 2025-04-11

### Added

- Added media resolution control to `GeminiMultimodalLiveLLMService` with
  `GeminiMediaResolution` enum, allowing configuration of token usage for
  image processing (LOW: 64 tokens, MEDIUM: 256 tokens, HIGH: zoomed reframing
  with 256 tokens).

- Added Gemini's Voice Activity Detection (VAD) configuration to
  `GeminiMultimodalLiveLLMService` with `GeminiVADParams`, allowing fine
  control over speech detection sensitivity and timing, including:

  - Start sensitivity (how quickly speech is detected)
  - End sensitivity (how quickly turns end after pauses)
  - Prefix padding (milliseconds of audio to keep before speech is detected)
  - Silence duration (milliseconds of silence required to end a turn)

- Added comprehensive language support to `GeminiMultimodalLiveLLMService`,
  supporting over 30 languages via the `language` parameter, with proper
  mapping between Pipecat's `Language` enum and Gemini's language codes.

- Added support in `SmallWebRTCTransport` to detect when remote tracks are
  muted.

- Added support for image capture from a video stream to the
  `SmallWebRTCTransport`.

- Added a new iOS client option to the `SmallWebRTCTransport`
  **video-transform** example.

- Added new processors `ProducerProcessor` and `ConsumerProcessor`. The
  producer processor processes frames from the pipeline and decides whether the
  consumers should consume it or not. If so, the same frame that is received by
  the producer is sent to the consumer. There can be multiple consumers per
  producer. These processors can be useful to push frames from one part of a
  pipeline to a different one (e.g. when using `ParallelPipeline`).

- Improvements for the `SmallWebRTCTransport`:
  - Wait until the pipeline is ready before triggering the `connected` event.
  - Queue messages if the data channel is not ready.
  - Update the aiortc dependency to fix an issue where the 'video/rtx' MIME
    type was incorrectly handled as a codec retransmission.
  - Avoid initial video delays.

### Changed

- In `GeminiMultimodalLiveLLMService`, removed the `transcribe_model_audio`
  parameter in favor of Gemini Live's native output transcription support. Now
  text transcriptions are produced directly by the model. No configuration is
  required.

- Updated `GeminiMultimodalLiveLLMService`’s default `model` to
  `models/gemini-2.0-flash-live-001` and `base_url` to the `v1beta` websocket
  URL.

### Fixed

- Updated `daily-python` to 0.17.0 to fix an issue that was preventing to run on
  older platforms.

- Fixed an issue where `CartesiaTTSService`'s spell feature would result in
  the spelled word in the context appearing as "F,O,O,B,A,R" instead of
  "FOOBAR".

- Fixed an issue in the Azure TTS services where the language was being set
  incorrectly.

- Fixed `SmallWebRTCTransport` to support dynamic values for
  `TransportParams.audio_out_10ms_chunks`. Previously, it only worked with 20ms
  chunks.

- Fixed an issue with `GeminiMultimodalLiveLLMService` where the assistant
  context messages had no space between words.

- Fixed an issue where `LLMAssistantContextAggregator` would prevent a
  `BotStoppedSpeakingFrame` from moving through the pipeline.

## [0.0.62] - 2025-04-01 "An April Fools' release"

### Added

- Added `TransportParams.audio_out_10ms_chunks` parameter to allow controlling
  the amount of audio being sent by the output transport. It defaults to 4, so
  40ms audio chunks are sent.

- Added `QwenLLMService` for Qwen integration with an OpenAI-compatible
  interface. Added foundational example `14q-function-calling-qwen.py`.

- Added `Mem0MemoryService`. Mem0 is a self-improving memory layer for LLM
  applications. Learn more at: https://mem0.ai/.

- Added `WhisperSTTServiceMLX` for Whisper transcription on Apple Silicon.
  See example in `examples/foundational/13e-whisper-mlx.py`. Latency of
  completed transcription using Whisper large-v3-turbo on an M4 macbook is
  ~500ms.

- Added `SmallWebRTCTransport`, a new P2P WebRTC transport.

  - Created two examples in `p2p-webrtc`:
    - **video-transform**: Demonstrates sending and receiving audio/video with
      `SmallWebRTCTransport` using `TypeScript`. Includes video frame
      processing with OpenCV.
    - **voice-agent**: A minimal example of creating a voice agent with
      `SmallWebRTCTransport`.

- `GladiaSTTService` now have comprehensive support for the latest API config
  options, including model, language detection, preprocessing, custom
  vocabulary, custom spelling, translation, and message filtering options.

- Added `SmallWebRTCTransport`, a new P2P WebRTC transport.

  - Created two examples in `p2p-webrtc`:
    - **video-transform**: Demonstrates sending and receiving audio/video with
      `SmallWebRTCTransport` using `TypeScript`. Includes video frame
      processing with OpenCV.
    - **voice-agent**: A minimal example of creating a voice agent with
      `SmallWebRTCTransport`.

- Added support to `ProtobufFrameSerializer` to send the messages from
  `TransportMessageFrame` and `TransportMessageUrgentFrame`.

- Added support for a new TTS service, `PiperTTSService`.
  (see https://github.com/rhasspy/piper/)

- It is now possible to tell whether `UserStartedSpeakingFrame` or
  `UserStoppedSpeakingFrame` have been generated because of emulation frames.

### Changed

- `FunctionCallResultFrame`a are now system frames. This is to prevent function
  call results to be discarded during interruptions.

- Pipecat services have been reorganized into packages. Each package can have
  one or more of the following modules (in the future new module names might be
  needed) depending on the services implemented:

  - image: for image generation services
  - llm: for LLM services
  - memory: for memory services
  - stt: for Speech-To-Text services
  - tts: for Text-To-Speech services
  - video: for video generation services
  - vision: for video recognition services

- Base classes for AI services have been reorganized into modules. They can now
  be found in
  `pipecat.services.[ai_service,image_service,llm_service,stt_service,vision_service]`.

- `GladiaSTTService` now uses the `solaria-1` model by default. Other params
  use Gladia's default values. Added support for more language codes.

### Deprecated

- All Pipecat services imports have been deprecated and a warning will be shown
  when using the old import. The new import should be
  `pipecat.services.[service].[image,llm,memory,stt,tts,video,vision]`. For
  example, `from pipecat.services.openai.llm import OpenAILLMService`.

- Import for AI services base classes from `pipecat.services.ai_services` is now
  deprecated, use one of
  `pipecat.services.[ai_service,image_service,llm_service,stt_service,vision_service]`.

- Deprecated the `language` parameter in `GladiaSTTService.InputParams` in
  favor of `language_config`, which better aligns with Gladia's API.

- Deprecated using `GladiaSTTService.InputParams` directly. Use the new
  `GladiaInputParams` class instead.

### Fixed

- Fixed a `FastAPIWebsocketTransport` and `WebsocketClientTransport` issue that
  would cause the transport to be closed prematurely, preventing the internally
  queued audio to be sent. The same issue could also cause an infinite loop
  while using an output mixer and when sending an `EndFrame`, preventing the bot
  to finish.

- Fixed an issue that could cause the `TranscriptionUpdateFrame` being pushed
  because of an interruption to be discarded.

- Fixed an issue that would cause `SegmentedSTTService` based services
  (e.g. `OpenAISTTService`) to try to transcribe non-spoken audio, causing
  invalid transcriptions.

- Fixed an issue where `GoogleTTSService` was emitting two `TTSStoppedFrames`.

### Performance

- Output transports now send 40ms audio chunks instead of 20ms. This should
  improve performance.

- `BotSpeakingFrame`s are now sent every 200ms. If the output transport audio chunks
  are higher than 200ms then they will be sent at every audio chunk.

### Other

- Added foundational example `37-mem0.py` demonstrating how to use the
  `Mem0MemoryService`.

- Added foundational example `13e-whisper-mlx.py` demonstrating how to use the
  `WhisperSTTServiceMLX`.

## [0.0.61] - 2025-03-26

### Added

- Added a new frame, `LLMSetToolChoiceFrame`, which provides a mechanism
  for modifying the `tool_choice` in the context.

- Added `GroqTTSService` which provides text-to-speech functionality using
  Groq's API.

- Added support in `DailyTransport` for updating remote participants'
  `canReceive` permission via the `update_remote_participants()` method, by
  bumping the daily-python dependency to >= 0.16.0.

- ElevenLabs TTS services now support a sample rate of 8000.

- Added support for `instructions` in `OpenAITTSService`.

- Added support for `base_url` in `OpenAIImageGenService` and
  `OpenAITTSService`.

### Fixed

- Fixed an issue in `RTVIObserver` that prevented handling of Google LLM
  context messages. The observer now processes both OpenAI-style and
  Google-style contexts.

- Fixed an issue in Daily involving switching virtual devices, by bumping the
  daily-python dependency to >= 0.16.1.

- Fixed a `GoogleAssistantContextAggregator` issue where function calls
  placeholders where not being updated when then function call result was
  different from a string.

- Fixed an issue that would cause `LLMAssistantContextAggregator` to block
  processing more frames while processing a function call result.

- Fixed an issue where the `RTVIObserver` would report two bot started and
  stopped speaking events for each bot turn.

- Fixed an issue in `UltravoxSTTService` that caused improper audio processing
  and incorrect LLM frame output.

### Other

- Added `examples/foundational/07x-interruptible-local.py` to show how a local
  transport can be used.

## [0.0.60] - 2025-03-20

### Added

- Added `default_headers` parameter to `BaseOpenAILLMService` constructor.

### Changed

- Rollback to `deepgram-sdk` 3.8.0 since 3.10.1 was causing connections issues.

- Changed the default `InputAudioTranscription` model to `gpt-4o-transcribe`
  for `OpenAIRealtimeBetaLLMService`.

### Other

- Update the `19-openai-realtime-beta.py` and `19a-azure-realtime-beta.py`
  examples to use the FunctionSchema format.

## [0.0.59] - 2025-03-20

### Added

- When registering a function call it is now possible to indicate if you want
  the function call to be cancelled if there's a user interruption via
  `cancel_on_interruption` (defaults to False). This is now possible because
  function calls are executed concurrently.

- Added support for detecting idle pipelines. By default, if no activity has
  been detected during 5 minutes, the `PipelineTask` will be automatically
  cancelled. It is possible to override this behavior by passing
  `cancel_on_idle_timeout=False`. It is also possible to change the default
  timeout with `idle_timeout_secs` or the frames that prevent the pipeline from
  being idle with `idle_timeout_frames`. Finally, an `on_idle_timeout` event
  handler will be triggered if the idle timeout is reached (whether the pipeline
  task is cancelled or not).

- Added `FalSTTService`, which provides STT for Fal's Wizper API.

- Added a `reconnect_on_error` parameter to websocket-based TTS services as well
  as a `on_connection_error` event handler. The `reconnect_on_error` indicates
  whether the TTS service should reconnect on error. The `on_connection_error`
  will always get called if there's any error no matter the value of
  `reconnect_on_error`. This allows, for example, to fallback to a different TTS
  provider if something goes wrong with the current one.

- Added new `SkipTagsAggregator` that extends `BaseTextAggregator` to aggregate
  text and skips end of sentence matching if aggregated text is between
  start/end tags.

- Added new `PatternPairAggregator` that extends `BaseTextAggregator` to
  identify content between matching pattern pairs in streamed text. This allows
  for detection and processing of structured content like XML-style tags that
  may span across multiple text chunks or sentence boundaries.

- Added new `BaseTextAggregator`. Text aggregators are used by the TTS service
  to aggregate LLM tokens and decide when the aggregated text should be pushed
  to the TTS service. They also allow for the text to be manipulated while it's
  being aggregated. A text aggregator can be passed via `text_aggregator` to the
  TTS service.

- Added new `sample_rate` constructor parameter to `TavusVideoService` to allow
  changing the output sample rate.

- Added new `NeuphonicTTSService`.
  (see https://neuphonic.com)

- Added new `UltravoxSTTService`.
  (see https://github.com/fixie-ai/ultravox)

- Added `on_frame_reached_upstream` and `on_frame_reached_downstream` event
  handlers to `PipelineTask`. Those events will be called when a frame reaches
  the beginning or end of the pipeline respectively. Note that by default, the
  event handlers will not be called unless a filter is set with
  `PipelineTask.set_reached_upstream_filter()` or
  `PipelineTask.set_reached_downstream_filter()`.

- Added support for Chirp voices in `GoogleTTSService`.

- Added a `flush_audio()` method to `FishTTSService` and `LmntTTSService`.

- Added a `set_language` convenience method for `GoogleSTTService`, allowing
  you to set a single language. This is in addition to the `set_languages`
  method which allows you to set a list of languages.

- Added `on_user_turn_audio_data` and `on_bot_turn_audio_data` to
  `AudioBufferProcessor`. This gives the ability to grab the audio of only that
  turn for both the user and the bot.

- Added new base class `BaseObject` which is now the base class of
  `FrameProcessor`, `PipelineRunner`, `PipelineTask` and `BaseTransport`. The
  new `BaseObject` adds supports for event handlers.

- Added support for a unified format for specifying function calling across all
  LLM services.

```python
  weather_function = FunctionSchema(
      name="get_current_weather",
      description="Get the current weather",
      properties={
          "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA",
          },
          "format": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"],
              "description": "The temperature unit to use. Infer this from the user's location.",
          },
      },
      required=["location"],
  )
  tools = ToolsSchema(standard_tools=[weather_function])
```

- Added `speech_threshold` parameter to `GladiaSTTService`.

- Allow passing user (`user_kwargs`) and assistant (`assistant_kwargs`) context
  aggregator parameters when using `create_context_aggregator()`. The values are
  passed as a mapping that will then be converted to arguments.

- Added `speed` as an `InputParam` for both `ElevenLabsTTSService` and
  `ElevenLabsHttpTTSService`.

- Added new `LLMFullResponseAggregator` to aggregate full LLM completions. At
  every completion the `on_completion` event handler is triggered.

- Added a new frame, `RTVIServerMessageFrame`, and RTVI message
  `RTVIServerMessage` which provides a generic mechanism for sending custom
  messages from server to client. The `RTVIServerMessageFrame` is processed by
  the `RTVIObserver` and will be delivered to the client's `onServerMessage`
  callback or `ServerMessage` event.

- Added `GoogleLLMOpenAIBetaService` for Google LLM integration with an
  OpenAI-compatible interface. Added foundational example
  `14o-function-calling-gemini-openai-format.py`.

- Added `AzureRealtimeBetaLLMService` to support Azure's OpeanAI Realtime API. Added
  foundational example `19a-azure-realtime-beta.py`.

- Introduced `GoogleVertexLLMService`, a new class for integrating with Vertex AI
  Gemini models. Added foundational example
  `14p-function-calling-gemini-vertex-ai.py`.

- Added support in `OpenAIRealtimeBetaLLMService` for a slate of new features:

  - The `'gpt-4o-transcribe'` input audio transcription model, along
    with new `language` and `prompt` options specific to that model.
  - The `input_audio_noise_reduction` session property.

    ```python
    session_properties = SessionProperties(
      # ...
      input_audio_noise_reduction=InputAudioNoiseReduction(
        type="near_field" # also supported: "far_field"
      )
      # ...
    )
    ```

  - The `'semantic_vad'` `turn_detection` session property value, a more
    sophisticated model for detecting when the user has stopped speaking.
  - `on_conversation_item_created` and `on_conversation_item_updated`
    events to `OpenAIRealtimeBetaLLMService`.

    ```python
    @llm.event_handler("on_conversation_item_created")
    async def on_conversation_item_created(llm, item_id, item):
      # ...

    @llm.event_handler("on_conversation_item_updated")
    async def on_conversation_item_updated(llm, item_id, item):
      # `item` may not always be available here
      # ...
    ```

  - The `retrieve_conversation_item(item_id)` method for introspecting a
    conversation item on the server.

    ```python
    item = await llm.retrieve_conversation_item(item_id)
    ```

### Changed

- Updated `OpenAISTTService` to use `gpt-4o-transcribe` as the default
  transcription model.

- Updated `OpenAITTSService` to use `gpt-4o-mini-tts` as the default TTS model.

- Function calls are now executed in tasks. This means that the pipeline will
  not be blocked while the function call is being executed.

- ⚠️ `PipelineTask` will now be automatically cancelled if no bot activity is
  happening in the pipeline. There are a few settings to configure this
  behavior, see `PipelineTask` documentation for more details.

- All event handlers are now executed in separate tasks in order to prevent
  blocking the pipeline. It is possible that event handlers take some time to
  execute in which case the pipeline would be blocked waiting for the event
  handler to complete.

- Updated `TranscriptProcessor` to support text output from
  `OpenAIRealtimeBetaLLMService`.

- `OpenAIRealtimeBetaLLMService` and `GeminiMultimodalLiveLLMService` now push
  a `TTSTextFrame`.

- Updated the default mode for `CartesiaTTSService` and
  `CartesiaHttpTTSService` to `sonic-2`.

### Deprecated

- Passing a `start_callback` to `LLMService.register_function()` is now
  deprecated, simply move the code from the start callback to the function call.

- `TTSService` parameter `text_filter` is now deprecated, use `text_filters`
  instead which is now a list. This allows passing multiple filters that will be
  executed in order.

### Removed

- Removed deprecated `audio.resample_audio()`, use `create_default_resampler()`
  instead.

- Removed deprecated`stt_service` parameter from `STTMuteFilter`.

- Removed deprecated RTVI processors, use an `RTVIObserver` instead.

- Removed deprecated `AWSTTSService`, use `PollyTTSService` instead.

- Removed deprecated field `tier` from `DailyTranscriptionSettings`, use `model`
  instead.

- Removed deprecated `pipecat.vad` package, use `pipecat.audio.vad` instead.

### Fixed

- Fixed an assistant aggregator issue that could cause assistant text to be
  split into multiple chunks during function calls.

- Fixed an assistant aggregator issue that was causing assistant text to not be
  added to the context during function calls. This could lead to duplications.

- Fixed a `SegmentedSTTService` issue that was causing audio to be sent
  prematurely to the STT service. Instead of analyzing the volume in this
  service we rely on VAD events which use both VAD and volume.

- Fixed a `GeminiMultimodalLiveLLMService` issue that was causing messages to be
  duplicated in the context when pushing `LLMMessagesAppendFrame` frames.

- Fixed an issue with `SegmentedSTTService` based services
  (e.g. `GroqSTTService`) that was not allow audio to pass-through downstream.

- Fixed a `CartesiaTTSService` and `RimeTTSService` issue that would consider
  text between spelling out tags end of sentence.

- Fixed a `match_endofsentence` issue that would result in floating point
  numbers to be considered an end of sentence.

- Fixed a `match_endofsentence` issue that would result in emails to be
  considered an end of sentence.

- Fixed an issue where the RTVI message `disconnect-bot` was pushing an
  `EndFrame`, resulting in the pipeline not shutting down. It now pushes an
  `EndTaskFrame` upstream to shutdown the pipeline.

- Fixed an issue with the `GoogleSTTService` where stream timeouts during
  periods of inactivity were causing connection failures. The service now
  properly detects timeout errors and handles reconnection gracefully,
  ensuring continuous operation even after periods of silence or when using an
  `STTMuteFilter`.

- Fixed an issue in `RimeTTSService` where the last line of text sent didn't
  result in an audio output being generated.

- Fixed `OpenAIRealtimeBetaLLMService` by adding proper handling for:
  - The `conversation.item.input_audio_transcription.delta` server message,
    which was added server-side at some point and not handled client-side.
  - Errors reported by the `response.done` server message.

### Other

- Add foundational example `07w-interruptible-fal.py`, showing `FalSTTService`.

- Added a new Ultravox example
  `examples/foundational/07u-interruptible-ultravox.py`.

- Added new Neuphonic examples
  `examples/foundational/07v-interruptible-neuphonic.py` and
  `examples/foundational/07v-interruptible-neuphonic-http.py`.

- Added a new example `examples/foundational/36-user-email-gathering.py` to show
  how to gather user emails. The example uses's Cartesia's `<spell></spell>`
  tags and Rime `spell()` function to spell out the emails for confirmation.

- Update the `34-audio-recording.py` example to include an STT processor.

- Added foundational example `35-voice-switching.py` showing how to use the new
  `PatternPairAggregator`. This example shows how to encode information for the
  LLM to instruct TTS voice changes, but this can be used to encode any
  information into the LLM response, which you want to parse and use in other
  parts of your application.

- Added a Pipecat Cloud deployment example to the `examples` directory.

- Removed foundational examples 28b and 28c as the TranscriptProcessor no
  longer has an LLM depedency. Renamed foundational example 28a to
  `28-transcript-processor.py`.

## [0.0.58] - 2025-02-26

### Added

- Added track-specific audio event `on_track_audio_data` to
  `AudioBufferProcessor` for accessing separate input and output audio tracks.

- Pipecat version will now be logged on every application startup. This will
  help us identify what version we are running in case of any issues.

- Added a new `StopFrame` which can be used to stop a pipeline task while
  keeping the frame processors running. The frame processors could then be used
  in a different pipeline. The difference between a `StopFrame` and a
  `StopTaskFrame` is that, as with `EndFrame` and `EndTaskFrame`, the
  `StopFrame` is pushed from the task and the `StopTaskFrame` is pushed upstream
  inside the pipeline by any processor.

- Added a new `PipelineTask` parameter `observers` that replaces the previous
  `PipelineParams.observers`.

- Added a new `PipelineTask` parameter `check_dangling_tasks` to enable or
  disable checking for frame processors' dangling tasks when the Pipeline
  finishes running.

- Added new `on_completion_timeout` event for LLM services (all OpenAI-based
  services, Anthropic and Google). Note that this event will only get triggered
  if LLM timeouts are setup and if the timeout was reached. It can be useful to
  retrigger another completion and see if the timeout was just a blip.

- Added new log observers `LLMLogObserver` and `TranscriptionLogObserver` that
  can be useful for debugging your pipelines.

- Added `room_url` property to `DailyTransport`.

- Added `addons` argument to `DeepgramSTTService`.

- Added `exponential_backoff_time()` to `utils.network` module.

### Changed

- ⚠️ `PipelineTask` now requires keyword arguments (except for the first one for
  the pipeline).

- Updated `PlayHTHttpTTSService` to take a `voice_engine` and `protocol` input
  in the constructor. The previous method of providing a `voice_engine` input
  that contains the engine and protocol is deprecated by PlayHT.

- The base `TTSService` class now strips leading newlines before sending text
  to the TTS provider. This change is to solve issues where some TTS providers,
  like Azure, would not output text due to newlines.

- `GrokLLMSService` now uses `grok-2` as the default model.

- `AnthropicLLMService` now uses `claude-3-7-sonnet-20250219` as the default
  model.

- `RimeHttpTTSService` needs an `aiohttp.ClientSession` to be passed to the
  constructor as all the other HTTP-based services.

- `RimeHttpTTSService` doesn't use a default voice anymore.

- `DeepgramSTTService` now uses the new `nova-3` model by default. If you want
  to use the previous model you can pass `LiveOptions(model="nova-2-general")`.
  (see https://deepgram.com/learn/introducing-nova-3-speech-to-text-api)

```python
stt = DeepgramSTTService(..., live_options=LiveOptions(model="nova-2-general"))
```

### Deprecated

- `PipelineParams.observers` is now deprecated, you the new `PipelineTask`
  parameter `observers`.

### Removed

- Remove `TransportParams.audio_out_is_live` since it was not being used at all.

### Fixed

- Fixed an issue that would cause undesired interruptions via
  `EmulateUserStartedSpeakingFrame`.

- Fixed a `GoogleLLMService` that was causing an exception when sending inline
  audio in some cases.

- Fixed an `AudioContextWordTTSService` issue that would cause an `EndFrame` to
  disconnect from the TTS service before audio from all the contexts was
  received. This affected services like Cartesia and Rime.

- Fixed an issue that was not allowing to pass an `OpenAILLMContext` to create
  `GoogleLLMService`'s context aggregators.

- Fixed a `ElevenLabsTTSService`, `FishAudioTTSService`, `LMNTTTSService` and
  `PlayHTTTSService` issue that was resulting in audio requested before an
  interruption being played after an interruption.

- Fixed `match_endofsentence` support for ellipses.

- Fixed an issue where `EndTaskFrame` was not triggering
  `on_client_disconnected` or closing the WebSocket in FastAPI.

- Fixed an issue in `DeepgramSTTService` where the `sample_rate` passed to the
  `LiveOptions` was not being used, causing the service to use the default
  sample rate of pipeline.

- Fixed a context aggregator issue that would not append the LLM text response
  to the context if a function call happened in the same LLM turn.

- Fixed an issue that was causing HTTP TTS services to push `TTSStoppedFrame`
  more than once.

- Fixed a `FishAudioTTSService` issue where `TTSStoppedFrame` was not being
  pushed.

- Fixed an issue that `start_callback` was not invoked for some LLM services.

- Fixed an issue that would cause `DeepgramSTTService` to stop working after an
  error occurred (e.g. sudden network loss). If the network recovered we would
  not reconnect.

- Fixed a `STTMuteFilter` issue that would not mute user audio frames causing
  transcriptions to be generated by the STT service.

### Other

- Added Gemini support to `examples/phone-chatbot`.

- Added foundational example `34-audio-recording.py` showing how to use the
  AudioBufferProcessor callbacks to save merged and track recordings.

## [0.0.57] - 2025-02-14

### Added

- Added new `AudioContextWordTTSService`. This is a TTS base class for TTS
  services that handling multiple separate audio requests.

- Added new frames `EmulateUserStartedSpeakingFrame` and
  `EmulateUserStoppedSpeakingFrame` which can be used to emulated VAD behavior
  without VAD being present or not being triggered.

- Added a new `audio_in_stream_on_start` field to `TransportParams`.

- Added a new method `start_audio_in_streaming` in the `BaseInputTransport`.

  - This method should be used to start receiving the input audio in case the
    field `audio_in_stream_on_start` is set to `false`.

- Added support for the `RTVIProcessor` to handle buffered audio in `base64`
  format, converting it into InputAudioRawFrame for transport.

- Added support for the `RTVIProcessor` to trigger `start_audio_in_streaming`
  only after the `client-ready` message.

- Added new `MUTE_UNTIL_FIRST_BOT_COMPLETE` strategy to `STTMuteStrategy`. This
  strategy starts muted and remains muted until the first bot speech completes,
  ensuring the bot's first response cannot be interrupted. This complements the
  existing `FIRST_SPEECH` strategy which only mutes during the first detected
  bot speech.

- Added support for Google Cloud Speech-to-Text V2 through `GoogleSTTService`.

- Added `RimeTTSService`, a new `WordTTSService`. Updated the foundational
  example `07q-interruptible-rime.py` to use `RimeTTSService`.

- Added support for Groq's Whisper API through the new `GroqSTTService` and
  OpenAI's Whisper API through the new `OpenAISTTService`. Introduced a new
  base class `BaseWhisperSTTService` to handle common Whisper API
  functionality.

- Added `PerplexityLLMService` for Perplexity NIM API integration, with an
  OpenAI-compatible interface. Also, added foundational example
  `14n-function-calling-perplexity.py`.

- Added `DailyTransport.update_remote_participants()`. This allows you to update
  remote participant's settings, like their permissions or which of their
  devices are enabled. Requires that the local participant have participant
  admin permission.

### Changed

- We don't consider a colon `:` and end of sentence any more.

- Updated `DailyTransport` to respect the `audio_in_stream_on_start` field,
  ensuring it only starts receiving the audio input if it is enabled.

- Updated `FastAPIWebsocketOutputTransport` to send `TransportMessageFrame` and
  `TransportMessageUrgentFrame` to the serializer.

- Updated `WebsocketServerOutputTransport` to send `TransportMessageFrame` and
  `TransportMessageUrgentFrame` to the serializer.

- Enhanced `STTMuteConfig` to validate strategy combinations, preventing
  `MUTE_UNTIL_FIRST_BOT_COMPLETE` and `FIRST_SPEECH` from being used together
  as they handle first bot speech differently.

- Updated foundational example `07n-interruptible-google.py` to use all Google
  services.

- `RimeHttpTTSService` now uses the `mistv2` model by default.

- Improved error handling in `AzureTTSService` to properly detect and log
  synthesis cancellation errors.

- Enhanced `WhisperSTTService` with full language support and improved model
  documentation.

- Updated foundation example `14f-function-calling-groq.py` to use
  `GroqSTTService` for transcription.

- Updated `GroqLLMService` to use `llama-3.3-70b-versatile` as the default
  model.

- `RTVIObserver` doesn't handle `LLMSearchResponseFrame` frames anymore. For
  now, to handle those frames you need to create a `GoogleRTVIObserver`
  instead.

### Deprecated

- `STTMuteFilter` constructor's `stt_service` parameter is now deprecated and
  will be removed in a future version. The filter now manages mute state
  internally instead of querying the STT service.

- `RTVI.observer()` is now deprecated, instantiate an `RTVIObserver` directly
  instead.

- All RTVI frame processors (e.g. `RTVISpeakingProcessor`,
  `RTVIBotLLMProcessor`) are now deprecated, instantiate an `RTVIObserver`
  instead.

### Fixed

- Fixed a `FalImageGenService` issue that was causing the event loop to be
  blocked while loading the downloadded image.

- Fixed a `CartesiaTTSService` service issue that would cause audio overlapping
  in some cases.

- Fixed a websocket-based service issue (e.g. `CartesiaTTSService`) that was
  preventing a reconnection after the server disconnected cleanly, which was
  causing an inifite loop instead.

- Fixed a `BaseOutputTransport` issue that was causing upstream frames to no be
  pushed upstream.

- Fixed multiple issue where user transcriptions where not being handled
  properly. It was possible for short utterances to not trigger VAD which would
  cause user transcriptions to be ignored. It was also possible for one or more
  transcriptions to be generated after VAD in which case they would also be
  ignored.

- Fixed an issue that was causing `BotStoppedSpeakingFrame` to be generated too
  late. This could then cause issues unblocking `STTMuteFilter` later than
  desired.

- Fixed an issue that was causing `AudioBufferProcessor` to not record
  synchronized audio.

- Fixed an `RTVI` issue that was causing `bot-tts-text` messages to be sent
  before being processed by the output transport.

- Fixed an issue[#1192] in 11labs where we are trying to reconnect/disconnect
  the websocket connection even when the connection is already closed.

- Fixed an issue where `has_regular_messages` condition was always true in
  `GoogleLLMContext` due to `Part` having `function_call` & `function_response`
  with `None` values.

### Other

- Added new `instant-voice` example. This example showcases how to enable
  instant voice communication as soon as a user connects.

- Added new `local-input-select-stt` example. This examples allows you to play
  with local audio inputs by slecting them through a nice text interface.

## [0.0.56] - 2025-02-06

### Changed

- Use `gemini-2.0-flash-001` as the default model for `GoogleLLMSerivce`.

- Improved foundational examples 22b, 22c, and 22d to support function calling.
  With these base examples, `FunctionCallInProgressFrame` and
  `FunctionCallResultFrame` will no longer be blocked by the gates.

### Fixed

- Fixed a `TkLocalTransport` and `LocalAudioTransport` issues that was causing
  errors on cleanup.

- Fixed an issue that was causing `tests.utils` import to fail because of
  logging setup.

- Fixed a `SentryMetrics` issue that was preventing any metrics to be sent to
  Sentry and also was preventing from metrics frames to be pushed to the
  pipeline.

- Fixed an issue in `BaseOutputTransport` where incoming audio would not be
  resampled to the desired output sample rate.

- Fixed an issue with the `TwilioFrameSerializer` and `TelnyxFrameSerializer`
  where `twilio_sample_rate` and `telnyx_sample_rate` were incorrectly
  initialized to `audio_in_sample_rate`. Those values currently default to 8000
  and should be set manually from the serializer constructor if a different
  value is needed.

### Other

- Added a new `sentry-metrics` example.

## [0.0.55] - 2025-02-05

### Added

- Added a new `start_metadata` field to `PipelineParams`. The provided metadata
  will be set to the initial `StartFrame` being pushed from the `PipelineTask`.

- Added new fields to `PipelineParams` to control audio input and output sample
  rates for the whole pipeline. This allows controlling sample rates from a
  single place instead of having to specify sample rates in each
  service. Setting a sample rate to a service is still possible and will
  override the value from `PipelineParams`.

- Introduce audio resamplers (`BaseAudioResampler`). This is just a base class
  to implement audio resamplers. Currently, two implementations are provided
  `SOXRAudioResampler` and `ResampyResampler`. A new
  `create_default_resampler()` has been added (replacing the now deprecated
  `resample_audio()`).

- It is now possible to specify the asyncio event loop that a `PipelineTask` and
  all the processors should run on by passing it as a new argument to the
  `PipelineRunner`. This could allow running pipelines in multiple threads each
  one with its own event loop.

- Added a new `utils.TaskManager`. Instead of a global task manager we now have
  a task manager per `PipelineTask`. In the previous version the task manager
  was global, so running multiple simultaneous `PipelineTask`s could result in
  dangling task warnings which were not actually true. In order, for all the
  processors to know about the task manager, we pass it through the
  `StartFrame`. This means that processors should create tasks when they receive
  a `StartFrame` but not before (because they don't have a task manager yet).

- Added `TelnyxFrameSerializer` to support Telnyx calls. A full running example
  has also been added to `examples/telnyx-chatbot`.

- Allow pushing silence audio frames before `TTSStoppedFrame`. This might be
  useful for testing purposes, for example, passing bot audio to an STT service
  which usually needs additional audio data to detect the utterance stopped.

- `TwilioSerializer` now supports transport message frames. With this we can
  create Twilio emulators.

- Added a new transport: `WebsocketClientTransport`.

- Added a `metadata` field to `Frame` which makes it possible to pass custom
  data to all frames.

- Added `test/utils.py` inside of pipecat package.

### Changed

- `GatedOpenAILLMContextAggregator` now require keyword arguments. Also, a new
  `start_open` argument has been added to set the initial state of the gate.

- Added `organization` and `project` level authentication to
  `OpenAILLMService`.

- Improved the language checking logic in `ElevenLabsTTSService` and
  `ElevenLabsHttpTTSService` to properly handle language codes based on model
  compatibility, with appropriate warnings when language codes cannot be
  applied.

- Updated `GoogleLLMContext` to support pushing `LLMMessagesUpdateFrame`s that
  contain a combination of function calls, function call responses, system
  messages, or just messages.

- `InputDTMFFrame` is now based on `DTMFFrame`. There's also a new
  `OutputDTMFFrame` frame.

### Deprecated

- `resample_audio()` is now deprecated, use `create_default_resampler()`
  instead.

### Removed

- `AudioBufferProcessor.reset_audio_buffers()` has been removed, use
  `AudioBufferProcessor.start_recording()` and
  `AudioBufferProcessor.stop_recording()` instead.

### Fixed

- Fixed a `AudioBufferProcessor` that would cause crackling in some recordings.

- Fixed an issue in `AudioBufferProcessor` where user callback would not be
  called on task cancellation.

- Fixed an issue in `AudioBufferProcessor` that would cause wrong silence
  padding in some cases.

- Fixed an issue where `ElevenLabsTTSService` messages would return a 1009
  websocket error by increasing the max message size limit to 16MB.

- Fixed a `DailyTransport` issue that would cause events to be triggered before
  join finished.

- Fixed a `PipelineTask` issue that was preventing processors to be cleaned up
  after cancelling the task.

- Fixed an issue where queuing a `CancelFrame` to a pipeline task would not
  cause the task to finish. However, using `PipelineTask.cancel()` is still the
  recommended way to cancel a task.

### Other

- Improved Unit Test `run_test()` to use `PipelineTask` and
  `PipelineRunner`. There's now also some control around `StartFrame` and
  `EndFrame`. The `EndTaskFrame` has been removed since it doesn't seem
  necessary with this new approach.

- Updated `twilio-chatbot` with a few new features: use 8000 sample rate and
  avoid resampling, a new client useful for stress testing and testing locally
  without the need to make phone calls. Also, added audio recording on both the
  client and the server to make sure the audio sounds good.

- Updated examples to use `task.cancel()` to immediately exit the example when a
  participant leaves or disconnects, instead of pushing an `EndFrame`. Pushing
  an `EndFrame` causes the bot to run through everything that is internally
  queued (which could take some seconds). Note that using `task.cancel()` might
  not always be the best option and pushing an `EndFrame` could still be
  desirable to make sure all the pipeline is flushed.

## [0.0.54] - 2025-01-27

### Added

- In order to create tasks in Pipecat frame processors it is now recommended to
  use `FrameProcessor.create_task()` (which uses the new
  `utils.asyncio.create_task()`). It takes care of uncaught exceptions, task
  cancellation handling and task management. To cancel or wait for a task there
  is `FrameProcessor.cancel_task()` and `FrameProcessor.wait_for_task()`. All of
  Pipecat processors have been updated accordingly. Also, when a pipeline runner
  finishes, a warning about dangling tasks might appear, which indicates if any
  of the created tasks was never cancelled or awaited for (using these new
  functions).

- It is now possible to specify the period of the `PipelineTask` heartbeat
  frames with `heartbeats_period_secs`.

- Added `DailyMeetingTokenProperties` and `DailyMeetingTokenParams` Pydantic models
  for meeting token creation in `get_token` method of `DailyRESTHelper`.

- Added `enable_recording` and `geo` parameters to `DailyRoomProperties`.

- Added `RecordingsBucketConfig` to `DailyRoomProperties` to upload recordings
  to a custom AWS bucket.

### Changed

- Enhanced `UserIdleProcessor` with retry functionality and control over idle
  monitoring via new callback signature `(processor, retry_count) -> bool`.
  Updated the `17-detect-user-idle.py` to show how to use the `retry_count`.

- Add defensive error handling for `OpenAIRealtimeBetaLLMService`'s audio
  truncation. Audio truncation errors during interruptions now log a warning
  and allow the session to continue instead of throwing an exception.

- Modified `TranscriptProcessor` to use TTS text frames for more accurate assistant
  transcripts. Assistant messages are now aggregated based on bot speaking boundaries
  rather than LLM context, providing better handling of interruptions and partial
  utterances.

- Updated foundational examples `28a-transcription-processor-openai.py`,
  `28b-transcript-processor-anthropic.py`, and
  `28c-transcription-processor-gemini.py` to use the updated
  `TranscriptProcessor`.

### Fixed

- Fixed an `GeminiMultimodalLiveLLMService` issue that was preventing the user
  to push initial LLM assistant messages (using `LLMMessagesAppendFrame`).

- Added missing `FrameProcessor.cleanup()` calls to `Pipeline`,
  `ParallelPipeline` and `UserIdleProcessor`.

- Fixed a type error when using `voice_settings` in `ElevenLabsHttpTTSService`.

- Fixed an issue where `OpenAIRealtimeBetaLLMService` function calling resulted
  in an error.

- Fixed an issue in `AudioBufferProcessor` where the last audio buffer was not
  being processed, in cases where the `_user_audio_buffer` was smaller than the
  buffer size.

### Performance

- Replaced audio resampling library `resampy` with `soxr`. Resampling a 2:21s
  audio file from 24KHz to 16KHz took 1.41s with `resampy` and 0.031s with
  `soxr` with similar audio quality.

### Other

- Added initial unit test infrastructure.

## [0.0.53] - 2025-01-18

### Added

- Added `ElevenLabsHttpTTSService` which uses EleveLabs' HTTP API instead of the
  websocket one.

- Introduced pipeline frame observers. Observers can view all the frames that go
  through the pipeline without the need to inject processors in the
  pipeline. This can be useful, for example, to implement frame loggers or
  debuggers among other things. The example
  `examples/foundational/30-observer.py` shows how to add an observer to a
  pipeline for debugging.

- Introduced heartbeat frames. The pipeline task can now push periodic
  heartbeats down the pipeline when `enable_heartbeats=True`. Heartbeats are
  system frames that are supposed to make it all the way to the end of the
  pipeline. When a heartbeat frame is received the traversing time (i.e. the
  time it took to go through the whole pipeline) will be displayed (with TRACE
  logging) otherwise a warning will be shown. The example
  `examples/foundational/31-heartbeats.py` shows how to enable heartbeats and
  forces warnings to be displayed.

- Added `LLMTextFrame` and `TTSTextFrame` which should be pushed by LLM and TTS
  services respectively instead of `TextFrame`s.

- Added `OpenRouter` for OpenRouter integration with an OpenAI-compatible
  interface. Added foundational example `14m-function-calling-openrouter.py`.

- Added a new `WebsocketService` based class for TTS services, containing
  base functions and retry logic.

- Added `DeepSeekLLMService` for DeepSeek integration with an OpenAI-compatible
  interface. Added foundational example `14l-function-calling-deepseek.py`.

- Added `FunctionCallResultProperties` dataclass to provide a structured way to
  control function call behavior, including:

  - `run_llm`: Controls whether to trigger LLM completion
  - `on_context_updated`: Optional callback triggered after context update

- Added a new foundational example `07e-interruptible-playht-http.py` for easy
  testing of `PlayHTHttpTTSService`.

- Added support for Google TTS Journey voices in `GoogleTTSService`.

- Added `29-livekit-audio-chat.py`, as a new foundational examples for
  `LiveKitTransportLayer`.

- Added `enable_prejoin_ui`, `max_participants` and `start_video_off` params
  to `DailyRoomProperties`.

- Added `session_timeout` to `FastAPIWebsocketTransport` and
  `WebsocketServerTransport` for configuring session timeouts (in
  seconds). Triggers `on_session_timeout` for custom timeout handling.
  See [examples/websocket-server/bot.py](https://github.com/pipecat-ai/pipecat/blob/main/examples/websocket-server/bot.py).

- Added the new modalities option and helper function to set Gemini output
  modalities.

- Added `examples/foundational/26d-gemini-multimodal-live-text.py` which is
  using Gemini as TEXT modality and using another TTS provider for TTS process.

### Changed

- Modified `UserIdleProcessor` to start monitoring only after first
  conversation activity (`UserStartedSpeakingFrame` or
  `BotStartedSpeakingFrame`) instead of immediately.

- Modified `OpenAIAssistantContextAggregator` to support controlled completions
  and to emit context update callbacks via `FunctionCallResultProperties`.

- Added `aws_session_token` to the `PollyTTSService`.

- Changed the default model for `PlayHTHttpTTSService` to `Play3.0-mini-http`.

- `api_key`, `aws_access_key_id` and `region` are no longer required parameters
  for the PollyTTSService (AWSTTSService)

- Added `session_timeout` example in `examples/websocket-server/bot.py` to
  handle session timeout event.

- Changed `InputParams` in
  `src/pipecat/services/gemini_multimodal_live/gemini.py` to support different
  modalities.

- Changed `DeepgramSTTService` to send `finalize` event whenever VAD detects
  `UserStoppedSpeakingFrame`. This helps in faster transcriptions and clearing
  the `Deepgram` audio buffer.

### Fixed

- Fixed an issue where `DeepgramSTTService` was not generating metrics using
  pipeline's VAD.

- Fixed `UserIdleProcessor` not properly propagating `EndFrame`s through the
  pipeline.

- Fixed an issue where websocket based TTS services could incorrectly terminate
  their connection due to a retry counter not resetting.

- Fixed a `PipelineTask` issue that would cause a dangling task after stopping
  the pipeline with an `EndFrame`.

- Fixed an import issue for `PlayHTHttpTTSService`.

- Fixed an issue where languages couldn't be used with the `PlayHTHttpTTSService`.

- Fixed an issue where `OpenAIRealtimeBetaLLMService` audio chunks were hitting
  an error when truncating audio content.

- Fixed an issue where setting the voice and model for `RimeHttpTTSService`
  wasn't working.

- Fixed an issue where `IdleFrameProcessor` and `UserIdleProcessor` were getting
  initialized before the start of the pipeline.

## [0.0.52] - 2024-12-24

### Added

- Constructor arguments for GoogleLLMService to directly set tools and tool_config.

- Smart turn detection example (`22d-natural-conversation-gemini-audio.py`) that
  leverages Gemini 2.0 capabilities ().
  (see https://x.com/kwindla/status/1870974144831275410)

- Added `DailyTransport.send_dtmf()` to send dial-out DTMF tones.

- Added `DailyTransport.sip_call_transfer()` to forward SIP and PSTN calls to
  another address or number. For example, transfer a SIP call to a different
  SIP address or transfer a PSTN phone number to a different PSTN phone number.

- Added `DailyTransport.sip_refer()` to transfer incoming SIP/PSTN calls from
  outside Daily to another SIP/PSTN address.

- Added an `auto_mode` input parameter to `ElevenLabsTTSService`. `auto_mode`
  is set to `True` by default. Enabling this setting disables the chunk
  schedule and all buffers, which reduces latency.

- Added `KoalaFilter` which implement on device noise reduction using Koala
  Noise Suppression.
  (see https://picovoice.ai/platform/koala/)

- Added `CerebrasLLMService` for Cerebras integration with an OpenAI-compatible
  interface. Added foundational example `14k-function-calling-cerebras.py`.

- Pipecat now supports Python 3.13. We had a dependency on the `audioop` package
  which was deprecated and now removed on Python 3.13. We are now using
  `audioop-lts` (https://github.com/AbstractUmbra/audioop) to provide the same
  functionality.

- Added timestamped conversation transcript support:

  - New `TranscriptProcessor` factory provides access to user and assistant
    transcript processors.
  - `UserTranscriptProcessor` processes user speech with timestamps from
    transcription.
  - `AssistantTranscriptProcessor` processes assistant responses with LLM
    context timestamps.
  - Messages emitted with ISO 8601 timestamps indicating when they were spoken.
  - Supports all LLM formats (OpenAI, Anthropic, Google) via standard message
    format.
  - New examples: `28a-transcription-processor-openai.py`,
    `28b-transcription-processor-anthropic.py`, and
    `28c-transcription-processor-gemini.py`.

- Add support for more languages to ElevenLabs (Arabic, Croatian, Filipino,
  Tamil) and PlayHT (Afrikans, Albanian, Amharic, Arabic, Bengali, Croatian,
  Galician, Hebrew, Mandarin, Serbian, Tagalog, Urdu, Xhosa).

### Changed

- `PlayHTTTSService` uses the new v4 websocket API, which also fixes an issue
  where text inputted to the TTS didn't return audio.

- The default model for `ElevenLabsTTSService` is now `eleven_flash_v2_5`.

- `OpenAIRealtimeBetaLLMService` now takes a `model` parameter in the
  constructor.

- Updated the default model for the `OpenAIRealtimeBetaLLMService`.

- Room expiration (`exp`) in `DailyRoomProperties` is now optional (`None`) by
  default instead of automatically setting a 5-minute expiration time. You must
  explicitly set expiration time if desired.

### Deprecated

- `AWSTTSService` is now deprecated, use `PollyTTSService` instead.

### Fixed

- Fixed token counting in `GoogleLLMService`. Tokens were summed incorrectly
  (double-counted in many cases).

- Fixed an issue that could cause the bot to stop talking if there was a user
  interruption before getting any audio from the TTS service.

- Fixed an issue that would cause `ParallelPipeline` to handle `EndFrame`
  incorrectly causing the main pipeline to not terminate or terminate too early.

- Fixed an audio stuttering issue in `FastPitchTTSService`.

- Fixed a `BaseOutputTransport` issue that was causing non-audio frames being
  processed before the previous audio frames were played. This will allow, for
  example, sending a frame `A` after a `TTSSpeakFrame` and the frame `A` will
  only be pushed downstream after the audio generated from `TTSSpeakFrame` has
  been spoken.

- Fixed a `DeepgramSTTService` issue that was causing language to be passed as
  an object instead of a string resulting in the connection to fail.

## [0.0.51] - 2024-12-16

### Fixed

- Fixed an issue in websocket-based TTS services that was causing infinite
  reconnections (Cartesia, ElevenLabs, PlayHT and LMNT).

## [0.0.50] - 2024-12-11

### Added

- Added `GeminiMultimodalLiveLLMService`. This is an integration for Google's
  Gemini Multimodal Live API, supporting:

  - Real-time audio and video input processing
  - Streaming text responses with TTS
  - Audio transcription for both user and bot speech
  - Function calling
  - System instructions and context management
  - Dynamic parameter updates (temperature, top_p, etc.)

- Added `AudioTranscriber` utility class for handling audio transcription with
  Gemini models.

- Added new context classes for Gemini:

  - `GeminiMultimodalLiveContext`
  - `GeminiMultimodalLiveUserContextAggregator`
  - `GeminiMultimodalLiveAssistantContextAggregator`
  - `GeminiMultimodalLiveContextAggregatorPair`

- Added new foundational examples for `GeminiMultimodalLiveLLMService`:

  - `26-gemini-multimodal-live.py`
  - `26a-gemini-multimodal-live-transcription.py`
  - `26b-gemini-multimodal-live-video.py`
  - `26c-gemini-multimodal-live-video.py`

- Added `SimliVideoService`. This is an integration for Simli AI avatars.
  (see https://www.simli.com)

- Added NVIDIA Riva's `FastPitchTTSService` and `ParakeetSTTService`.
  (see https://www.nvidia.com/en-us/ai-data-science/products/riva/)

- Added `IdentityFilter`. This is the simplest frame filter that lets through
  all incoming frames.

- New `STTMuteStrategy` called `FUNCTION_CALL` which mutes the STT service
  during LLM function calls.

- `DeepgramSTTService` now exposes two event handlers `on_speech_started` and
  `on_utterance_end` that could be used to implement interruptions. See new
  example `examples/foundational/07c-interruptible-deepgram-vad.py`.

- Added `GroqLLMService`, `GrokLLMService`, and `NimLLMService` for Groq, Grok,
  and NVIDIA NIM API integration, with an OpenAI-compatible interface.

- New examples demonstrating function calling with Groq, Grok, Azure OpenAI,
  Fireworks, and NVIDIA NIM: `14f-function-calling-groq.py`,
  `14g-function-calling-grok.py`, `14h-function-calling-azure.py`,
  `14i-function-calling-fireworks.py`, and `14j-function-calling-nvidia.py`.

- In order to obtain the audio stored by the `AudioBufferProcessor` you can now
  also register an `on_audio_data` event handler. The `on_audio_data` handler
  will be called every time `buffer_size` (a new constructor argument) is
  reached. If `buffer_size` is 0 (default) you need to manually get the audio as
  before using `AudioBufferProcessor.merge_audio_buffers()`.

```
@audiobuffer.event_handler("on_audio_data")
async def on_audio_data(processor, audio, sample_rate, num_channels):
    await save_audio(audio, sample_rate, num_channels)
```

- Added a new RTVI message called `disconnect-bot`, which when handled pushes
  an `EndFrame` to trigger the pipeline to stop.

### Changed

- `STTMuteFilter` now supports multiple simultaneous muting strategies.

- `XTTSService` language now defaults to `Language.EN`.

- `SoundfileMixer` doesn't resample input files anymore to avoid startup
  delays. The sample rate of the provided sound files now need to match the
  sample rate of the output transport.

- Input frames (audio, image and transport messages) are now system frames. This
  means they are processed immediately by all processors instead of being queued
  internally.

- Expanded the transcriptions.language module to support a superset of
  languages.

- Updated STT and TTS services with language options that match the supported
  languages for each service.

- Updated the `AzureLLMService` to use the `OpenAILLMService`. Updated the
  `api_version` to `2024-09-01-preview`.

- Updated the `FireworksLLMService` to use the `OpenAILLMService`. Updated the
  default model to `accounts/fireworks/models/firefunction-v2`.

- Updated the `simple-chatbot` example to include a Javascript and React client
  example, using RTVI JS and React.

### Removed

- Removed `AppFrame`. This was used as a special user custom frame, but there's
  actually no use case for that.

### Fixed

- Fixed a `ParallelPipeline` issue that would cause system frames to be queued.

- Fixed `FastAPIWebsocketTransport` so it can work with binary data (e.g. using
  the protobuf serializer).

- Fixed an issue in `CartesiaTTSService` that could cause previous audio to be
  received after an interruption.

- Fixed Cartesia, ElevenLabs, LMNT and PlayHT TTS websocket
  reconnection. Before, if an error occurred no reconnection was happening.

- Fixed a `BaseOutputTransport` issue that was causing audio to be discarded
  after an `EndFrame` was received.

- Fixed an issue in `WebsocketServerTransport` and `FastAPIWebsocketTransport`
  that would cause a busy loop when using audio mixer.

- Fixed a `DailyTransport` and `LiveKitTransport` issue where connections were
  being closed in the input transport prematurely. This was causing frames
  queued inside the pipeline being discarded.

- Fixed an issue in `DailyTransport` that would cause some internal callbacks to
  not be executed.

- Fixed an issue where other frames were being processed while a `CancelFrame`
  was being pushed down the pipeline.

- `AudioBufferProcessor` now handles interruptions properly.

- Fixed a `WebsocketServerTransport` issue that would prevent interruptions with
  `TwilioSerializer` from working.

- `DailyTransport.capture_participant_video` now allows capturing user's screen
  share by simply passing `video_source="screenVideo"`.

- Fixed Google Gemini message handling to properly convert appended messages to
  Gemini's required format.

- Fixed an issue with `FireworksLLMService` where chat completions were failing
  by removing the `stream_options` from the chat completion options.

## [0.0.49] - 2024-11-17

### Added

- Added RTVI `on_bot_started` event which is useful in a single turn
  interaction.

- Added `DailyTransport` events `dialin-connected`, `dialin-stopped`,
  `dialin-error` and `dialin-warning`. Needs daily-python >= 0.13.0.

- Added `RimeHttpTTSService` and the `07q-interruptible-rime.py` foundational
  example.

- Added `STTMuteFilter`, a general-purpose processor that combines STT
  muting and interruption control. When active, it prevents both transcription
  and interruptions during bot speech. The processor supports multiple
  strategies: `FIRST_SPEECH` (mute only during bot's first
  speech), `ALWAYS` (mute during all bot speech), or `CUSTOM` (using provided
  callback).

- Added `STTMuteFrame`, a control frame that enables/disables speech
  transcription in STT services.

## [0.0.48] - 2024-11-10 "Antonio release"

### Added

- There's now an input queue in each frame processor. When you call
  `FrameProcessor.push_frame()` this will internally call
  `FrameProcessor.queue_frame()` on the next processor (upstream or downstream)
  and the frame will be internally queued (except system frames). Then, the
  queued frames will get processed. With this input queue it is also possible
  for FrameProcessors to block processing more frames by calling
  `FrameProcessor.pause_processing_frames()`. The way to resume processing
  frames is by calling `FrameProcessor.resume_processing_frames()`.

- Added audio filter `NoisereduceFilter`.

- Introduce input transport audio filters (`BaseAudioFilter`). Audio filters can
  be used to remove background noises before audio is sent to VAD.

- Introduce output transport audio mixers (`BaseAudioMixer`). Output transport
  audio mixers can be used, for example, to add background sounds or any other
  audio mixing functionality before the output audio is actually written to the
  transport.

- Added `GatedOpenAILLMContextAggregator`. This aggregator keeps the last
  received OpenAI LLM context frame and it doesn't let it through until the
  notifier is notified.

- Added `WakeNotifierFilter`. This processor expects a list of frame types and
  will execute a given callback predicate when a frame of any of those type is
  being processed. If the callback returns true the notifier will be notified.

- Added `NullFilter`. A null filter doesn't push any frames upstream or
  downstream. This is usually used to disable one of the pipelines in
  `ParallelPipeline`.

- Added `EventNotifier`. This can be used as a very simple synchronization
  feature between processors.

- Added `TavusVideoService`. This is an integration for Tavus digital twins.
  (see https://www.tavus.io/)

- Added `DailyTransport.update_subscriptions()`. This allows you to have fine
  grained control of what media subscriptions you want for each participant in a
  room.

- Added audio filter `KrispFilter`.

### Changed

- The following `DailyTransport` functions are now `async` which means they need
  to be awaited: `start_dialout`, `stop_dialout`, `start_recording`,
  `stop_recording`, `capture_participant_transcription` and
  `capture_participant_video`.

- Changed default output sample rate to 24000. This changes all TTS service to
  output to 24000 and also the default output transport sample rate. This
  improves audio quality at the cost of some extra bandwidth.

- `AzureTTSService` now uses Azure websockets instead of HTTP requests.

- The previous `AzureTTSService` HTTP implementation is now
  `AzureHttpTTSService`.

### Fixed

- Websocket transports (FastAPI and Websocket) now synchronize with time before
  sending data. This allows for interruptions to just work out of the box.

- Improved bot speaking detection for all TTS services by using actual bot
  audio.

- Fixed an issue that was generating constant bot started/stopped speaking
  frames for HTTP TTS services.

- Fixed an issue that was causing stuttering with AWS TTS service.

- Fixed an issue with PlayHTTTSService, where the TTFB metrics were reporting
  very small time values.

- Fixed an issue where AzureTTSService wasn't initializing the specified
  language.

### Other

- Add `23-bot-background-sound.py` foundational example.

- Added a new foundational example `22-natural-conversation.py`. This example
  shows how to achieve a more natural conversation detecting when the user ends
  statement.

## [0.0.47] - 2024-10-22

### Added

- Added `AssemblyAISTTService` and corresponding foundational examples
  `07o-interruptible-assemblyai.py` and `13d-assemblyai-transcription.py`.

- Added a foundational example for Gladia transcription:
  `13c-gladia-transcription.py`

### Changed

- Updated `GladiaSTTService` to use the V2 API.

- Changed `DailyTransport` transcription model to `nova-2-general`.

### Fixed

- Fixed an issue that would cause an import error when importing
  `SileroVADAnalyzer` from the old package `pipecat.vad.silero`.

- Fixed `enable_usage_metrics` to control LLM/TTS usage metrics separately
  from `enable_metrics`.

## [0.0.46] - 2024-10-19

### Added

- Added `audio_passthrough` parameter to `STTService`. If enabled it allows
  audio frames to be pushed downstream in case other processors need them.

- Added input parameter options for `PlayHTTTSService` and
  `PlayHTHttpTTSService`.

### Changed

- Changed `DeepgramSTTService` model to `nova-2-general`.

- Moved `SileroVAD` audio processor to `processors.audio.vad`.

- Module `utils.audio` is now `audio.utils`. A new `resample_audio` function has
  been added.

- `PlayHTTTSService` now uses PlayHT websockets instead of HTTP requests.

- The previous `PlayHTTTSService` HTTP implementation is now
  `PlayHTHttpTTSService`.

- `PlayHTTTSService` and `PlayHTHttpTTSService` now use a `voice_engine` of
  `PlayHT3.0-mini`, which allows for multi-lingual support.

- Renamed `OpenAILLMServiceRealtimeBeta` to `OpenAIRealtimeBetaLLMService` to
  match other services.

### Deprecated

- `LLMUserResponseAggregator` and `LLMAssistantResponseAggregator` are
  mostly deprecated, use `OpenAILLMContext` instead.

- The `vad` package is now deprecated and `audio.vad` should be used
  instead. The `avd` package will get removed in a future release.

### Fixed

- Fixed an issue that would cause an error if no VAD analyzer was passed to
  `LiveKitTransport` params.

- Fixed `SileroVAD` processor to support interruptions properly.

### Other

- Added `examples/foundational/07-interruptible-vad.py`. This is the same as
  `07-interruptible.py` but using the `SileroVAD` processor instead of passing
  the `VADAnalyzer` in the transport.

## [0.0.45] - 2024-10-16

### Changed

- Metrics messages have moved out from the transport's base output into RTVI.

## [0.0.44] - 2024-10-15

### Added

- Added support for OpenAI Realtime API with the new
  `OpenAILLMServiceRealtimeBeta` processor.
  (see https://platform.openai.com/docs/guides/realtime/overview)

- Added `RTVIBotTranscriptionProcessor` which will send the RTVI
  `bot-transcription` protocol message. These are TTS text aggregated (into
  sentences) messages.

- Added new input params to the `MarkdownTextFilter` utility. You can set
  `filter_code` to filter code from text and `filter_tables` to filter tables
  from text.

- Added `CanonicalMetricsService`. This processor uses the new
  `AudioBufferProcessor` to capture conversation audio and later send it to
  Canonical AI.
  (see https://canonical.chat/)

- Added `AudioBufferProcessor`. This processor can be used to buffer mixed user and
  bot audio. This can later be saved into an audio file or processed by some
  audio analyzer.

- Added `on_first_participant_joined` event to `LiveKitTransport`.

### Changed

- LLM text responses are now logged properly as unicode characters.

- `UserStartedSpeakingFrame`, `UserStoppedSpeakingFrame`,
  `BotStartedSpeakingFrame`, `BotStoppedSpeakingFrame`, `BotSpeakingFrame` and
  `UserImageRequestFrame` are now based from `SystemFrame`

### Fixed

- Merge `RTVIBotLLMProcessor`/`RTVIBotLLMTextProcessor` and
  `RTVIBotTTSProcessor`/`RTVIBotTTSTextProcessor` to avoid out of order issues.

- Fixed an issue in RTVI protocol that could cause a `bot-llm-stopped` or
  `bot-tts-stopped` message to be sent before a `bot-llm-text` or `bot-tts-text`
  message.

- Fixed `DeepgramSTTService` constructor settings not being merged with default
  ones.

- Fixed an issue in Daily transport that would cause tasks to be hanging if
  urgent transport messages were being sent from a transport event handler.

- Fixed an issue in `BaseOutputTransport` that would cause `EndFrame` to be
  pushed downed too early and call `FrameProcessor.cleanup()` before letting the
  transport stop properly.

## [0.0.43] - 2024-10-10

### Added

- Added a new util called `MarkdownTextFilter` which is a subclass of a new
  base class called `BaseTextFilter`. This is a configurable utility which
  is intended to filter text received by TTS services.

- Added new `RTVIUserLLMTextProcessor`. This processor will send an RTVI
  `user-llm-text` message with the user content's that was sent to the LLM.

### Changed

- `TransportMessageFrame` doesn't have an `urgent` field anymore, instead
  there's now a `TransportMessageUrgentFrame` which is a `SystemFrame` and
  therefore skip all internal queuing.

- For TTS services, convert inputted languages to match each service's language
  format

### Fixed

- Fixed an issue where changing a language with the Deepgram STT service
  wouldn't apply the change. This was fixed by disconnecting and reconnecting
  when the language changes.

## [0.0.42] - 2024-10-02

### Added

- `SentryMetrics` has been added to report frame processor metrics to
  Sentry. This is now possible because `FrameProcessorMetrics` can now be passed
  to `FrameProcessor`.

- Added Google TTS service and corresponding foundational example
  `07n-interruptible-google.py`

- Added AWS Polly TTS support and `07m-interruptible-aws.py` as an example.

- Added InputParams to Azure TTS service.

- Added `LivekitTransport` (audio-only for now).

- RTVI 0.2.0 is now supported.

- All `FrameProcessors` can now register event handlers.

```
tts = SomeTTSService(...)

@tts.event_handler("on_connected"):
async def on_connected(processor):
  ...
```

- Added `AsyncGeneratorProcessor`. This processor can be used together with a
  `FrameSerializer` as an async generator. It provides a `generator()` function
  that returns an `AsyncGenerator` and that yields serialized frames.

- Added `EndTaskFrame` and `CancelTaskFrame`. These are new frames that are
  meant to be pushed upstream to tell the pipeline task to stop nicely or
  immediately respectively.

- Added configurable LLM parameters (e.g., temperature, top_p, max_tokens, seed)
  for OpenAI, Anthropic, and Together AI services along with corresponding
  setter functions.

- Added `sample_rate` as a constructor parameter for TTS services.

- Pipecat has a pipeline-based architecture. The pipeline consists of frame
  processors linked to each other. The elements traveling across the pipeline
  are called frames.

  To have a deterministic behavior the frames traveling through the pipeline
  should always be ordered, except system frames which are out-of-band
  frames. To achieve that, each frame processor should only output frames from a
  single task.

  In this version all the frame processors have their own task to push
  frames. That is, when `push_frame()` is called the given frame will be put
  into an internal queue (with the exception of system frames) and a frame
  processor task will push it out.

- Added pipeline clocks. A pipeline clock is used by the output transport to
  know when a frame needs to be presented. For that, all frames now have an
  optional `pts` field (prensentation timestamp). There's currently just one
  clock implementation `SystemClock` and the `pts` field is currently only used
  for `TextFrame`s (audio and image frames will be next).

- A clock can now be specified to `PipelineTask` (defaults to
  `SystemClock`). This clock will be passed to each frame processor via the
  `StartFrame`.

- Added `CartesiaHttpTTSService`.

- `DailyTransport` now supports setting the audio bitrate to improve audio
  quality through the `DailyParams.audio_out_bitrate` parameter. The new
  default is 96kbps.

- `DailyTransport` now uses the number of audio output channels (1 or 2) to set
  mono or stereo audio when needed.

- Interruptions support has been added to `TwilioFrameSerializer` when using
  `FastAPIWebsocketTransport`.

- Added new `LmntTTSService` text-to-speech service.
  (see https://www.lmnt.com/)

- Added `TTSModelUpdateFrame`, `TTSLanguageUpdateFrame`, `STTModelUpdateFrame`,
  and `STTLanguageUpdateFrame` frames to allow you to switch models, language
  and voices in TTS and STT services.

- Added new `transcriptions.Language` enum.

### Changed

- Context frames are now pushed downstream from assistant context aggregators.

- Removed Silero VAD torch dependency.

- Updated individual update settings frame classes into a single
  `ServiceUpdateSettingsFrame` class.

- We now distinguish between input and output audio and image frames. We
  introduce `InputAudioRawFrame`, `OutputAudioRawFrame`, `InputImageRawFrame`
  and `OutputImageRawFrame` (and other subclasses of those). The input frames
  usually come from an input transport and are meant to be processed inside the
  pipeline to generate new frames. However, the input frames will not be sent
  through an output transport. The output frames can also be processed by any
  frame processor in the pipeline and they are allowed to be sent by the output
  transport.

- `ParallelTask` has been renamed to `SyncParallelPipeline`. A
  `SyncParallelPipeline` is a frame processor that contains a list of different
  pipelines to be executed concurrently. The difference between a
  `SyncParallelPipeline` and a `ParallelPipeline` is that, given an input frame,
  the `SyncParallelPipeline` will wait for all the internal pipelines to
  complete. This is achieved by making sure the last processor in each of the
  pipelines is synchronous (e.g. an HTTP-based service that waits for the
  response).

- `StartFrame` is back a system frame to make sure it's processed immediately by
  all processors. `EndFrame` stays a control frame since it needs to be ordered
  allowing the frames in the pipeline to be processed.

- Updated `MoondreamService` revision to `2024-08-26`.

- `CartesiaTTSService` and `ElevenLabsTTSService` now add presentation
  timestamps to their text output. This allows the output transport to push the
  text frames downstream at almost the same time the words are spoken. We say
  "almost" because currently the audio frames don't have presentation timestamp
  but they should be played at roughly the same time.

- `DailyTransport.on_joined` event now returns the full session data instead of
  just the participant.

- `CartesiaTTSService` is now a subclass of `TTSService`.

- `DeepgramSTTService` is now a subclass of `STTService`.

- `WhisperSTTService` is now a subclass of `SegmentedSTTService`. A
  `SegmentedSTTService` is a `STTService` where the provided audio is given in a
  big chunk (i.e. from when the user starts speaking until the user stops
  speaking) instead of a continous stream.

### Fixed

- Fixed OpenAI multiple function calls.

- Fixed a Cartesia TTS issue that would cause audio to be truncated in some
  cases.

- Fixed a `BaseOutputTransport` issue that would stop audio and video rendering
  tasks (after receiving and `EndFrame`) before the internal queue was emptied,
  causing the pipeline to finish prematurely.

- `StartFrame` should be the first frame every processor receives to avoid
  situations where things are not initialized (because initialization happens on
  `StartFrame`) and other frames come in resulting in undesired behavior.

### Performance

- `obj_id()` and `obj_count()` now use `itertools.count` avoiding the need of
  `threading.Lock`.

### Other

- Pipecat now uses Ruff as its formatter (https://github.com/astral-sh/ruff).

## [0.0.41] - 2024-08-22

### Added

- Added `LivekitFrameSerializer` audio frame serializer.

### Fixed

- Fix `FastAPIWebsocketOutputTransport` variable name clash with subclass.

- Fix an `AnthropicLLMService` issue with empty arguments in function calling.

### Other

- Fixed `studypal` example errors.

## [0.0.40] - 2024-08-20

### Added

- VAD parameters can now be dynamicallt updated using the
  `VADParamsUpdateFrame`.

- `ErrorFrame` has now a `fatal` field to indicate the bot should exit if a
  fatal error is pushed upstream (false by default). A new `FatalErrorFrame`
  that sets this flag to true has been added.

- `AnthropicLLMService` now supports function calling and initial support for
  prompt caching.
  (see https://www.anthropic.com/news/prompt-caching)

- `ElevenLabsTTSService` can now specify ElevenLabs input parameters such as
  `output_format`.

- `TwilioFrameSerializer` can now specify Twilio's and Pipecat's desired sample
  rates to use.

- Added new `on_participant_updated` event to `DailyTransport`.

- Added `DailyRESTHelper.delete_room_by_name()` and
  `DailyRESTHelper.delete_room_by_url()`.

- Added LLM and TTS usage metrics. Those are enabled when
  `PipelineParams.enable_usage_metrics` is True.

- `AudioRawFrame`s are now pushed downstream from the base output
  transport. This allows capturing the exact words the bot says by adding an STT
  service at the end of the pipeline.

- Added new `GStreamerPipelineSource`. This processor can generate image or
  audio frames from a GStreamer pipeline (e.g. reading an MP4 file, and RTP
  stream or anything supported by GStreamer).

- Added `TransportParams.audio_out_is_live`. This flag is False by default and
  it is useful to indicate we should not synchronize audio with sporadic images.

- Added new `BotStartedSpeakingFrame` and `BotStoppedSpeakingFrame` control
  frames. These frames are pushed upstream and they should wrap
  `BotSpeakingFrame`.

- Transports now allow you to register event handlers without decorators.

### Changed

- Support RTVI message protocol 0.1. This includes new messages, support for
  messages responses, support for actions, configuration, webhooks and a bunch
  of new cool stuff.
  (see https://docs.rtvi.ai/)

- `SileroVAD` dependency is now imported via pip's `silero-vad` package.

- `ElevenLabsTTSService` now uses `eleven_turbo_v2_5` model by default.

- `BotSpeakingFrame` is now a control frame.

- `StartFrame` is now a control frame similar to `EndFrame`.

- `DeepgramTTSService` now is more customizable. You can adjust the encoding and
  sample rate.

### Fixed

- `TTSStartFrame` and `TTSStopFrame` are now sent when TTS really starts and
  stops. This allows for knowing when the bot starts and stops speaking even
  with asynchronous services (like Cartesia).

- Fixed `AzureSTTService` transcription frame timestamps.

- Fixed an issue with `DailyRESTHelper.create_room()` expirations which would
  cause this function to stop working after the initial expiration elapsed.

- Improved `EndFrame` and `CancelFrame` handling. `EndFrame` should end things
  gracefully while a `CancelFrame` should cancel all running tasks as soon as
  possible.

- Fixed an issue in `AIService` that would cause a yielded `None` value to be
  processed.

- RTVI's `bot-ready` message is now sent when the RTVI pipeline is ready and
  a first participant joins.

- Fixed a `BaseInputTransport` issue that was causing incoming system frames to
  be queued instead of being pushed immediately.

- Fixed a `BaseInputTransport` issue that was causing start/stop interruptions
  incoming frames to not cancel tasks and be processed properly.

### Other

- Added `studypal` example (from to the Cartesia folks!).

- Most examples now use Cartesia.

- Added examples `foundational/19a-tools-anthropic.py`,
  `foundational/19b-tools-video-anthropic.py` and
  `foundational/19a-tools-togetherai.py`.

- Added examples `foundational/18-gstreamer-filesrc.py` and
  `foundational/18a-gstreamer-videotestsrc.py` that show how to use
  `GStreamerPipelineSource`

- Remove `requests` library usage.

- Cleanup examples and use `DailyRESTHelper`.

## [0.0.39] - 2024-07-23

### Fixed

- Fixed a regression introduced in 0.0.38 that would cause Daily transcription
  to stop the Pipeline.

## [0.0.38] - 2024-07-23

### Added

- Added `force_reload`, `skip_validation` and `trust_repo` to `SileroVAD` and
  `SileroVADAnalyzer`. This allows caching and various GitHub repo validations.

- Added `send_initial_empty_metrics` flag to `PipelineParams` to request for
  initial empty metrics (zero values). True by default.

### Fixed

- Fixed initial metrics format. It was using the wrong keys name/time instead of
  processor/value.

- STT services should be using ISO 8601 time format for transcription frames.

- Fixed an issue that would cause Daily transport to show a stop transcription
  error when actually none occurred.

## [0.0.37] - 2024-07-22

### Added

- Added `RTVIProcessor` which implements the RTVI-AI standard.
  See https://github.com/rtvi-ai

- Added `BotInterruptionFrame` which allows interrupting the bot while talking.

- Added `LLMMessagesAppendFrame` which allows appending messages to the current
  LLM context.

- Added `LLMMessagesUpdateFrame` which allows changing the LLM context for the
  one provided in this new frame.

- Added `LLMModelUpdateFrame` which allows updating the LLM model.

- Added `TTSSpeakFrame` which causes the bot say some text. This text will not
  be part of the LLM context.

- Added `TTSVoiceUpdateFrame` which allows updating the TTS voice.

### Removed

- We remove the `LLMResponseStartFrame` and `LLMResponseEndFrame` frames. These
  were added in the past to properly handle interruptions for the
  `LLMAssistantContextAggregator`. But the `LLMContextAggregator` is now based
  on `LLMResponseAggregator` which handles interruptions properly by just
  processing the `StartInterruptionFrame`, so there's no need for these extra
  frames any more.

### Fixed

- Fixed an issue with `StatelessTextTransformer` where it was pushing a string
  instead of a `TextFrame`.

- `TTSService` end of sentence detection has been improved. It now works with
  acronyms, numbers, hours and others.

- Fixed an issue in `TTSService` that would not properly flush the current
  aggregated sentence if an `LLMFullResponseEndFrame` was found.

### Performance

- `CartesiaTTSService` now uses websockets which improves speed. It also
  leverages the new Cartesia contexts which maintains generated audio prosody
  when multiple inputs are sent, therefore improving audio quality a lot.

## [0.0.36] - 2024-07-02

### Added

- Added `GladiaSTTService`.
  See https://docs.gladia.io/chapters/speech-to-text-api/pages/live-speech-recognition

- Added `XTTSService`. This is a local Text-To-Speech service.
  See https://github.com/coqui-ai/TTS

- Added `UserIdleProcessor`. This processor can be used to wait for any
  interaction with the user. If the user doesn't say anything within a given
  timeout a provided callback is called.

- Added `IdleFrameProcessor`. This processor can be used to wait for frames
  within a given timeout. If no frame is received within the timeout a provided
  callback is called.

- Added new frame `BotSpeakingFrame`. This frame will be continuously pushed
  upstream while the bot is talking.

- It is now possible to specify a Silero VAD version when using `SileroVADAnalyzer`
  or `SileroVAD`.

- Added `AysncFrameProcessor` and `AsyncAIService`. Some services like
  `DeepgramSTTService` need to process things asynchronously. For example, audio
  is sent to Deepgram but transcriptions are not returned immediately. In these
  cases we still require all frames (except system frames) to be pushed
  downstream from a single task. That's what `AsyncFrameProcessor` is for. It
  creates a task and all frames should be pushed from that task. So, whenever a
  new Deepgram transcription is ready that transcription will also be pushed
  from this internal task.

- The `MetricsFrame` now includes processing metrics if metrics are enabled. The
  processing metrics indicate the time a processor needs to generate all its
  output. Note that not all processors generate these kind of metrics.

### Changed

- `WhisperSTTService` model can now also be a string.

- Added missing \* keyword separators in services.

### Fixed

- `WebsocketServerTransport` doesn't try to send frames anymore if serializers
  returns `None`.

- Fixed an issue where exceptions that occurred inside frame processors were
  being swallowed and not displayed.

- Fixed an issue in `FastAPIWebsocketTransport` where it would still try to send
  data to the websocket after being closed.

### Other

- Added Fly.io deployment example in `examples/deployment/flyio-example`.

- Added new `17-detect-user-idle.py` example that shows how to use the new
  `UserIdleProcessor`.

## [0.0.35] - 2024-06-28

### Changed

- `FastAPIWebsocketParams` now require a serializer.

- `TwilioFrameSerializer` now requires a `streamSid`.

### Fixed

- Silero VAD number of frames needs to be 512 for 16000 sample rate or 256 for
  8000 sample rate.

## [0.0.34] - 2024-06-25

### Fixed

- Fixed an issue with asynchronous STT services (Deepgram and Azure) that could
  interruptions to ignore transcriptions.

- Fixed an issue introduced in 0.0.33 that would cause the LLM to generate
  shorter output.

## [0.0.33] - 2024-06-25

### Changed

- Upgraded to Cartesia's new Python library 1.0.0. `CartesiaTTSService` now
  expects a voice ID instead of a voice name (you can get the voice ID from
  Cartesia's playground). You can also specify the audio `sample_rate` and
  `encoding` instead of the previous `output_format`.

### Fixed

- Fixed an issue with asynchronous STT services (Deepgram and Azure) that could
  cause static audio issues and interruptions to not work properly when dealing
  with multiple LLMs sentences.

- Fixed an issue that could mix new LLM responses with previous ones when
  handling interruptions.

- Fixed a Daily transport blocking situation that occurred while reading audio
  frames after a participant left the room. Needs daily-python >= 0.10.1.

## [0.0.32] - 2024-06-22

### Added

- Allow specifying a `DeepgramSTTService` url which allows using on-prem
  Deepgram.

- Added new `FastAPIWebsocketTransport`. This is a new websocket transport that
  can be integrated with FastAPI websockets.

- Added new `TwilioFrameSerializer`. This is a new serializer that knows how to
  serialize and deserialize audio frames from Twilio.

- Added Daily transport event: `on_dialout_answered`. See
  https://reference-python.daily.co/api_reference.html#daily.EventHandler

- Added new `AzureSTTService`. This allows you to use Azure Speech-To-Text.

### Performance

- Convert `BaseOutputTransport` and `BaseOutputTransport` to fully use asyncio
  and remove the use of threads.

### Other

- Added `twilio-chatbot`. This is an example that shows how to integrate Twilio
  phone numbers with a Pipecat bot.

- Updated `07f-interruptible-azure.py` to use `AzureLLMService`,
  `AzureSTTService` and `AzureTTSService`.

## [0.0.31] - 2024-06-13

### Performance

- Break long audio frames into 20ms chunks instead of 10ms.

## [0.0.30] - 2024-06-13

### Added

- Added `report_only_initial_ttfb` to `PipelineParams`. This will make it so
  only the initial TTFB metrics after the user stops talking are reported.

- Added `OpenPipeLLMService`. This service will let you run OpenAI through
  OpenPipe's SDK.

- Allow specifying frame processors' name through a new `name` constructor
  argument.

- Added `DeepgramSTTService`. This service has an ongoing websocket
  connection. To handle this, it subclasses `AIService` instead of
  `STTService`. The output of this service will be pushed from the same task,
  except system frames like `StartFrame`, `CancelFrame` or
  `StartInterruptionFrame`.

### Changed

- `FrameSerializer.deserialize()` can now return `None` in case it is not
  possible to desearialize the given data.

- `daily_rest.DailyRoomProperties` now allows extra unknown parameters.

### Fixed

- Fixed an issue where `DailyRoomProperties.exp` always had the same old
  timestamp unless set by the user.

- Fixed a couple of issues with `WebsocketServerTransport`. It needed to use
  `push_audio_frame()` and also VAD was not working properly.

- Fixed an issue that would cause LLM aggregator to fail with small
  `VADParams.stop_secs` values.

- Fixed an issue where `BaseOutputTransport` would send longer audio frames
  preventing interruptions.

### Other

- Added new `07h-interruptible-openpipe.py` example. This example shows how to
  use OpenPipe to run OpenAI LLMs and get the logs stored in OpenPipe.

- Added new `dialin-chatbot` example. This examples shows how to call the bot
  using a phone number.

## [0.0.29] - 2024-06-07

### Added

- Added a new `FunctionFilter`. This filter will let you filter frames based on
  a given function, except system messages which should never be filtered.

- Added `FrameProcessor.can_generate_metrics()` method to indicate if a
  processor can generate metrics. In the future this might get an extra argument
  to ask for a specific type of metric.

- Added `BasePipeline`. All pipeline classes should be based on this class. All
  subclasses should implement a `processors_with_metrics()` method that returns
  a list of all `FrameProcessor`s in the pipeline that can generate metrics.

- Added `enable_metrics` to `PipelineParams`.

- Added `MetricsFrame`. The `MetricsFrame` will report different metrics in the
  system. Right now, it can report TTFB (Time To First Byte) values for
  different services, that is the time spent between the arrival of a `Frame` to
  the processor/service until the first `DataFrame` is pushed downstream. If
  metrics are enabled an intial `MetricsFrame` with all the services in the
  pipeline will be sent.

- Added TTFB metrics and debug logging for TTS services.

### Changed

- Moved `ParallelTask` to `pipecat.pipeline.parallel_task`.

### Fixed

- Fixed PlayHT TTS service to work properly async.

## [0.0.28] - 2024-06-05

### Fixed

- Fixed an issue with `SileroVADAnalyzer` that would cause memory to keep
  growing indefinitely.

## [0.0.27] - 2024-06-05

### Added

- Added `DailyTransport.participants()` and `DailyTransport.participant_counts()`.

## [0.0.26] - 2024-06-05

### Added

- Added `OpenAITTSService`.

- Allow passing `output_format` and `model_id` to `CartesiaTTSService` to change
  audio sample format and the model to use.

- Added `DailyRESTHelper` which helps you create Daily rooms and tokens in an
  easy way.

- `PipelineTask` now has a `has_finished()` method to indicate if the task has
  completed. If a task is never ran `has_finished()` will return False.

- `PipelineRunner` now supports SIGTERM. If received, the runner will be
  cancelled.

### Fixed

- Fixed an issue where `BaseInputTransport` and `BaseOutputTransport` where
  stopping push tasks before pushing `EndFrame` frames could cause the bots to
  get stuck.

- Fixed an error closing local audio transports.

- Fixed an issue with Deepgram TTS that was introduced in the previous release.

- Fixed `AnthropicLLMService` interruptions. If an interruption occurred, a
  `user` message could be appended after the previous `user` message. Anthropic
  does not allow that because it requires alternate `user` and `assistant`
  messages.

### Performance

- The `BaseInputTransport` does not pull audio frames from sub-classes any
  more. Instead, sub-classes now push audio frames into a queue in the base
  class. Also, `DailyInputTransport` now pushes audio frames every 20ms instead
  of 10ms.

- Remove redundant camera input thread from `DailyInputTransport`. This should
  improve performance a little bit when processing participant videos.

- Load Cartesia voice on startup.

## [0.0.25] - 2024-05-31

### Added

- Added WebsocketServerTransport. This will create a websocket server and will
  read messages coming from a client. The messages are serialized/deserialized
  with protobufs. See `examples/websocket-server` for a detailed example.

- Added function calling (LLMService.register_function()). This will allow the
  LLM to call functions you have registered when needed. For example, if you
  register a function to get the weather in Los Angeles and ask the LLM about
  the weather in Los Angeles, the LLM will call your function.
  See https://platform.openai.com/docs/guides/function-calling

- Added new `LangchainProcessor`.

- Added Cartesia TTS support (https://cartesia.ai/)

### Fixed

- Fixed SileroVAD frame processor.

- Fixed an issue where `camera_out_enabled` would cause the highg CPU usage if
  no image was provided.

### Performance

- Removed unnecessary audio input tasks.

## [0.0.24] - 2024-05-29

### Added

- Exposed `on_dialin_ready` for Daily transport SIP endpoint handling. This
  notifies when the Daily room SIP endpoints are ready. This allows integrating
  with third-party services like Twilio.

- Exposed Daily transport `on_app_message` event.

- Added Daily transport `on_call_state_updated` event.

- Added Daily transport `start_recording()`, `stop_recording` and
  `stop_dialout`.

### Changed

- Added `PipelineParams`. This replaces the `allow_interruptions` argument in
  `PipelineTask` and will allow future parameters in the future.

- Fixed Deepgram Aura TTS base_url and added ErrorFrame reporting.

- GoogleLLMService `api_key` argument is now mandatory.

### Fixed

- Daily tranport `dialin-ready` doesn't not block anymore and it now handles
  timeouts.

- Fixed AzureLLMService.

## [0.0.23] - 2024-05-23

### Fixed

- Fixed an issue handling Daily transport `dialin-ready` event.

## [0.0.22] - 2024-05-23

### Added

- Added Daily transport `start_dialout()` to be able to make phone or SIP calls.
  See https://reference-python.daily.co/api_reference.html#daily.CallClient.start_dialout

- Added Daily transport support for dial-in use cases.

- Added Daily transport events: `on_dialout_connected`, `on_dialout_stopped`,
  `on_dialout_error` and `on_dialout_warning`. See
  https://reference-python.daily.co/api_reference.html#daily.EventHandler

## [0.0.21] - 2024-05-22

### Added

- Added vision support to Anthropic service.

- Added `WakeCheckFilter` which allows you to pass information downstream only
  if you say a certain phrase/word.

### Changed

- `FrameSerializer.serialize()` and `FrameSerializer.deserialize()` are now
  `async`.

- `Filter` has been renamed to `FrameFilter` and it's now under
  `processors/filters`.

### Fixed

- Fixed Anthropic service to use new frame types.

- Fixed an issue in `LLMUserResponseAggregator` and `UserResponseAggregator`
  that would cause frames after a brief pause to not be pushed to the LLM.

- Clear the audio output buffer if we are interrupted.

- Re-add exponential smoothing after volume calculation. This makes sure the
  volume value being used doesn't fluctuate so much.

## [0.0.20] - 2024-05-22

### Added

- In order to improve interruptions we now compute a loudness level using
  [pyloudnorm](https://github.com/csteinmetz1/pyloudnorm). The audio coming
  WebRTC transports (e.g. Daily) have an Automatic Gain Control (AGC) algorithm
  applied to the signal, however we don't do that on our local PyAudio
  signals. This means that currently incoming audio from PyAudio is kind of
  broken. We will fix it in future releases.

### Fixed

- Fixed an issue where `StartInterruptionFrame` would cause
  `LLMUserResponseAggregator` to push the accumulated text causing the LLM
  respond in the wrong task. The `StartInterruptionFrame` should not trigger any
  new LLM response because that would be spoken in a different task.

- Fixed an issue where tasks and threads could be paused because the executor
  didn't have more tasks available. This was causing issues when cancelling and
  recreating tasks during interruptions.

## [0.0.19] - 2024-05-20

### Changed

- `LLMUserResponseAggregator` and `LLMAssistantResponseAggregator` internal
  messages are now exposed through the `messages` property.

### Fixed

- Fixed an issue where `LLMAssistantResponseAggregator` was not accumulating the
  full response but short sentences instead. If there's an interruption we only
  accumulate what the bot has spoken until now in a long response as well.

## [0.0.18] - 2024-05-20

### Fixed

- Fixed an issue in `DailyOuputTransport` where transport messages were not
  being sent.

## [0.0.17] - 2024-05-19

### Added

- Added `google.generativeai` model support, including vision. This new `google`
  service defaults to using `gemini-1.5-flash-latest`. Example in
  `examples/foundational/12a-describe-video-gemini-flash.py`.

- Added vision support to `openai` service. Example in
  `examples/foundational/12a-describe-video-gemini-flash.py`.

- Added initial interruptions support. The assistant contexts (or aggregators)
  should now be placed after the output transport. This way, only the completed
  spoken context is added to the assistant context.

- Added `VADParams` so you can control voice confidence level and others.

- `VADAnalyzer` now uses an exponential smoothed volume to improve speech
  detection. This is useful when voice confidence is high (because there's
  someone talking near you) but volume is low.

### Fixed

- Fixed an issue where TTSService was not pushing TextFrames downstream.

- Fixed issues with Ctrl-C program termination.

- Fixed an issue that was causing `StopTaskFrame` to actually not exit the
  `PipelineTask`.

## [0.0.16] - 2024-05-16

### Fixed

- `DailyTransport`: don't publish camera and audio tracks if not enabled.

- Fixed an issue in `BaseInputTransport` that was causing frames pushed
  downstream not pushed in the right order.

## [0.0.15] - 2024-05-15

### Fixed

- Quick hot fix for receiving `DailyTransportMessage`.

## [0.0.14] - 2024-05-15

### Added

- Added `DailyTransport` event `on_participant_left`.

- Added support for receiving `DailyTransportMessage`.

### Fixed

- Images are now resized to the size of the output camera. This was causing
  images not being displayed.

- Fixed an issue in `DailyTransport` that would not allow the input processor to
  shutdown if no participant ever joined the room.

- Fixed base transports start and stop. In some situation processors would halt
  or not shutdown properly.

## [0.0.13] - 2024-05-14

### Changed

- `MoondreamService` argument `model_id` is now `model`.

- `VADAnalyzer` arguments have been renamed for more clarity.

### Fixed

- Fixed an issue with `DailyInputTransport` and `DailyOutputTransport` that
  could cause some threads to not start properly.

- Fixed `STTService`. Add `max_silence_secs` and `max_buffer_secs` to handle
  better what's being passed to the STT service. Also add exponential smoothing
  to the RMS.

- Fixed `WhisperSTTService`. Add `no_speech_prob` to avoid garbage output text.

## [0.0.12] - 2024-05-14

### Added

- Added `DailyTranscriptionSettings` to be able to specify transcription
  settings much easier (e.g. language).

### Other

- Updated `simple-chatbot` with Spanish.

- Add missing dependencies in some of the examples.

## [0.0.11] - 2024-05-13

### Added

- Allow stopping pipeline tasks with new `StopTaskFrame`.

### Changed

- TTS, STT and image generation service now use `AsyncGenerator`.

### Fixed

- `DailyTransport`: allow registering for participant transcriptions even if
  input transport is not initialized yet.

### Other

- Updated `storytelling-chatbot`.

## [0.0.10] - 2024-05-13

### Added

- Added Intel GPU support to `MoondreamService`.

- Added support for sending transport messages (e.g. to communicate with an app
  at the other end of the transport).

- Added `FrameProcessor.push_error()` to easily send an `ErrorFrame` upstream.

### Fixed

- Fixed Azure services (TTS and image generation).

### Other

- Updated `simple-chatbot`, `moondream-chatbot` and `translation-chatbot`
  examples.

## [0.0.9] - 2024-05-12

### Changed

Many things have changed in this version. Many of the main ideas such as frames,
processors, services and transports are still there but some things have changed
a bit.

- `Frame`s describe the basic units for processing. For example, text, image or
  audio frames. Or control frames to indicate a user has started or stopped
  speaking.

- `FrameProcessor`s process frames (e.g. they convert a `TextFrame` to an
  `ImageRawFrame`) and push new frames downstream or upstream to their linked
  peers.

- `FrameProcessor`s can be linked together. The easiest wait is to use the
  `Pipeline` which is a container for processors. Linking processors allow
  frames to travel upstream or downstream easily.

- `Transport`s are a way to send or receive frames. There can be local
  transports (e.g. local audio or native apps), network transports
  (e.g. websocket) or service transports (e.g. https://daily.co).

- `Pipeline`s are just a processor container for other processors.

- A `PipelineTask` know how to run a pipeline.

- A `PipelineRunner` can run one or more tasks and it is also used, for example,
  to capture Ctrl-C from the user.

## [0.0.8] - 2024-04-11

### Added

- Added `FireworksLLMService`.

- Added `InterimTranscriptionFrame` and enable interim results in
  `DailyTransport` transcriptions.

### Changed

- `FalImageGenService` now uses new `fal_client` package.

### Fixed

- `FalImageGenService`: use `asyncio.to_thread` to not block main loop when
  generating images.

- Allow `TranscriptionFrame` after an end frame (transcriptions can be delayed
  and received after `UserStoppedSpeakingFrame`).

## [0.0.7] - 2024-04-10

### Added

- Add `use_cpu` argument to `MoondreamService`.

## [0.0.6] - 2024-04-10

### Added

- Added `FalImageGenService.InputParams`.

- Added `URLImageFrame` and `UserImageFrame`.

- Added `UserImageRequestFrame` and allow requesting an image from a participant.

- Added base `VisionService` and `MoondreamService`

### Changed

- Don't pass `image_size` to `ImageGenService`, images should have their own size.

- `ImageFrame` now receives a tuple`(width,height)` to specify the size.

- `on_first_other_participant_joined` now gets a participant argument.

### Fixed

- Check if camera, speaker and microphone are enabled before writing to them.

### Performance

- `DailyTransport` only subscribe to desired participant video track.

## [0.0.5] - 2024-04-06

### Changed

- Use `camera_bitrate` and `camera_framerate`.

- Increase `camera_framerate` to 30 by default.

### Fixed

- Fixed `LocalTransport.read_audio_frames`.

## [0.0.4] - 2024-04-04

### Added

- Added project optional dependencies `[silero,openai,...]`.

### Changed

- Moved thransports to its own directory.

- Use `OPENAI_API_KEY` instead of `OPENAI_CHATGPT_API_KEY`.

### Fixed

- Don't write to microphone/speaker if not enabled.

### Other

- Added live translation example.

- Fix foundational examples.

## [0.0.3] - 2024-03-13

### Other

- Added `storybot` and `chatbot` examples.

## [0.0.2] - 2024-03-12

Initial public release.

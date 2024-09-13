# Changelog

All notable changes to **pipecat** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

- `DailyTransport.on_joined` event now returns the full session data instead of
  just the participant.

- `CartesiaTTSService` is now a subclass of `TTSService`.

- `DeepgramSTTService` is now a subclass of `STTService`.

- `WhisperSTTService` is now a subclass of `SegmentedSTTService`. A
  `SegmentedSTTService` is a `STTService` where the provided audio is given in a
  big chunk (i.e. from when the user starts speaking until the user stops
  speaking) instead of a continous stream.

### Fixed

- `StartFrame` should be the first frame every processor receives to avoid
  situations where things are not initialized (because initialization happens on
  `StartFrame`) and other frames come in resulting in undesired behavior.

### Performance

- `obj_id()` and `obj_count()` now use `itertools.count` avoiding the need of
  `threading.Lock`.

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

- Added `AysncFrameProcessor` and `AsyncAIService`.  Some services like
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

- Added missing * keyword separators in services.

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

- Added Daily transport event: `on_dialout_answered`.  See
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
  canceled.

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
  `on_dialout_error` and `on_dialout_warning`.  See
  https://reference-python.daily.co/api_reference.html#daily.EventHandler

## [0.0.21] - 2024-05-22

### Added

- Added vision support to Anthropic service.

- Added `WakeCheckFilter` which allows you to pass information downstream only
  if you say a certain phrase/word.

### Changed

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

# Changelog

All notable changes to **pipecat** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

- `SpeechmaticsTTSService` now reports the Pipecat version in the `sm-app` tag it
  sends to Speechmatics, matching `SpeechmaticsSTTService`. It previously sent the
  `speechmatics-rt` SDK version under the `pipecat/` label.

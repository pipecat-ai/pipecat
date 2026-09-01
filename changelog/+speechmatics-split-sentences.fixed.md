- Fixed a `SpeechmaticsSTTService` error where setting `split_sentences` raised
  `ValueError: "VoiceAgentConfig" object has no field "split_sentences"` and the
  service failed to construct.

- `TogetherTTSService` now sends its `language` setting to Together, and
  lowercases locale codes as the API requires. The setting was previously
  ignored, so every voice was synthesized with Together's default language.

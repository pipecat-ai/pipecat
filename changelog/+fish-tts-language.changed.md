- `FishAudioTTSService` now sends `Settings.language` as Fish Audio's BCP-47 `language`
  field, which controls how numbers, dates, units and symbols are spelled out before
  synthesis. Unset, Fish Audio keeps detecting the language of each chunk.

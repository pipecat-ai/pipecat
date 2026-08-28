- Eval scenario turns can play an audio file as the user instead of
  synthesizing text. A turn's `audio:` names a recording, resolved relative to
  the scenario file, in any format `soundfile` reads (WAV, MP3, FLAC, OGG, ...);
  multi-channel audio is downmixed to mono and the file keeps its own sample
  rate. `user:` is required alongside it and is what the recording says, so the
  judge and `text_contains` still have the turn's input. A scenario whose audio
  turns all name a file needs no `user.speech:` block.

- Fixed `ParallelPipeline` treating a lifecycle frame (`StartFrame`, `EndFrame`,
  `CancelFrame`) pushed internally by a branch processor as if it were the
  externally-arriving lifecycle frame completing synchronization. This could let
  the internal frame escape downstream before the real one (e.g. an `EndFrame`
  arriving before its `StartFrame`), corrupting lifecycle ordering and
  synchronization state.

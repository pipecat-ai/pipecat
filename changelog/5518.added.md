- `SmallestSTTService.Settings` gained `eou_timeout_ms`, which sets how much
  trailing silence (100–10000 ms) Pulse waits through before finalizing a
  transcript. Pulse's default of 800 ms applies when it's unset.

- `AzureSTTService.Settings` gained `segmentation_silence_timeout_ms`, which sets
  how much silence (100–5000 ms) Azure allows inside a phrase before it emits a
  final transcript. Azure's default of 500 ms applies when it's unset.

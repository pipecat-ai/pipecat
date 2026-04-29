Added `BandwidthFrameSerializer` for Bandwidth Programmable Voice WebSocket
media streaming, used with `FastAPIWebsocketTransport`. Supports bidirectional
PCMU 8kHz audio plus higher-fidelity PCM at 8/16/24kHz outbound, interruption
handling via `clear` events, and automatic call termination via the Bandwidth
Voice API when an `EndFrame` or `CancelFrame` is processed.

- Corrected the `ExotelFrameSerializer` docstring, which claimed support for
  "automatic call termination". Unlike the Twilio, Telnyx and Plivo serializers, it
  has no `auto_hang_up` parameter and ignores `EndFrame` and `CancelFrame`, so the
  call is only terminated when the transport closes the WebSocket. Added tests
  pinning that behavior.

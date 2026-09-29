- The SIP transport's receive jitter buffer is configurable:
  `SIPConnection(jitter_buffer_mode=..., jitter_buffer_ms=(min, max))`, or
  `SIP_JITTER_BUFFER=off|fixed:MIN-MAX|adaptive:MIN-MAX` with the development
  runner. The buffer holds media before the transport reads it, so its size is
  turn latency: a fixed buffer costs exactly its minimum on every packet and
  returns to it after any disturbance, while an adaptive one widens under
  jitter and does not narrow again for the rest of the call. The default is
  unchanged — the stack's fixed 100–200 ms.

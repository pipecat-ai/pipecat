Added Bandwidth support to `pipecat.runner.utils.parse_telephony_websocket`
and `create_transport`. Applications can now pass `runner_args` through the
runner abstraction with a `"bandwidth"` entry in the `transport_params` dict
instead of writing a custom WebSocket handler. Builds on the
`BandwidthFrameSerializer` introduced in #4387.

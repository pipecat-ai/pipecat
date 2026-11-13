# RTVI protocol specification

A machine-readable description of the RTVI protocol between a client and a Pipecat server: every message the server accepts from a client, every message it sends back, and the payload of each.

```
spec/
├── __init__.py              # the MESSAGES table and the document builder
├── 2.3.0/
│   └── rtvi.asyncapi.json   # the document of protocol 2.3.0
└── <version>/
    └── rtvi.asyncapi.json
```

Each `<version>/rtvi.asyncapi.json` is an [AsyncAPI 3.0](https://www.asyncapi.com/docs/reference/specification/v3.0.0) document, written from the server's point of view: the `receiveClientMessages` operation lists what clients send, and `sendServerMessages` what the server sends. Client SDKs generate their protocol code from the document of the version they implement.

The documents are generated, never edited by hand. The current version's document is built from `MESSAGES` in `__init__.py` and the models in `../models.py`; documents of earlier versions stay as they were when that version was current.

## What a message's entry says

Each message in `components.messages` has:

- `payload`: the schema of the whole message: `label` (always `"rtvi-ai"`), `type`, `id` when the message carries one, and `data`.
- `examples`: an example message.
- `x-rtvi-enabled-by`: for a message the server only sends when it is turned on, the `RTVIObserverParams` field that turns it on.
- `x-rtvi-min-client-version`: the lowest client protocol version the server sends the message to.
- `x-rtvi-deprecated`: for a deprecated message, what replaces it.

Payloads of client messages describe what the server accepts, and payloads of server messages what it sends, so a field the server leaves out when it has no value is optional.

## Changing the protocol

1. Change the models in `../models.py`, and where the server sends a message, the observer or processor that sends it. A message that clients older than the new version must not receive is gated with `_client_is_at_least([...])` in the observer.
2. Add a new message to `MESSAGES` in `__init__.py`, with an example and, where they apply, `enabled_by` and `min_client_version`. A new field on an existing message needs no change there, since the schema comes from the model.
3. For a new protocol version, bump `PROTOCOL_VERSION` in `../models.py`.
4. Regenerate the document:

   ```bash
   uv run python scripts/rtvi/write_rtvi_spec.py
   ```

   It writes `<PROTOCOL_VERSION>/rtvi.asyncapi.json`, and leaves the other versions' documents alone.

5. Run `uv run pytest tests/test_rtvi_spec.py`. It fails when the current version's document doesn't match the code, and when an example doesn't match its message's payload.

A change to the wire format without a version bump rewrites the current version's document, so keep changes to a released version backward compatible, such as a new optional field.


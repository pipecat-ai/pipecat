# RTVI protocol specification

A machine-readable description of the RTVI protocol between a client and a Pipecat server: every message the server accepts from a client, every message it sends back, and the payload of each.

```
spec/
├── __init__.py              # the MESSAGES and BEHAVIORS tables, and the document builder
├── released.json            # the protocol versions Pipecat has released
├── 2.3.0/
│   └── rtvi.asyncapi.json   # the document of protocol 2.3.0
└── <version>/
    └── rtvi.asyncapi.json
```

Each `<version>/rtvi.asyncapi.json` is an [AsyncAPI 3.0](https://www.asyncapi.com/docs/reference/specification/v3.0.0) document, written from the server's point of view: the `receiveClientMessages` operation lists what clients send, and `sendServerMessages` what the server sends. Client SDKs generate their protocol code from the document of the version they implement.

The documents are generated, never edited by hand. The current version's document is built from `MESSAGES` and `BEHAVIORS` in `__init__.py` and the models in `../models.py`; documents of earlier versions stay as they were when that version was current.

## What a message's entry says

Each message in `components.messages` has:

- `payload`: the schema of the whole message: `label` (always `"rtvi-ai"`), `type`, `id` when the message carries one, and `data`.
- `examples`: an example message.
- `x-rtvi-enabled-by`: for a message the server only sends when it is turned on, the `RTVIObserverParams` field that turns it on.
- `x-rtvi-min-client-version`: the lowest client protocol version the server sends the message to.
- `x-rtvi-deprecated`: for a deprecated message, what replaces it.

Payloads of client messages describe what the server accepts, and payloads of server messages what it sends, so a field the server leaves out when it has no value is optional.

## Behaviors

The document's `x-rtvi-behaviors` lists the rules the server follows across messages, beyond the shape of each one: what answers a request and with which `id`, what `bot-ready` says about the version, which messages the observer params turn off, and how the server holds the bot's spoken text until the bot speaks and drops it on an interruption. Each behavior has an `id`, a `description`, and the types of the `messages` it involves.

## Changing the protocol

1. Change the models in `../models.py`, and where the server sends a message, the observer or processor that sends it. A message that clients older than the new version must not receive is gated with `_client_is_at_least([...])` in the observer.
2. Add a new message to `MESSAGES` in `__init__.py`, with an example and, where they apply, `enabled_by` and `min_client_version`. A new field on an existing message needs no change there, since the schema comes from the model. A new rule across messages goes in `BEHAVIORS`, with its test in `tests/test_rtvi_conformance.py`.
3. For a new protocol version, bump `PROTOCOL_VERSION` in `../models.py`.
4. Regenerate the document:

   ```bash
   uv run python scripts/rtvi/write_rtvi_spec.py
   ```

   It writes `<PROTOCOL_VERSION>/rtvi.asyncapi.json`, and leaves the other versions' documents alone.

5. Run `uv run pytest tests/test_rtvi_spec.py`. It fails when the current version's document doesn't match the code, and when an example doesn't match its message's payload.

A released version's document never changes, since clients implement the version from it: any change to the wire format after a release, even a new optional field, is a new protocol version.

## Releases

`released.json` records each protocol version a Pipecat release has shipped, with the SHA-256 of its document and the Pipecat release that first shipped it:

```json
{ "2.3.0": { "sha256": "57e5…", "pipecat": "1.15.0" } }
```

The changelog workflow (`.github/workflows/generate-changelog.yml`) runs `scripts/rtvi/record_released_spec.py --version <release>` at release-prep, so the current protocol version is recorded in the release PR. A version already recorded stays as it is. `tests/test_rtvi_spec.py` fails when a recorded document has changed, and when the current version is recorded and its document no longer matches the code, which means it's time to bump `PROTOCOL_VERSION`.

## Testing the server against the documents

`tests/test_rtvi_conformance.py` checks that Pipecat speaks each version as its document says. It runs the real `RTVIProcessor` and `RTVIObserver` through a session with a client of each version that has a document, and checks every message they put on the wire:

- It is a message the document lists for the server to send.
- Its payload matches the document's schema. The check reads the JSON document, not the Python models.
- The client is at least the message's `x-rtvi-min-client-version`. Sessions with clients just below each minimum check that those messages are left out.

For messages a client sends, the server must accept each document's example without an error response.

Each behavior in a document has a test named after its `id`, `test_<id>` with underscores, in `TestRTVIBehaviors`, and a behavior without one fails the test.

The session must make the server send every message of the current version, so a message added to the spec fails the test until `_run_session` in the test makes the server send it.

Schemas allow fields they don't list, since a client ignores fields it doesn't know. A field added to a model without regenerating the document passes this test, and `tests/test_rtvi_spec.py` catches it.

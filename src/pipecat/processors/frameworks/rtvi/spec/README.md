# RTVI protocol specification

RTVI (Real-Time Voice Interface) is the protocol a client app, such as a web page or a mobile app, and a Pipecat bot use to talk to each other during a session. Next to the audio and video, the session's transport carries RTVI messages as data messages: the client says it's ready, sends text or files, and answers requests; the bot says when the user and the bot start and stop speaking, what they said, and what the bot is doing.

Every message is a JSON object like this one:

```json
{ "label": "rtvi-ai", "type": "send-text", "id": "msg_1", "data": { "content": "Hi!" } }
```

`label` is always `"rtvi-ai"`, `type` says which message it is, and `data` carries its content. Every message a client sends has an `id`, and the server's answer to it repeats that `id`.

This folder describes every message, for each version of the protocol, in a form a program can read: client libraries can generate their code from it, and anyone can build a client or a server from it without reading Pipecat's code.

```
spec/
├── __init__.py              # the MESSAGES and BEHAVIORS tables, and the document builder
├── released.json            # the protocol versions Pipecat has released
├── 2.3.0/
│   └── rtvi.asyncapi.json   # the document of protocol 2.3.0
└── <version>/
    └── rtvi.asyncapi.json
```

## Reading a document

Each `<version>/rtvi.asyncapi.json` is an [AsyncAPI 3.0](https://www.asyncapi.com/docs/reference/specification/v3.0.0) document. AsyncAPI is a standard format for describing message-based APIs, and its tools can render a document as a web page or generate code from it.

The document is written from the server's point of view. Its `receiveClientMessages` operation lists the messages a client sends, and `sendServerMessages` the messages the server sends. Each message in `components.messages` has:

- `summary`: what the message means and when it is sent.
- `payload`: the schema of the whole message, as [JSON Schema](https://json-schema.org/): `label`, `type`, `id` when the message carries one, and `data`. Each field of `data` has a description.
- `examples`: an example message.
- `x-rtvi-enabled-by`: for a message the server only sends when a setting turns it on, the setting's name. The settings are the fields of Pipecat's `RTVIObserverParams`.
- `x-rtvi-min-client-version`: the lowest client protocol version the server sends the message to. A client tells the server its version in `client-ready`.
- `x-rtvi-deprecated`: for a deprecated message, what replaces it.

The schema of a client message says what the server accepts, and the schema of a server message what the server sends. A field the server leaves out when it has no value is optional.

The document's `x-rtvi-behaviors` lists the rules the server follows across messages, beyond the shape of each one: which message answers a request and with which `id`, which version `bot-ready` reports, which messages the settings turn off, and how the server holds the bot's text until the bot speaks it and drops it when the user interrupts. Each behavior has an `id`, a `description`, and the types of the `messages` it involves.

## Changing the protocol

The documents are generated, never edited by hand. The current version's document is built from `MESSAGES` and `BEHAVIORS` in `__init__.py` and from the models in `../models.py`: each message's schema comes from its model, and each field's description from the `Parameters:` section of the model's docstring.

1. Change the models in `../models.py`, and the code that sends or handles the message: `RTVIObserver` in `../observer.py` sends most server messages, and `RTVIProcessor` in `../processor.py` handles client messages. A message that clients older than the new version must not receive is gated with `_client_is_at_least([...])` in the observer.
2. Add a new message to `MESSAGES` in `__init__.py`, with a description, an example and, where they apply, `enabled_by` and `min_client_version`. A new field on an existing message needs no change there, since the schema comes from the model; describe the field in the model's docstring. A new rule across messages goes in `BEHAVIORS`, with its test in `tests/test_rtvi_conformance.py`.
3. If the change is to the wire format of a released version, even a new optional field, bump `PROTOCOL_VERSION` in `../models.py`: see [Releases](#releases).
4. Regenerate the document:

   ```bash
   uv run python scripts/rtvi/write_rtvi_spec.py
   ```

   It writes `<PROTOCOL_VERSION>/rtvi.asyncapi.json` and leaves the other versions' documents alone.

5. Run `uv run pytest tests/test_rtvi_spec.py tests/test_rtvi_conformance.py`.

## Releases

Clients implement a protocol version from its document, so once a Pipecat release has shipped a version, its wire format never changes: what the server accepts and sends, and how. Any change to it is a new protocol version. The document's descriptions and examples can still be improved.

A released version's schemas can also describe its wire format more precisely, where the server already sends exactly what the more precise schema says, such as `metrics` data typed by kind rather than left as any object. `tests/test_rtvi_conformance.py` must show the server's messages match it. Since the hash covers the schemas, update the version's `contract_sha256` in `released.json` in the same change, by hand: the record script never overwrites a version it has recorded.

`released.json` records each protocol version a Pipecat release has shipped, with a hash of its wire format and the Pipecat release that first shipped it:

```json
{ "2.3.0": { "contract_sha256": "56d5…", "pipecat": "1.15.0" } }
```

The hash covers the document without its descriptions, summaries, titles, examples and tags, so rewording a description doesn't change it, and a change to the wire format does.

When a release is prepared, the changelog workflow (`.github/workflows/generate-changelog.yml`) runs `scripts/rtvi/record_released_spec.py --version <release>`, which adds the current protocol version to `released.json` in the release PR. A version already recorded stays as it is. `tests/test_rtvi_spec.py` fails when the wire format of a recorded version has changed, which means it's time to bump `PROTOCOL_VERSION`.

## How the tests use the documents

`tests/test_rtvi_spec.py` checks the current version's document against the code: it fails when the document is out of date, when an example doesn't match its message's schema, when a field has no description, and when a released version's wire format has changed.

`tests/test_rtvi_conformance.py` checks that Pipecat behaves as each version's document says. It runs Pipecat's real `RTVIProcessor` and `RTVIObserver`, holds a session with a client of each version that has a document, and checks every message the server sends:

- It is a message the document lists for the server to send.
- Its payload matches the document's schema. The check reads the JSON document, not Pipecat's Python models.
- The client's version is at least the message's `x-rtvi-min-client-version`. Sessions with clients just below each minimum check that those messages are left out.

For the messages a client sends, the example of each must match its schema in the JSON document, and the server must accept it without an error. The server must also answer an `error-response` to each message the document forbids, built from the message's schema: one without `data` when `data` is required, one missing each required field, and one with each field of the wrong type. `_LENIENT` in the test lists the few the server accepts anyway, with the reason. Each behavior in a document has a test named after its `id` (`test_<id>`, with underscores) in `TestRTVIBehaviors`, and a behavior without a test fails.

The session must make the server send every message of the current version, so a message added to `MESSAGES` fails until `_run_session` in the test makes the server send it.

Schemas allow fields they don't list, since a client ignores fields it doesn't know. So a field added to a model without regenerating the document passes the conformance test, and `tests/test_rtvi_spec.py` catches it.

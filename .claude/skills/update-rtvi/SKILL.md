---
name: update-rtvi
description: Carry a change to RTVI's models or message handling through the protocol specification, its version, and its tests
---

Carry a change to the RTVI protocol through everything that describes it. The developer changes the code: the models in `models.py`, and what `RTVIObserver` sends or `RTVIProcessor` accepts. This skill decides whether the change needs a new protocol version, gates it for older clients, updates the specification and its tests, regenerates the document, and previews the docs.

`src/pipecat/processors/frameworks/rtvi/spec/README.md` is the source of truth for how the specification works: what the document holds, how it is built, and the rules for released versions. Read it first, and follow it where this skill and it differ. This skill applies the README to one change; it doesn't restate it.

Changelog fragments are left to the `changelog` skill, and the docs PR to `update-docs`, which runs when the PR merges.

## Arguments

```
/update-rtvi [DOCS_PATH]
```

- `DOCS_PATH` (optional): the docs repository root, for the preview in Step 7. Without it, skip the preview and say so in the summary.

## Instructions

Paths below are relative to `src/pipecat/processors/frameworks/rtvi/` unless they start at the repository root.

### Step 1: Find what changed on the wire

Read the README, then the diff of the branch against `main`, including uncommitted changes:

```bash
git diff main -- src/pipecat/processors/frameworks/rtvi/ src/pipecat/frames/
```

List each change a client would see, by kind:

- **Added**: a message, a field, a value of an enum or a literal, an output type.
- **Loosened**: a required field made optional, a type widened.
- **Tightened**: an optional field made required, a type narrowed, a value no longer accepted.
- **Removed or renamed**: a message, a field, a value.
- **Behavior**: when a message is sent, which message answers which, which `id` it carries, what a setting turns off, the order of messages. Check the README's description of `x-rtvi-behaviors` and the existing `BEHAVIORS` to see whether one of them changed.
- **Deprecated**: a message or field that still works, with a replacement.

Read the observer and processor code too, not just the models: a model change that no code sends isn't on the wire, and a code change can put something new on the wire with no model change.

If the list is empty (logging, internals, refactoring), stop and report "no protocol change". Otherwise show the list to the developer before editing anything.

### Step 2: Decide the protocol version

Read `PROTOCOL_VERSION` in `models.py` and `spec/released.json`.

- **`PROTOCOL_VERSION` isn't in `released.json`**: it hasn't shipped, so the change joins it. No bump.
- **It is, and the change is Added, Loosened, Behavior or Deprecated**: bump the minor version, `X.Y.0` to `X.(Y+1).0`.
- **It is, and the change is Tightened or Removed or renamed**: this breaks existing clients and needs a major version. Stop and ask the developer: a major bump changes which clients the server serves (see `LEGACY_SUPPORTED_MAJOR` and the client-ready version check), and usually the change can be made compatible instead, by adding the new form next to the old one and deprecating the old.
- **It is, and the change only makes a schema more precise** while the server already sends exactly that: the README's rule for released versions applies. No bump; the version's `contract_sha256` in `released.json` is updated by hand. Confirm with the developer before touching `released.json`, and show the conformance test that proves the server sends what the new schema says.

State the decision and its reason. To bump, change `PROTOCOL_VERSION` in `models.py`; the new version's document goes in a new folder in Step 6, and the released folders are never edited.

### Step 3: Gate new output for older clients

Every Added item the server sends is new to clients of the earlier versions, and they must not receive it. For each:

- **A new message**: give its `MESSAGES` entry `min_client_version="<new version>"`, and send it only when `self._client_is_at_least([X, Y, 0])` in the observer. Use a `_client_supports_<feature>` property, as the observer's existing gates do, when more than one place checks it.
- **A new field, value or output type in an existing message**: gate it in the observer the same way, leaving it out or falling back to what older clients get. The document can't mark a field's minimum version, so add a test that a client of the previous version doesn't get it, next to the observer's tests of that message (`tests/test_rtvi_observer_*.py`).

A new client message needs no gate on the server: clients only send it when `bot-ready` reports a version that has it. Check that the processor still accepts every message of the earlier versions.

### Step 4: Update the specification

In `spec/__init__.py`:

- **A new server message**: add it to `MESSAGES` with `_server(Model, description, example=...)`, plus `enabled_by=` (the `RTVIObserverParams` field that turns it on, when one does), `min_client_version=`, and `details=` when how it's sent needs more than the description. Use `data_type=` only when the spec must describe `data` more precisely than the runtime model does.
- **A new client message**: add it with `_client(type, description, data=Model, example=...)`.
- **A new message's group**: add its type to the right group in `GROUPS`. Every message belongs to exactly one group.
- **A new rule across messages**: add an `RTVIBehaviorSpec` to `BEHAVIORS`, with an `id`, a description, and the types of the messages it involves. A changed rule changes its description.
- **A deprecated message**: set `deprecated=` to what replaces it, and follow AGENTS.md's deprecation rules for the model and the observer, including `scripts/deprecations/deprecations.json`.

In `models.py`, every new field gets its description in the `Parameters:` section of its model's docstring, which the document copies; a new model gets a docstring.

The descriptions are what client developers read, in the document and on the docs site. Write each so someone new to Pipecat understands it: what the message or field means, when it is sent, what its values are. Say it in the terms of the protocol, not of Pipecat's internals: name the setting that turns a message on, not the frame that triggers it. Match the length and tone of the entries next to it.

### Step 5: Update the tests

In `tests/test_rtvi_conformance.py`:

- **A new server message**: make `_run_session` send it, and turn it on in `_EVERYTHING` if a setting is off by default. `test_the_session_sends_every_message_of_the_current_version` fails until it does. Sessions with clients just below each `min_client_version` are generated from the document, so the gate from Step 3 is tested without more work.
- **A new client message**: its example in `MESSAGES` is sent by `test_examples_are_accepted`. Add a test of what the server does with it when that is more than accepting it.
- **A new behavior**: add `test_<id>` (dashes to underscores) to `TestRTVIBehaviors`; `test_every_behavior_has_a_test` fails without it. A changed behavior updates its test.

Test the observer or processor change itself in their own test files, as for any code change.

### Step 6: Regenerate and verify

```bash
uv run python scripts/rtvi/write_rtvi_spec.py
uv run pytest tests/test_rtvi_spec.py tests/test_rtvi_conformance.py
uv run ruff check && uv run ruff format --check
uv run pyright src/pipecat/processors/frameworks/rtvi/ tests/test_rtvi_spec.py tests/test_rtvi_conformance.py
```

The script writes `spec/<PROTOCOL_VERSION>/rtvi.asyncapi.json`, creating the folder on a bump. Check that `git status` shows no change to a released version's folder.

When a test fails, find what it means before changing anything; the README's last section says what each one checks. In particular:

- **A released version's contract hash changed**: the change reached a released version's wire format. Go back to Step 2; never update the hash to make the test pass, except under the rule for more precise schemas, with the developer's confirmation.
- **A message doesn't match its schema**: the server sends something its document doesn't allow. Fix the code or the model, not the test.

Never edit a generated document by hand, and never loosen a conformance test to make it pass.

### Step 7: Preview the docs

With `DOCS_PATH`, generate the reference in a temporary worktree of the docs repo, so the docs checkout is left alone:

```bash
PREVIEW=$(mktemp -d)/docs
git -C DOCS_PATH worktree add --detach "$PREVIEW" HEAD
python3 "$PREVIEW/scripts/gen-rtvi-reference.py" --pipecat .
git -C "$PREVIEW" diff --stat
```

The generator writes each version's reference page, the RTVI group of `docs.json`, and the Protocol Reference list of the RTVI standard page. Read the changed page's diff and check that every item from Step 1 appears, with readable descriptions.

On a bump, the generator also reports that the Protocol Versions table has no row for the new version. Draft that row for the summary, in the table's format: the version, the Pipecat release that will ship it (the next release, to be confirmed), the client-js release (unknown until it's released), and what changed, in the style of the existing rows. Note that the Note at the top of the RTVI standard page names the current version and needs the same update.

Then remove the worktree:

```bash
git -C DOCS_PATH worktree remove --force "$PREVIEW"
```

### Step 8: Summary

Report, for the PR description:

```markdown
## RTVI protocol

- **Version**: <unchanged X.Y.Z (unreleased) | X.Y.Z → X.Y'.Z'>, because <reason>.
- **Wire changes**: <the Step 1 list>.
- **Gated for older clients**: <each item and its minimum version, or "none">.
- **Specification**: <MESSAGES, GROUPS, BEHAVIORS entries added or changed>.
- **Tests**: <what was added>; spec and conformance tests pass.

### Docs preview

<the changed messages on the reference page, or "skipped: no DOCS_PATH">

Protocol Versions row for the docs PR:

| [X.Y.Z](/api-reference/standards/rtvi/X.Y.Z) | <next release> | <TBD> | <changes> |
```

List anything left for the developer: a question from Step 2, a description you weren't sure of, client libraries that need the change.

## Guardrails

- Never edit a released version's folder under `spec/`, or a generated document.
- Never change `released.json` without the developer's confirmation.
- Never change the docs repository; Step 7 works in a temporary worktree.
- Never bump to a new major version without the developer's decision.

## Checklist

- [ ] The README was read, and every wire change listed with its kind
- [ ] The version decision was stated with its reason
- [ ] Every new server output is gated for clients of earlier versions, and tested
- [ ] `MESSAGES`, `GROUPS`, `BEHAVIORS` and the models' docstrings describe the change in terms a newcomer understands
- [ ] The conformance session sends every new message, and each new behavior has its test
- [ ] The document was regenerated, no released folder changed, and the spec and conformance tests, ruff and pyright pass
- [ ] The docs preview was checked and the Protocol Versions row drafted, or the preview was reported as skipped

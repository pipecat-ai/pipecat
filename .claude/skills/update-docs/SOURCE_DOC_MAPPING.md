# Source-to-Doc Mapping

Maps pipecat source files to their documentation pages. Source paths are relative to `src/pipecat/`. Doc paths are relative to `DOCS_PATH`.

Doc paths in this file are candidates. Confirm each exists in `DOCS_PATH` before editing it; if it doesn't exist, fall through to the Search section.

## Scope

Every `.py` file under `src/pipecat/` is in scope. The package ships public API
well beyond the per-provider service files — frames, workers, the bus, the eval
harness, the CLI, the runner, and the service base classes are all documented
somewhere on the site.

Exclude only:

- `src/pipecat/tests/**` (test helpers)
- `__pycache__/`, `*.pyc`, `py.typed`
- `__init__.py` files that only re-export names defined elsewhere

Changes outside `src/pipecat/` — examples, CI config, the docs directory — don't
trigger doc updates on their own.


## Non-standard locations

These source paths don't follow the standard `services/{provider}/{type}.py` → `api-reference/server/services/{type}/{provider}.mdx` pattern. Use the doc page below as the candidate path.

| Source path                                 | Doc page                                                                                           |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `services/google/vertex/llm.py`             | `api-reference/server/services/llm/google-vertex.mdx`                                              |
| `services/google/llm.py`                    | `api-reference/server/services/llm/google.mdx` (shared base; also affects `llm/google-vertex.mdx`) |
| `services/google/gemini_live/**`            | `api-reference/server/services/s2s/gemini-live.mdx`                                                |
| `services/google/gemini_live/vertex/llm.py` | `api-reference/server/services/s2s/gemini-live-vertex.mdx`                                         |
| `services/aws/nova_sonic/**`                | `api-reference/server/services/s2s/aws.mdx`                                                        |
| `services/ultravox/**`                      | `api-reference/server/services/s2s/ultravox.mdx`                                                   |
| `services/grok/realtime/**`                 | `api-reference/server/services/s2s/grok.mdx`                                                       |
| `services/openai/realtime/**`               | `api-reference/server/services/s2s/openai.mdx`                                                     |
| `services/openai/responses/llm.py`          | `api-reference/server/services/llm/openai-responses.mdx`                                           |
| `processors/frameworks/rtvi.py`             | `api-reference/server/rtvi/rtvi-processor.mdx` and `api-reference/server/rtvi/rtvi-observer.mdx`   |
| `processors/idle_frame_processor.py`        | `api-reference/server/pipeline/pipeline-idle-detection.mdx`                                        |
| `pipeline/worker.py`                        | `api-reference/server/pipeline/pipeline-worker.mdx`                                                |
| `pipeline/runner.py`                        | `api-reference/server/utilities/runner/guide.mdx`                                                  |
| `transports/base_transport.py`              | `api-reference/server/services/transport/transport-params.mdx`                                     |
| `flows/types.py`                            | `api-reference/pipecat-flows/types.mdx`                                                          |
| `flows/manager.py`                          | `api-reference/pipecat-flows/flow-manager.mdx`                                                   |
| `flows/actions.py`                          | `api-reference/pipecat-flows/flow-manager.mdx` and `api-reference/pipecat-flows/types.mdx`       |
| `flows/adapters.py`                         | `api-reference/pipecat-flows/overview.mdx`                                                       |
| `flows/exceptions.py`                       | `api-reference/pipecat-flows/exceptions.mdx`                                                     |

## Base classes

A base class is not internal. Its constructor parameters, event handlers, and
behavior are public API that every service inheriting from it exposes, documented
in guides and concept pages rather than on a per-provider reference page.

| Source path                      | Doc page                                                                                                      |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `services/tts_service.py`        | `pipecat/learn/text-to-speech.mdx`                                                                            |
| `services/stt_service.py`        | `pipecat/learn/speech-to-text.mdx`                                                                            |
| `services/llm_service.py`        | `pipecat/learn/llm.mdx` and `pipecat/learn/function-calling.mdx`                                              |
| `services/websocket_service.py`  | `api-reference/server/events/service-events.mdx`                                                              |
| `services/ai_service.py`         | `api-reference/server/events/service-events.mdx`                                                              |
| `serializers/base_serializer.py` | `api-reference/server/services/serializers/introduction.mdx`                                                  |
| `transports/base_input.py`       | `api-reference/server/services/transport/transport-params.mdx`                                                |
| `transports/base_output.py`      | `api-reference/server/services/transport/transport-params.mdx`                                                |
| `pipeline/pipeline.py`           | `pipecat/learn/pipeline.mdx`                                                                                  |
| `processors/frame_processor.py`  | `pipecat/fundamentals/custom-frame-processor.mdx` and `api-reference/server/events/frame-processor-events.mdx` |

Several of these carry documented parameters with no reference page of their own.
Where a change fits none of the pages above, report it as a missing-page gap in
SKILL.md Step 8 rather than skipping it.

### What to document from a base class

Most of a base class is framework machinery. A name without a leading underscore
doesn't make it public — `process_frame`, `push_frame`, and `tts_process_generator`
are all machinery. Apply this test instead:

**Can someone change or observe this without subclassing?**

| Kind                                                                | Verdict                                     |
| ------------------------------------------------------------------- | ------------------------------------------- |
| Constructor parameter that changes behavior                         | Document                                    |
| Event handler                                                       | Document                                    |
| Method called on a live instance (`set_model`, `set_voice`)         | Document                                    |
| Only meaningful when implementing `run_tts` / `run_stt` / `setup()` | Skip — it's the subclass contract           |
| Anything else                                                       | Skip                                        |

The subclass contract is a real audience, but it lives in the pipecat repo
alongside `COMMUNITY_INTEGRATIONS.md`, not on the docs site. A base-class change
that touches only that contract is a legitimate no-op — say so, naming the
methods, rather than editing a guide.

Worked example: of `TTSService`'s 19 constructor parameters, `push_text_frames`,
`push_stop_frames`, `push_start_frame`, and `reuse_context_id_within_turn` exist
so `run_tts` implementations don't have to do that work themselves. They fail the
test. `max_consecutive_zero_audio_contexts` passes it — it decides whether a
silent provider gets written off mid-call.

A deprecated parameter gets a deprecation notice and nothing more. Don't explain
a mechanism that no longer runs: `pause_watchdog_timeout_s` is documented in
source as "Unused", so its doc entry says it does nothing and is removed in
2.0.0.

### Inherited parameters belong to the guide, not the provider page

A provider page documents what that provider **adds or overrides**. Parameters
inherited from a base class are documented once, in the guide's "Base Class
Configuration" section, and left out of the per-provider pages.

Copying them onto provider pages doesn't scale: `text_aggregation_mode` reached
15 of 53 TTS pages that way, which means 15 copies to keep current and 38 pages
where the parameter appears not to exist. When a base-class parameter changes,
edit the guide — don't fan the change out.

## Skip list

These files never trigger doc updates. Keep this list short — it is for files
with no observable public surface, not for files that are merely hard to place.

| Pattern                             | Reason                                    |
| ----------------------------------- | ----------------------------------------- |
| `services/image_service.py`         | Abstract interface only, no public params |
| `services/vision_service.py`        | Abstract interface only, no public params |
| `services/settings.py`              | Internal plumbing                         |
| `services/aws/agent_core.py`        | Internal                                  |
| `services/aws/sagemaker/**`         | No doc page                               |
| `transports/websocket/client.py`    | No doc page                               |
| `serializers/protobuf.py`           | Internal wire format                      |
| `processors/audio/vad_processor.py` | No doc page                               |
| `tests/**`                          | Test helpers                              |

## Pattern matching

For files not in the tables above, apply these patterns. Convert underscores to hyphens in provider names for doc filenames.

| Source pattern                    | Doc pattern                                                       |
| --------------------------------- | ----------------------------------------------------------------- |
| `services/{provider}/stt*.py`     | `api-reference/server/services/stt/{provider}.mdx`                |
| `services/{provider}/tts*.py`     | `api-reference/server/services/tts/{provider}.mdx`                |
| `services/{provider}/llm*.py`     | `api-reference/server/services/llm/{provider}.mdx`                |
| `services/{provider}/image*.py`   | `api-reference/server/services/image-generation/{provider}.mdx`   |
| `services/{provider}/video*.py`   | `api-reference/server/services/video/{provider}.mdx`              |
| `services/{provider}/realtime/**` | `api-reference/server/services/s2s/{provider}.mdx`                |
| `transports/{name}/**`            | `api-reference/server/services/transport/{name}.mdx`              |
| `serializers/{name}.py`           | `api-reference/server/services/serializers/{name}.mdx`            |
| `observers/**`                    | `api-reference/server/utilities/observers/` (match by class name) |
| `audio/vad/**`                    | `api-reference/server/utilities/audio/` (match by class name)     |
| `audio/filters/**`                | `api-reference/server/utilities/audio/` (match by class name)     |
| `audio/mixers/**`                 | `api-reference/server/utilities/audio/` (match by class name)     |
| `processors/audio/**`             | `api-reference/server/utilities/audio/` (match by class name)     |
| `processors/filters/**`           | `api-reference/server/utilities/filters/` (match by class name)   |
| `workers/**`                      | `api-reference/server/workers/` (match by class name)             |
| `bus/**`                          | `api-reference/server/bus/` (match by class name)                 |
| `turns/**`                        | `api-reference/server/utilities/turn-management/`                 |
| `frames/frames.py`                | `api-reference/server/frames/` (match by frame class name)        |
| `evals/**`                        | `pipecat/evals/` and `api-reference/cli/eval.mdx`                 |
| `cli/**`                          | `api-reference/cli/` (match by command name)                      |
| `runner/**`                       | `api-reference/server/utilities/runner/guide.mdx`                 |
| `metrics/**`                      | `pipecat/fundamentals/metrics.mdx`                                |
| `adapters/**`                     | the LLM page for that provider under `api-reference/server/services/llm/` |
| `utils/**`                        | match by class or function name across `api-reference/` and `pipecat/`    |

A frame class is documented on the page matching its base class:
`SystemFrame` subclasses on `frames/system-frames.mdx`, `ControlFrame` subclasses
on `frames/control-frames.mdx`, and so on. When a frame changes base class, move
its entry to the page for its new base and fix any prose that explains its
ordering or interruption behavior.

A pattern result is only valid if the file exists in `DOCS_PATH`. If it doesn't exist, fall through to the Search section before treating the file as unmapped.

## Search

For files that match no pattern above, or whose candidate doesn't exist in `DOCS_PATH`:

1. Extract the main class name(s) from the source file.
2. Grep `DOCS_PATH` for that class name: `grep -rl "ClassName" DOCS_PATH/api-reference/ DOCS_PATH/pipecat/`.
3. If a page is found, use it. If nothing is found, the file is **unmapped** — report it in SKILL.md Step 8.

## Section vocabulary

Service pages are built from these sections. Check each against the source when
the corresponding construct changed:

| Section | Built from | Form |
| --- | --- | --- |
| Configuration | the `__init__` signature | `<ParamField>` entries |
| InputParams | the `InputParams(BaseModel)` class fields | markdown table: `\| Parameter \| Type \| Default \| Description \|` |
| Event Handlers | `_register_event_handler` calls and handler definitions | event table plus example |
| Usage | current class names and import paths | code block |
| Notes | behavioral caveats | prose |

**InputParams** is the one most often out of step: match the field names, types,
and defaults to the `InputParams(BaseModel)` class rather than to the
constructor, which usually takes the whole object.

## Guide directories

Prose that cites pipecat API lives in:

- `pipecat/learn/` — conceptual tutorials (pipeline, LLM, STT, TTS, etc.)
- `pipecat/fundamentals/` — practical how-tos (metrics, recording, transcripts, etc.)
- `pipecat/features/` — feature-specific guides (Gemini Live, OpenAI audio, WhatsApp, etc.)
- `pipecat/telephony/` — telephony integration guides (Twilio, Plivo, Telnyx, etc.)
- `pipecat/flows/` — Pipecat Flows guides (nodes-and-messages, functions, context-strategies, state-management, actions); check these when `src/pipecat/flows/**` changed

## New pages

### Location and template

Create the new `.mdx` file under
`DOCS_PATH/api-reference/server/services/{category}/{provider}.mdx` using this
structure:

````
---
title: "Service Name"
description: "Brief description"
---

## Overview

[Description from class docstring or source analysis]

<CardGroup cols={2}>
  [Cards for API reference and examples if available]
</CardGroup>

## Installation

```bash
uv add "pipecat-ai[package-name]"
```

## Prerequisites

[Environment variables and account setup]

## Configuration

[ParamField entries for constructor params]

## InputParams

[Table of InputParams fields, if the service has them]

## Usage

### Basic Setup

```python
[Minimal working example]
```

## Notes

[Important caveats]

## Event Handlers

[Event table and example code]
````

### Registration — both are required

A page that exists but isn't registered is invisible. Do both.

**1. `docs.json` navigation.** Add the path without the `.mdx` extension, in the
matching group under Services:

| Category | Group |
| --- | --- |
| STT | `Speech-to-Text` |
| TTS | `Text-to-Speech` |
| LLM | `LLM` |
| S2S | `Speech-to-Speech` |
| Transport | `Transport` |
| Serializer | `Serializers` |
| Image generation | `Image Generation` |
| Video | `Video` |
| Memory | `Memory` |
| Vision | `Vision` |
| Analytics | `Analytics & Monitoring` |

Insert **alphabetically** within the group's `pages` array.

**2. `supported-services.mdx`.** Add a row to the matching category table in
`DOCS_PATH/api-reference/server/services/supported-services.mdx`:

```
| [DisplayName](/api-reference/server/services/{category}/{provider}) | `uv add "pipecat-ai[package]"` |
```

- **DisplayName** — the human-readable name ("ElevenLabs", "AWS Polly", "Google Gemini")
- **package** — from the service's `pyproject.toml` extras or its import pattern; a service in `src/pipecat/services/foo/` is typically `foo`. Use `No dependencies required` when it needs none.

Insert **alphabetically**, matching the column alignment of existing rows.

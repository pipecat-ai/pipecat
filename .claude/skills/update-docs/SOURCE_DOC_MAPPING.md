# Source-to-Doc Mapping

Maps pipecat source files to their documentation pages. Source paths are relative to `src/pipecat/`. Doc paths are relative to `DOCS_PATH`.

Doc paths in this file are candidates. Confirm each exists in `DOCS_PATH` before editing it; if it doesn't exist, fall through to the Search section.

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

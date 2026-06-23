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

## Skip list

These files should never trigger doc updates.

| Pattern                              | Reason                               |
| ------------------------------------ | ------------------------------------ |
| `services/ai_service.py`             | Internal base class                  |
| `services/stt_service.py`            | Internal base class                  |
| `services/tts_service.py`            | Internal base class                  |
| `services/llm_service.py`            | Internal base class                  |
| `services/websocket_service.py`      | Internal base class                  |
| `services/image_service.py`          | Internal base class                  |
| `services/vision_service.py`         | Internal base class                  |
| `services/settings.py`               | Internal                             |
| `services/aws/agent_core.py`         | Internal                             |
| `services/aws/sagemaker/**`          | No doc page                          |
| `transports/base_input.py`           | Internal base class                  |
| `transports/base_output.py`          | Internal base class                  |
| `transports/websocket/client.py`     | No doc page                          |
| `serializers/base_serializer.py`     | Internal base class                  |
| `serializers/protobuf.py`            | Internal                             |
| `processors/audio/vad_processor.py`  | No doc page                          |
| `pipeline/pipeline.py`               | Core architecture, not a service doc |

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

A pattern result is only valid if the file exists in `DOCS_PATH`. If it doesn't exist, fall through to the Search section before treating the file as unmapped.

## Search

For files that match no pattern above, or whose candidate doesn't exist in `DOCS_PATH`:

1. Extract the main class name(s) from the source file.
2. Grep `DOCS_PATH` for that class name: `grep -rl "ClassName" DOCS_PATH/api-reference/ DOCS_PATH/pipecat/`.
3. If a page is found, use it. If nothing is found, the file is **unmapped** — report it in SKILL.md Step 8.

# Source-to-Doc Mapping

Maps pipecat source files to their documentation pages. Source paths are relative to `src/pipecat/`. Doc paths are relative to `DOCS_PATH`.

## Name mismatches

These source paths don't follow the standard `services/{provider}/{type}.py` → `server/services/{type}/{provider}.mdx` pattern.

| Source path | Doc page |
|---|---|
| `services/google/llm.py` | `server/services/llm/gemini.mdx` |
| `services/google/llm_vertex.py` | `server/services/llm/google-vertex.mdx` |
| `services/google/google.py` | (shared base — check which services use it) |
| `services/google/gemini_live/**` | `server/services/s2s/gemini-live.mdx` |
| `services/google/gemini_live/llm_vertex.py` | `server/services/s2s/gemini-live-vertex.mdx` |
| `services/aws_nova_sonic/**` | `server/services/s2s/aws.mdx` |
| `services/ultravox/**` | `server/services/s2s/ultravox.mdx` |
| `services/grok/realtime/**` | `server/services/s2s/grok.mdx` |
| `services/openai/realtime/**` | `server/services/s2s/openai.mdx` |
| `processors/frameworks/rtvi.py` | `server/frameworks/rtvi/rtvi-processor.mdx` and `server/frameworks/rtvi/rtvi-observer.mdx` |
| `processors/transcript_processor.py` | `server/utilities/transcript-processor.mdx` |
| `processors/user_idle_processor.py` | `server/utilities/user-idle-processor.mdx` |
| `processors/idle_frame_processor.py` | `server/pipeline/pipeline-idle-detection.mdx` |
| `pipeline/task.py` | `server/pipeline/pipeline-task.mdx` |
| `pipeline/runner.py` | `server/utilities/runner/guide.mdx` |
| `transports/base_transport.py` | `server/services/transport/transport-params.mdx` |

## Skip list

These files should never trigger doc updates.

| Pattern | Reason |
|---|---|
| `services/ai_service.py` | Internal base class |
| `services/stt_service.py` | Internal base class |
| `services/tts_service.py` | Internal base class |
| `services/llm_service.py` | Internal base class |
| `services/websocket_service.py` | Internal base class |
| `services/openai_realtime_beta/**` | Deprecated |
| `services/openai_realtime/**` | Deprecated |
| `services/gemini_multimodal_live/**` | Deprecated |
| `services/aws/agent_core.py` | Internal |
| `services/aws/sagemaker/**` | No doc page |
| `transports/base_input.py` | Internal base class |
| `transports/base_output.py` | Internal base class |
| `transports/websocket/client.py` | No doc page |
| `serializers/base_serializer.py` | Internal base class |
| `serializers/protobuf.py` | Internal |
| `processors/audio/**` | Internal |
| `pipeline/pipeline.py` | Core architecture, not a service doc |

## Pattern matching

For files not in the tables above, apply these patterns. Convert underscores to hyphens in provider names for doc filenames.

| Source pattern | Doc pattern |
|---|---|
| `services/{provider}/stt*.py` | `server/services/stt/{provider}.mdx` |
| `services/{provider}/tts*.py` | `server/services/tts/{provider}.mdx` |
| `services/{provider}/llm*.py` | `server/services/llm/{provider}.mdx` |
| `services/{provider}/image*.py` | `server/services/image-generation/{provider}.mdx` |
| `services/{provider}/video*.py` | `server/services/video/{provider}.mdx` |
| `services/{provider}/realtime/**` | `server/services/s2s/{provider}.mdx` |
| `transports/{name}/**` | `server/services/transport/{name}.mdx` |
| `serializers/{name}.py` | `server/services/serializers/{name}.mdx` |
| `observers/**` | `server/utilities/observers/` (match by class name) |
| `audio/vad/**` | `server/utilities/audio/` (match by class name) |
| `audio/filters/**` | `server/utilities/audio/` (match by class name) |
| `audio/mixers/**` | `server/utilities/audio/` (match by class name) |
| `processors/filters/**` | `server/utilities/filters/` (match by class name) |

If the doc file doesn't exist at the resolved path, the file is **unmapped**.

## Search fallback

For files that don't match any table or pattern above:
1. Extract the main class name(s) from the source file
2. Search the docs directory for that class name: `grep -r "ClassName" DOCS_PATH/server/`
3. If found in a doc page, use that as the mapping

# Source-to-Doc Mapping

Maps pipecat source files to their documentation pages. Source paths are relative to `src/pipecat/`. Doc paths are relative to `DOCS_PATH`.

## Name mismatches

These source paths don't follow the standard `services/{provider}/{type}.py` → `api-reference/server/services/{type}/{provider}.mdx` pattern.

| Source path                                 | Doc page                                                                                         |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `services/google/llm_vertex.py`             | `api-reference/server/services/llm/google-vertex.mdx`                                            |
| `services/google/google.py`                 | (shared base — check which services use it)                                                      |
| `services/google/gemini_live/**`            | `api-reference/server/services/s2s/gemini-live.mdx`                                              |
| `services/google/gemini_live/llm_vertex.py` | `api-reference/server/services/s2s/gemini-live-vertex.mdx`                                       |
| `services/aws_nova_sonic/**`                | `api-reference/server/services/s2s/aws.mdx`                                                      |
| `services/ultravox/**`                      | `api-reference/server/services/s2s/ultravox.mdx`                                                 |
| `services/grok/realtime/**`                 | `api-reference/server/services/s2s/grok.mdx`                                                     |
| `services/openai/realtime/**`               | `api-reference/server/services/s2s/openai.mdx`                                                   |
| `processors/frameworks/rtvi.py`             | `api-reference/server/rtvi/rtvi-processor.mdx` and `api-reference/server/rtvi/rtvi-observer.mdx` |
| `processors/idle_frame_processor.py`        | `api-reference/server/pipeline/pipeline-idle-detection.mdx`                                      |
| `pipeline/worker.py`                        | `api-reference/server/pipeline/pipeline-worker.mdx`                                              |
| `pipeline/runner.py`                        | `api-reference/server/utilities/runner/guide.mdx`                                                |
| `transports/base_transport.py`              | `api-reference/server/services/transport/transport-params.mdx`                                   |

## Pipecat Flows

Flows lives under `src/pipecat/flows/` and is documented separately from the services/transports above. Source paths below are relative to `src/pipecat/`; doc paths live in the `pipecat-ai/docs` repo.

### API reference pages

| Source file          | Doc page(s)                                                                                                          | Notes                                                                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `flows/types.py`     | `api-reference/pipecat-flows/types.mdx`                                                                              | NodeConfig, FlowsFunctionSchema, ActionConfig, ContextStrategy, ContextStrategyConfig, type aliases, `flows_tool_options` (and the deprecated `flows_direct_function` alias) |
| `flows/manager.py`   | `api-reference/pipecat-flows/flow-manager.mdx`                                                                       | FlowManager constructor, properties, methods                                                                                                |
| `flows/actions.py`   | `api-reference/pipecat-flows/flow-manager.mdx` (register_action), `api-reference/pipecat-flows/types.mdx` (ActionConfig) | Built-in action types and custom action registration                                                                                  |
| `flows/adapters.py`  | `api-reference/pipecat-flows/overview.mdx`                                                                           | LLM Provider Support table                                                                                                                   |
| `flows/exceptions.py`| `api-reference/pipecat-flows/exceptions.mdx`                                                                         | Exception hierarchy and descriptions                                                                                                         |

### Guide pages

Changes to Flows source may also affect the guides under `pipecat-flows/guides/`.

| Source file          | Guide page(s)                                  | What to check                                                                                            |
| -------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `flows/types.py`     | `pipecat-flows/guides/nodes-and-messages.mdx`  | NodeConfig properties, message format, respond_immediately                                              |
| `flows/types.py`     | `pipecat-flows/guides/functions.mdx`           | FlowsFunctionSchema examples, direct functions, `flows_tool_options` (and the deprecated `flows_direct_function` alias) |
| `flows/types.py`     | `pipecat-flows/guides/context-strategies.mdx`  | ContextStrategy enum values, ContextStrategyConfig usage                                                 |
| `flows/manager.py`   | `pipecat-flows/guides/state-management.mdx`    | FlowManager initialization, state dict, global_functions                                                |
| `flows/actions.py`   | `pipecat-flows/guides/actions.mdx`             | Built-in actions, custom actions, action timing                                                         |
| `flows/adapters.py`  | `pipecat-flows/guides/nodes-and-messages.mdx`  | Cross-Provider Compatibility section                                                                    |
| `flows/exceptions.py`| (rarely affects guides)                        |                                                                                                         |

### Other pages

| Page                                       | When to check                                                                  |
| ------------------------------------------ | ------------------------------------------------------------------------------ |
| `pipecat-flows/introduction.mdx`           | Rarely changes; only if installation or high-level framing changes             |
| `pipecat-flows/guides/quickstart.mdx`      | If FlowManager init signature, FlowsFunctionSchema, or handler return types change |
| `pipecat-flows/examples.mdx`               | Only if examples are added or removed                                          |
| `pipecat-flows/migration/migration-1.0.mdx`| Only on major API removals or new deprecations                                 |

## Skip list

These files should never trigger doc updates.

| Pattern                              | Reason                               |
| ------------------------------------ | ------------------------------------ |
| `services/ai_service.py`             | Internal base class                  |
| `services/stt_service.py`            | Internal base class                  |
| `services/tts_service.py`            | Internal base class                  |
| `services/llm_service.py`            | Internal base class                  |
| `services/websocket_service.py`      | Internal base class                  |
| `services/openai_realtime_beta/**`   | Deprecated                           |
| `services/openai_realtime/**`        | Deprecated                           |
| `services/gemini_multimodal_live/**` | Deprecated                           |
| `services/aws/agent_core.py`         | Internal                             |
| `services/aws/sagemaker/**`          | No doc page                          |
| `transports/base_input.py`           | Internal base class                  |
| `transports/base_output.py`          | Internal base class                  |
| `transports/websocket/client.py`     | No doc page                          |
| `serializers/base_serializer.py`     | Internal base class                  |
| `serializers/protobuf.py`            | Internal                             |
| `processors/audio/**`                | Internal                             |
| `pipeline/pipeline.py`               | Core architecture, not a service doc |
| `flows/__init__.py`                  | Re-exports only                      |

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
| `processors/filters/**`           | `api-reference/server/utilities/filters/` (match by class name)   |

If the doc file doesn't exist at the resolved path, the file is **unmapped**.

## Search fallback

For files that don't match any table or pattern above:

1. Extract the main class name(s) from the source file
2. Search the docs directory for that class name: `grep -r "ClassName" DOCS_PATH/api-reference/ DOCS_PATH/pipecat/`
3. If found in a doc page, use that as the mapping

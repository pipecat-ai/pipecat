- `GoogleVertexLLMService` now defaults to `gemini-3.6-flash`, matching `GoogleLLMService`, and its `location` now defaults to `"global"`. The two go together: Vertex serves the Gemini 3 series only from the global endpoint, and a regional endpoint returns 404 for every Gemini 3 model.

Requests made with the default configuration are served from the global endpoint rather than the previous `us-east4`. Pass `location=...` to pin a region, which keeps the Gemini 2.5 series available.

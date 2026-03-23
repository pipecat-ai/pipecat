- Added experimental `anyio` compatibility layer (`pipecat.utils.asyncio.compat`)
  and `AnyioTaskManager` to enable running Pipecat pipelines under `trio` as
  well as `asyncio`. The compat module provides backend-agnostic `Queue`,
  `PriorityQueue`, `Event`, `Lock`, and helper functions that work on both
  event loops. `AnyioTaskManager` is an async-context-manager alternative to
  `TaskManager` that uses structured concurrency task groups under the hood.
  Install the `trio` extra (`pip install pipecat-ai[trio]`) to use the trio
  backend. Most services still require asyncio; this lays the foundation for
  incremental migration.

- `TaskManager.cancel_task` now names where a slow-to-cancel task is blocked.
  The warning previously said only that the wait had timed out, which was not
  enough to identify the un-cancellable await. The stack is now sampled while
  the task is still pending (walking the coroutine's `cr_await` chain, since a
  suspended coroutine's `Task.get_stack()` reports only the task wrapper), and a
  second line reports how long the cancel actually took.

- Corrected the contract of `cancel_task`'s `timeout`, and of
  `INPUT_TASK_CANCEL_TIMEOUT_SECS` / `PROCESS_TASK_CANCEL_TIMEOUT_SECS`, which
  documented a bound that does not exist. Cancelling an asyncio task and then
  awaiting it always waits for the cancellation to complete — `asyncio.wait_for`
  included, since it cancels the inner task and then waits for it before raising
  `TimeoutError`. The timeout is a reporting threshold: it decides when you hear
  about a slow cancel, not how long the canceller blocks. A real bound would
  orphan a still-running process task and needs a generation guard on the frames
  it can still push.

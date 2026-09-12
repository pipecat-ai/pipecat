- Eval scenarios can put a function call to the judge. `eval:` is now accepted
  on `function_call` and `function_call_stopped` expectations alongside `calls:`
  (or the `name:`/`args:` shorthand): each call matched by name, and by any
  verbatim `args:`, is judged by name and arguments over the conversation so
  far, so a scenario can check what `args:` cannot match word for word, such as
  "the suggestion is about OpenTelemetry tracing, submitted for Jennifer
  Smith". A rejected call fails the expectation with kind `judge_no` and the
  judge's reason; a `continue` counts as a `no`, since a call is not a partial
  reply. The parser no longer warns about `eval:` on these two events, and a
  judged scenario that asserts on calls asks the bot for the `full` report
  level, so the judge sees the arguments.

  The scripted judge now sees the bot's function calls too: every
  `function_call` the harness matches goes into the judge's conversation as an
  assistant message `[tool call] name(args)`, in arrival order with the reply's
  segments, and the judge is told such a line is a call the bot made and part
  of its reply. A `response` criterion can therefore check that a confirmation
  matches what was actually submitted. `EvalJudge` gains `add_tool_call()` and
  `evaluate_call()`, and `pipecat.evals.judge.format_tool_call()` formats a call
  as one line.

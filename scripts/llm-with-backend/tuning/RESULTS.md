# LLMWithBackend tuning results

The latest results of the [tuning checks](README.md), from the last change to `LLMWithBackend`'s prompts or classifier. Each update is committed with the change it measures, so `git log -p` on this file ties every number to its commit and shows the numbers before it.

## Classifier replay

Predictive delegation's classifier on the recorded turns in `classifier_replay/`, with `jev-1.13.0` (2026-10-14):

```bash
uv run python scripts/llm-with-backend/tuning/classifier_replay/replay.py
```

```
107 turns, threshold 0.6
  false "yes" (delegated early, not needed): 0/59
  missed "yes" (left to the frontend):       0/48
  latency: p50 83ms, p90 113ms

probability of a "yes" per utterance: median (min-max)
  no  0.12 (0.11-0.14)  What is a race condition in programming?
  yes 0.95 (0.95-0.95)  Can you fix the flaky retry test in the HTTP client for me?
  no  0.17 (0.15-0.22)  While we wait, tell me a quick joke.
  no  0.11 (0.11-0.12)  How's it going with the fix?
  yes 0.91 (0.89-0.92)  Also, what's the CI status on main?
  no  0.19 (0.15-0.21)  What's the capital of Australia?
  yes 0.95 (0.94-0.96)  Can you find our internal docs on the deploy process?
  yes 0.94 (0.82-0.96)  Actually, never mind the docs.
  no  0.21 (0.20-0.36)  Thanks. What year was the first moon landing?
```

## Frontend replay

The three frontend models on the recorded turns in `frontend_replay/`, five runs per turn (2026-10-22). A wrong decision is a turn the model should have delegated and answered itself, or the reverse; the turns from the eval scenarios include every stop and correction they make. The acknowledgement is what the user hears for a delegated turn: the text the model wrote with the `delegate` call, if any, and its reply after the call; it is said twice when both have text. When the model acknowledges alongside the call, which it sometimes does, the `AFTER DELEGATING` paragraph tells the reply after the call not to acknowledge again: saying it in other words, or adding that the work is under way or that it will report back, is repeating yourself, and the reply finishes the thought only when the user asked for something else that has not been answered.

```bash
uv run python scripts/llm-with-backend/tuning/frontend_replay/replay.py --frontend <openai|google|anthropic> --samples 5
```

| frontend                        | wrong decisions, 75 turns × 5 | where                                                                                                                                                              | text with the call | said twice | words heard, median |
| ------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------ | ---------- | ------------------- |
| `openai` (gpt-4.1)              | 0                             |                                                                                                                                                                    | 1 of 275           | 1          | 15                  |
| `google` (gemini-3.6-flash)     | 2                             | a fix request answered with nothing while the API timed out, 1/5; a second request recorded from the Gemini bot ("check whether CI is green on main as well"), 1/5 | 28 of 273          | 8          | 14                  |
| `anthropic` (claude-sonnet-4-6) | 0                             |                                                                                                                                                                    | 181 of 275         | 45         | 12                  |

Claude says it twice on none of the requests made first in a conversation (0 of 120, against 17 with the paragraph below). On the lookup question "What does the on-call runbook say to do about upstream timeouts?" it wrote "Let me look that up for you!" with the call and then "I'm checking the on-call runbook now, hang tight!" after it in 15 of 15 runs, with this paragraph and with #6267's; now the reply after the call is empty in all 15, and the user hears 7 words against 17. Of the 45 delegations still said twice, 30 are messages with two parts, answered in two: the joke, then "I'm on the flaky retry test now"; "Got it, I'll drop the tests", then "I'll just get you the runbook answer on upstream timeouts". The other 15 are every run of the stop "don't bother, Sam already ran them", where "Got it, I'll call those off" is followed by a remark on Sam's run and an offer ("If Sam's run came back clean you're probably good to go, just let me know if you need anything else"); that turn has been said twice in 15 of 15 runs with every paragraph. Claude writes with the call in 181 delegations and leaves the reply after it empty in 136, against 195 and 101. Heard in all, its acknowledgement is shorter again: 12 words at the median against 16, past 25 words in 23 delegations against 31, the longest 51 against 56. Acknowledgements with a promise or an offer in them ("I'll let you know", "as soon as", "hang tight", "anything else", "would you like" and the like) are 26, 4 and 16 of 275 against 32, 7 and 41. Gemini says it twice in 8 delegations against 9; gpt-4.1 is unchanged. The judge below accepts 92 of gpt-4.1's 105 fix-request acknowledgements, 101 with the paragraph below, waiting on the "Let me look into that flaky retry test" ones as before (34 of 120 fix requests, against 33), and every one of Claude's and Gemini's. The classifier replay above does not move.

With the rule phrased as "finish that thought and stop; add only what is still missing; if what you said was already enough, say nothing, or a word at most", earlier the same day:

| frontend                        | wrong decisions, 75 turns × 5 | where                                                                             | text with the call | said twice | words heard, median |
| ------------------------------- | ----------------------------- | --------------------------------------------------------------------------------- | ------------------ | ---------- | ------------------- |
| `openai` (gpt-4.1)              | 0                             |                                                                                   | 2 of 275           | 2          | 16                  |
| `google` (gemini-3.6-flash)     | 1                             | a status question recorded from the Claude bot ("How's it going with that?"), 1/5 | 22 of 276          | 9          | 14                  |
| `anthropic` (claude-sonnet-4-6) | 0                             |                                                                                   | 195 of 275         | 94         | 16                  |

The stops and corrections are where the acknowledgement was said twice. With the paragraph below, Claude confirmed a stop with the call and again after it: "Got it, I'll leave the code as is." then "No changes will be made to the retry test or the HTTP client."; on the stop that also asks for a CI check, "On it, I'll check CI on main right now." then "I'll check CI on main for you right now, and I've dropped the retry test work". Now the reply after the call is empty on every run of the plain stop, the "never mind" and the two corrections (said twice in 1, 3 and 2 of 15 runs with the paragraph below, rerun the same day), and the stop with a CI check is said twice in 7 of 15 against 13, the two parts splitting the request: "Stopping the retry test fix, I'll leave that alone for you." then "Checking CI on main for you now!". Gemini says it twice in 9 delegations against 25. Claude's count barely moves because it now writes its sentence with the call on fix requests too (27 of 60 on "Fix the flaky retry test in the HTTP client", against 4) and leaves the reply after it empty (101 of its 275 replies after the call, against 38), and on "run the integration tests and tell me what you find" it splits its usual sentence in two: "Sure, I'll kick off those integration tests now!" with the call, "I'll let you know as soon as I have the results." after it, in 9 of 15. Heard in all, the acknowledgement is shorter: Claude's runs past 25 words in 31 delegations against 67, the longest is 56 words against 154, and acknowledgements with a promise or an offer in them ("I'll let you know", "get back to you", "anything else", "would you like") are 30, 7 and 24 of 275 against 30, 18 and 41. The judge below accepts 101 of gpt-4.1's 105 fix-request acknowledgements and every one of Claude's and Gemini's, as before.

The rule holds only where it is, after the paragraph's examples. Put ahead of them, or phrased as a reply that says nothing when the call carried an acknowledgement, it had gpt-4.1 answer a stop without the `delegate` call (19 of 90 runs) and Gemini give a result the backend never sent ("CI is green on main", 7 of 60).

With the paragraph before that rule, on 2026-10-21:

| frontend                        | wrong decisions, 75 turns × 5 | where                                                                                                                                                                                                                           | acknowledgement words, median | said twice |
| ------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ---------- |
| `openai` (gpt-4.1)              | 0                             |                                                                                                                                                                                                                                 | 15                            | 0 of 275   |
| `google` (gemini-3.6-flash)     | 1                             | a cancellation recorded from the Claude bot ("never mind, don't change anything"): 4/5, and 9/10 on a rerun                                                                                                                     | 15                            | 25 of 274  |
| `anthropic` (claude-sonnet-4-6) | 4                             | a second request made while work is under way, recorded from the Gemini bot ("check whether CI is green on main as well"): 1/5, and 9/10 on a rerun; over every run of this wording 19/30, against 18/20 with the wording below | 17                            | 94 of 271  |

The paragraph asks for one short sentence that fits the request, naming the task briefly if that reads naturally, and gives three examples: a lookup, a task, and a task named. gpt-4.1 names the task in most of its acknowledgements ("Sure, I'll get on fixing the flaky retry test in the HTTP client now."; on a lookup, "Let me check the on-call runbook for guidance on upstream timeouts"). Claude's and Gemini's are one sentence too, with a second on the turns that asked for a joke or a stop. Acknowledgements with a promise or an offer in them ("I'll let you know", "get back to you", "anything else", "would you like"): 28, 16 and 47 of 275, against 28, 23 and 77 with the wording below.

The text evals judge the acknowledgement on a fix request against "acknowledges that it is starting on the fix" (`progress_inquiry`, `aside_while_waiting`, `superseding_delegation`), and a replay's acknowledgements can be put to the same judge, with the recorded conversation before them. It accepts 100 of gpt-4.1's 105 on the three recorded fix requests, and every one of Claude's and Gemini's; the five it waits on are "Let me look into the flaky retry test in the HTTP client."

With "Acknowledge in one short sentence of your own, such as "Let me look into that", "Sure, checking on that now" or "Okay, let me see what I can find"", earlier the same day, gpt-4.1 took the first example as its acknowledgement for every kind of request ("Let me look into that." 65 of 275, most of its fix requests), and the judge above does not take "look into" as starting on a fix: 34 of 105 accepted, and in the text evals gpt-4.1 passed 43 of 60 against 54 at baseline, the three scenarios timing out while the judge waited for a reply it would accept. Claude's wrong decision was the second request recorded from the Gemini bot, 4/5.

| frontend                        | wrong decisions, 75 turns × 5 | acknowledgement words, median | said twice |
| ------------------------------- | ----------------------------- | ----------------------------- | ---------- |
| `openai` (gpt-4.1)              | 0                             | 14                            | 2 of 275   |
| `google` (gemini-3.6-flash)     | 0                             | 15                            | 27 of 275  |
| `anthropic` (claude-sonnet-4-6) | 1                             | 20                            | 159 of 274 |

With "a few words of your own, one short sentence at most" and "On it" as the first example, on 2026-10-20, the acknowledgements were shorter but often a bare "On it." (gpt-4.1 128 of 275; Claude's text with the call was "On it!" in most of its delegations), which is too curt when it is all the user hears before the backend answers:

| frontend                        | wrong decisions, 75 turns × 5 | where                                                                                                                                                                                                                                   | acknowledgement words, median | said twice |
| ------------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ---------- |
| `openai` (gpt-4.1)              | 0                             |                                                                                                                                                                                                                                         | 8                             | 0 of 275   |
| `google` (gemini-3.6-flash)     | 0                             |                                                                                                                                                                                                                                         | 14                            | 28 of 275  |
| `anthropic` (claude-sonnet-4-6) | 2                             | two second requests made while work is under way, recorded from the Gemini bot ("check whether CI is green on main as well") and the gpt-4.1 bot ("also, look up what the on-call runbook says"): 4/5 each (the second 9/10 on a rerun) | 15                            | 114 of 273 |

With the previous `AFTER DELEGATING` paragraph ("Acknowledge briefly, without offering updates or asking whether to go ahead"), run on 2026-10-20: gpt-4.1 7 wrong decisions and 23 words, Gemini 1 and 18, Claude 1 and 28. gpt-4.1's usual acknowledgement was "I'm looking into the flaky retry test in the HTTP client now. I'll let you know as soon as I have an update."; now it is "On it." in 128 of 275, and otherwise one short sentence. Its acknowledgements with a second sentence are the turns that asked for a joke, where the joke is the second sentence, and stops, where the confirmation is.

Three things in the paragraph were settled by this replay. The examples name no particular request: an example that mirrored one of the recorded requests ("Let me pull up that runbook") had Claude answer that request with the words, delegating 4/10. "An acknowledgement is never a substitute for the delegate call" holds Claude to the `delegate` call on a second request made while work is under way ("also, look up what the runbook says": 9/10 with the sentence, 7/10 with a softer one, 3/5 without). And the paragraph says nothing about the reply after the call: told that reply adds nothing, Claude on one turn filled it with an invented backend message instead ("Backend: The on-call runbook says..."), in 7 to 11 of 375 runs; with this paragraph, 0 of 375.

Before the pairing instruction's rewrite around its one rule, the same replay had gpt-4.1 at 52 wrong decisions, missing every stop and correction in the scenarios (0/5 or 1/5 on each), Claude at 8 and Gemini at 5.

## Text evals

The twelve text scenarios on the three bots with a text frontend, five runs each, without predictive delegation (2026-10-22, with this `AFTER DELEGATING` paragraph). The frontends run on their services' default models (`gpt-4.1`, `gemini-3.6-flash`, `claude-sonnet-4-6`), the backend on `claude-sonnet-4-6`, and the judge is the release suite's text judge. Each table comes from the run's `results.jsonl` through `results_table.py`:

```bash
uv run python -m pipecat.evals suite evals/llm-with-backend/manifest.yaml -s text --repeat 5
uv run python scripts/llm-with-backend/tuning/results_table.py <runs-dir>/<run>/results.jsonl
```

| scenario                      | anthropic-frontend | google-frontend | openai-responses-frontend |
| ----------------------------- | ------------------ | --------------- | ------------------------- |
| `accepted_offer/text`         | 5/5                | 5/5             | 5/5                       |
| `aside_while_waiting/text`    | 5/5                | 5/5             | 5/5                       |
| `cancel_and_replace/text`     | 5/5                | 5/5             | 5/5                       |
| `cancel_delegation/text`      | 4/5                | 5/5             | 5/5                       |
| `concurrent_delegations/text` | 1/5                | 0/5             | 1/5                       |
| `delegate_ack/text`           | 5/5                | 5/5             | 5/5                       |
| `delegate_with_aside/text`    | 5/5                | 5/5             | 5/5                       |
| `frontend_only/text`          | 5/5                | 5/5             | 5/5                       |
| `implicit_cancel/text`        | 3/5                | 5/5             | 5/5                       |
| `partial_cancel/text`         | 5/5                | 5/5             | 4/5                       |
| `progress_inquiry/text`       | 5/5                | 5/5             | 5/5                       |
| `superseding_delegation/text` | 5/5                | 5/5             | 5/5                       |
| **total**                     | **53/60**          | **55/60**       | **55/60**                 |

`concurrent_delegations` fails on every frontend at every baseline. Claude's two stop failures are one shape: it confirms the stop alongside its `delegate` call and says nothing in the run after it, as the paragraph asks, while the scenario judges only after the call, having no `check_from: turn_start`. Its third, on `implicit_cancel`, is the "don't bother" turn the frontend replay shows said twice with every paragraph: the reply after the call offered a fresh test run later, which the judge rejects as promising further work. gpt-4.1 answered one `partial_cancel` stop without the `delegate` call, the scenario it has missed before.

The same sweep earlier on 2026-10-22, with the rule phrased as "finish that thought and stop":

| scenario                      | anthropic-frontend | google-frontend | openai-responses-frontend |
| ----------------------------- | ------------------ | --------------- | ------------------------- |
| `accepted_offer/text`         | 5/5                | 5/5             | 5/5                       |
| `aside_while_waiting/text`    | 5/5                | 5/5             | 5/5                       |
| `cancel_and_replace/text`     | 5/5                | 5/5             | 5/5                       |
| `cancel_delegation/text`      | 4/5                | 5/5             | 5/5                       |
| `concurrent_delegations/text` | 0/5                | 1/5             | 0/5                       |
| `delegate_ack/text`           | 5/5                | 5/5             | 5/5                       |
| `delegate_with_aside/text`    | 5/5                | 5/5             | 5/5                       |
| `frontend_only/text`          | 5/5                | 5/5             | 5/5                       |
| `implicit_cancel/text`        | 4/5                | 4/5             | 5/5                       |
| `partial_cancel/text`         | 5/5                | 5/5             | 5/5                       |
| `progress_inquiry/text`       | 5/5                | 5/5             | 5/5                       |
| `superseding_delegation/text` | 5/5                | 5/5             | 5/5                       |
| **total**                     | **53/60**          | **55/60**       | **55/60**                 |

`concurrent_delegations` fails on every frontend at every baseline. Claude's `cancel_delegation` and `implicit_cancel` failures are one shape: it confirms the stop in the same response as its `delegate` call ("Got it, I'll leave the code as is."; "Good to know, thanks for the heads up! I'll call off the test run since Sam already confirmed they're passing.") and says nothing in the run after the call, as the paragraph asks, while the scenario judges only after the call, having no `check_from: turn_start`. Gemini's `implicit_cancel` failure closed its acknowledgement with an offer ("anything else I can help with"), which the judge rejects.

The same sweep on 2026-10-21, with the paragraph before its rule for a reply that follows an acknowledgement made with the call:

| scenario                      | anthropic-frontend | google-frontend | openai-responses-frontend |
| ----------------------------- | ------------------ | --------------- | ------------------------- |
| `accepted_offer/text`         | 5/5                | 5/5             | 5/5                       |
| `aside_while_waiting/text`    | 5/5                | 5/5             | 4/5                       |
| `cancel_and_replace/text`     | 5/5                | 5/5             | 5/5                       |
| `cancel_delegation/text`      | 3/5                | 5/5             | 5/5                       |
| `concurrent_delegations/text` | 1/5                | 1/5             | 0/5                       |
| `delegate_ack/text`           | 5/5                | 5/5             | 5/5                       |
| `delegate_with_aside/text`    | 5/5                | 5/5             | 5/5                       |
| `frontend_only/text`          | 5/5                | 5/5             | 5/5                       |
| `implicit_cancel/text`        | 5/5                | 5/5             | 5/5                       |
| `partial_cancel/text`         | 5/5                | 5/5             | 4/5                       |
| `progress_inquiry/text`       | 5/5                | 5/5             | 5/5                       |
| `superseding_delegation/text` | 5/5                | 5/5             | 5/5                       |
| **total**                     | **54/60**          | **56/60**       | **53/60**                 |

`concurrent_delegations` fails on every frontend at every baseline. Claude's two `cancel_delegation` failures are one shape: it confirms the stop in the same response as its `delegate` call ("Got it, I'll leave the code as is.") and says nothing in the run after the call, and the scenario looks only after the call, having no `check_from: turn_start`; the day before, Claude confirmed again after the call ("No worries, standing down on that"), and that is what passed. gpt-4.1's two are a `partial_cancel` turn it answered without delegating, as at baseline, and an `aside_while_waiting` run whose final report never came.

The same sweep on 2026-10-20, with "a few words of your own, one short sentence at most" and "On it" as the first example:

| scenario                      | anthropic-frontend | google-frontend | openai-responses-frontend |
| ----------------------------- | ------------------ | --------------- | ------------------------- |
| `accepted_offer/text`         | 5/5                | 5/5             | 5/5                       |
| `aside_while_waiting/text`    | 5/5                | 5/5             | 5/5                       |
| `cancel_and_replace/text`     | 5/5                | 5/5             | 5/5                       |
| `cancel_delegation/text`      | 5/5                | 5/5             | 5/5                       |
| `concurrent_delegations/text` | 0/5                | 1/5             | 0/5                       |
| `delegate_ack/text`           | 5/5                | 4/5             | 5/5                       |
| `delegate_with_aside/text`    | 5/5                | 5/5             | 5/5                       |
| `frontend_only/text`          | 5/5                | 5/5             | 5/5                       |
| `implicit_cancel/text`        | 4/5                | 5/5             | 5/5                       |
| `partial_cancel/text`         | 5/5                | 5/5             | 4/5                       |
| `progress_inquiry/text`       | 5/5                | 5/5             | 5/5                       |
| `superseding_delegation/text` | 5/5                | 5/5             | 5/5                       |
| **total**                     | **54/60**          | **55/60**       | **54/60**                 |

The same sweep on 2026-10-14, with the previous `AFTER DELEGATING` paragraph, for comparison:

| scenario                      | anthropic-frontend | google-frontend | openai-responses-frontend |
| ----------------------------- | ------------------ | --------------- | ------------------------- |
| `accepted_offer/text`         | 5/5                | 5/5             | 5/5                       |
| `aside_while_waiting/text`    | 5/5                | 5/5             | 4/5                       |
| `cancel_and_replace/text`     | 5/5                | 5/5             | 5/5                       |
| `cancel_delegation/text`      | 5/5                | 5/5             | 5/5                       |
| `concurrent_delegations/text` | 1/5                | 0/5             | 1/5                       |
| `delegate_ack/text`           | 5/5                | 5/5             | 4/5                       |
| `delegate_with_aside/text`    | 5/5                | 5/5             | 5/5                       |
| `frontend_only/text`          | 5/5                | 5/5             | 5/5                       |
| `implicit_cancel/text`        | 2/5                | 3/5             | 4/5                       |
| `partial_cancel/text`         | 5/5                | 5/5             | 3/5                       |
| `progress_inquiry/text`       | 5/5                | 5/5             | 4/5                       |
| `superseding_delegation/text` | 5/5                | 5/5             | 5/5                       |
| **total**                     | **53/60**          | **53/60**       | **50/60**                 |

And the same sweep on 2026-10-14 on the pairing instruction before this one:

| scenario                      | anthropic-frontend | google-frontend | openai-responses-frontend |
| ----------------------------- | ------------------ | --------------- | ------------------------- |
| `accepted_offer/text`         | 5/5                | 5/5             | 5/5                       |
| `aside_while_waiting/text`    | 5/5                | 5/5             | 3/5                       |
| `cancel_and_replace/text`     | 5/5                | 5/5             | 5/5                       |
| `cancel_delegation/text`      | 2/5                | 5/5             | 2/5                       |
| `concurrent_delegations/text` | 0/5                | 0/5             | 0/5                       |
| `delegate_ack/text`           | 5/5                | 5/5             | 5/5                       |
| `delegate_with_aside/text`    | 5/5                | 5/5             | 5/5                       |
| `frontend_only/text`          | 5/5                | 5/5             | 5/5                       |
| `implicit_cancel/text`        | 2/5                | 4/5             | 1/5                       |
| `partial_cancel/text`         | 5/5                | 5/5             | 0/5                       |
| `progress_inquiry/text`       | 5/5                | 5/5             | 4/5                       |
| `superseding_delegation/text` | 5/5                | 5/5             | 1/5                       |
| **total**                     | **49/60**          | **54/60**       | **36/60**                 |

### With predictive delegation

The same twelve scenarios with `predictive_delegation=PredictiveDelegation(JevClassifier(...))` added to each bot's `LLMWithBackend`, five runs each, on 2026-10-14 with the `AFTER DELEGATING` paragraph of that day:

| scenario                      | anthropic-frontend | google-frontend | openai-responses-frontend |
| ----------------------------- | ------------------ | --------------- | ------------------------- |
| `accepted_offer/text`         | 5/5                | 5/5             | 5/5                       |
| `aside_while_waiting/text`    | 5/5                | 5/5             | 5/5                       |
| `cancel_and_replace/text`     | 5/5                | 5/5             | 5/5                       |
| `cancel_delegation/text`      | 5/5                | 5/5             | 5/5                       |
| `concurrent_delegations/text` | 1/5                | 0/5             | 1/5                       |
| `delegate_ack/text`           | 5/5                | 5/5             | 4/5                       |
| `delegate_with_aside/text`    | 5/5                | 5/5             | 5/5                       |
| `frontend_only/text`          | 5/5                | 5/5             | 5/5                       |
| `implicit_cancel/text`        | 5/5                | 4/5             | 4/5                       |
| `partial_cancel/text`         | 5/5                | 5/5             | 5/5                       |
| `progress_inquiry/text`       | 5/5                | 5/5             | 5/5                       |
| `superseding_delegation/text` | 5/5                | 5/5             | 5/5                       |
| **total**                     | **56/60**          | **54/60**       | **54/60**                 |

The classifier, live, from the bots' logs: every request and stop scored 0.88 or more and was delegated early; the status question in `progress_inquiry` scored 0.10–0.11 in all 15 runs and small talk 0.11–0.35, none delegated early. The failures are `concurrent_delegations` and the judge hesitating on acknowledgements.

### Stops

The scenarios' stops and corrections (`cancel_delegation`, `implicit_cancel`, `partial_cancel` and `superseding_delegation`, 20 runs per frontend) with no `delegate` call: gpt-4.1 1, Claude 0 and Gemini 0 (2026-10-22; gpt-4.1's one was `partial_cancel`), against 0, 0 and 0 earlier that day, 1, 0 and 0 on 2026-10-20 (gpt-4.1's one was `partial_cancel`, "forget the tests" while a runbook lookup is also running), 2, 0 and 0 on 2026-10-14 and 16, 1 and 0 before the rewrite.

### Known failures

- `concurrent_delegations` fails on every frontend most of the time: the judge rejects the reply that reports both results.
- `implicit_cancel`'s acknowledgement is judged "no" at P(no) 0.56–0.68 while the judge's explainer finds nothing wrong with it; `aside_while_waiting`, `delegate_ack` and `progress_inquiry` lose a run each to the judge answering "continue" to a correct one-line acknowledgement.

## Audio evals

The thirteen audio scenarios on all five bots, one run each, without predictive delegation (2026-10-22, with this `AFTER DELEGATING` paragraph). The user's voice is synthesized with Kokoro, the bot's speech transcribed with Moonshine, and the judge is the release suite's audio judge.

```bash
uv run python -m pipecat.evals suite evals/llm-with-backend/manifest.yaml -s audio
```

| scenario                           | anthropic-frontend | gemini-live-frontend | google-frontend | openai-realtime-frontend | openai-responses-frontend |
| ---------------------------------- | ------------------ | -------------------- | --------------- | ------------------------ | ------------------------- |
| `accepted_offer/audio`             | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `aside_while_waiting/audio`        | 0/1                | 1/1                  | 0/1             | 1/1                      | 0/1                       |
| `cancel_and_replace/audio`         | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `cancel_delegation/audio`          | 1/1                | 1/1                  | 1/1             | 0/1                      | 1/1                       |
| `concurrent_delegations/audio`     | 0/1                | 0/1                  | 1/1             | 0/1                      | 1/1                       |
| `delegate_ack/audio`               | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `delegate_with_aside/audio`        | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `frontend_only/audio`              | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `implicit_cancel/audio`            | 0/1                | 1/1                  | 0/1             | 1/1                      | 0/1                       |
| `partial_cancel/audio`             | 1/1                | 1/1                  | 0/1             | 1/1                      | 1/1                       |
| `progress_inquiry/audio`           | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `result_during_interruption/audio` | 1/1                | 1/1                  | 1/1             | 0/1                      | 1/1                       |
| `superseding_delegation/audio`     | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| **total**                          | **10/13**          | **12/13**            | **10/13**       | **10/13**                | **11/13**                 |

One run per scenario is noisy. Across the five bots, 53/65, against 49/65 earlier the same day and 50/65 on 2026-10-20. The scenarios' stops and corrections with no `delegate` call: one, gpt-4.1 on `implicit_cancel`, the transcription split described below. The Claude and OpenAI Realtime frontends each added a line after a stop's acknowledgement, which the absence check catches: Claude's `implicit_cancel` is the "don't bother" turn again, "Sounds like those are called off. let me know if there's anything else you need." after the confirmation with the call; OpenAI Realtime's `cancel_delegation` is "All set. I've stopped the effort to fix the retry test, and no changes will be made." Gemini closed its `implicit_cancel` acknowledgement with an offer, and the judge answered "continue" to its `partial_cancel` acknowledgement. OpenAI Realtime's `result_during_interruption` left the interrupted runbook answer unsaid. `aside_while_waiting` failed on three bots at the backend, which did not apply its patch within the turn's 150 s, and `concurrent_delegations` on three, as in every pass.

The same pass earlier on 2026-10-22, with the rule phrased as "finish that thought and stop":

| scenario                           | anthropic-frontend | gemini-live-frontend | google-frontend | openai-realtime-frontend | openai-responses-frontend |
| ---------------------------------- | ------------------ | -------------------- | --------------- | ------------------------ | ------------------------- |
| `accepted_offer/audio`             | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `aside_while_waiting/audio`        | 0/1                | 0/1                  | 0/1             | 1/1                      | 0/1                       |
| `cancel_and_replace/audio`         | 0/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `cancel_delegation/audio`          | 0/1                | 1/1                  | 1/1             | 0/1                      | 1/1                       |
| `concurrent_delegations/audio`     | 1/1                | 0/1                  | 0/1             | 0/1                      | 0/1                       |
| `delegate_ack/audio`               | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `delegate_with_aside/audio`        | 0/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `frontend_only/audio`              | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `implicit_cancel/audio`            | 0/1                | 1/1                  | 1/1             | 0/1                      | 0/1                       |
| `partial_cancel/audio`             | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `progress_inquiry/audio`           | 1/1                | 0/1                  | 1/1             | 1/1                      | 1/1                       |
| `result_during_interruption/audio` | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `superseding_delegation/audio`     | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| **total**                          | **8/13**           | **10/13**            | **11/13**       | **10/13**                | **10/13**                 |

One run per scenario is noisy. Across the five bots, 49/65, against 50/65 on 2026-10-20. The scenarios' stops and corrections with no `delegate` call: one, gpt-4.1 on `implicit_cancel`, where the transcription split the stop in two ("Oh, wait. Actually, don't bother." then "Sam already ran them on his branch this morning, and they were fine.") and the model answered the pair without the call; rerun five times, it did so in 1 of 5 with this paragraph and 2 of 5 with the paragraph before it, so the split is what trips it, not the wording. The other failures: the judge answering "continue" to a correct acknowledgement (gpt-4.1's `aside_while_waiting`, Gemini Live's `progress_inquiry`), the Claude and OpenAI Realtime frontends adding a line after a stop's acknowledgement, which the absence check catches (4: Claude's `implicit_cancel` is the double acknowledgement itself, "Got it, I'll stop that." then "I've called those off — Sam already ran them on his branch this morning and they came back clean, so you're all set.", the one stop the frontend replay also shows unchanged; its `cancel_delegation` adds "No update yet — I'll let you know as soon as I have something."), `aside_while_waiting` on Claude and Google, where the backend's fix outran the scenario's window, and on Gemini Live, which transcribed Canberra as "Kambara", `concurrent_delegations`, and on Claude `cancel_and_replace`, where the replacement's result missed the window, and `delegate_with_aside`, where the joke never came.

The same pass on 2026-10-20, with the paragraph before its rule for a reply that follows an acknowledgement made with the call:

| scenario                           | anthropic-frontend | gemini-live-frontend | google-frontend | openai-realtime-frontend | openai-responses-frontend |
| ---------------------------------- | ------------------ | -------------------- | --------------- | ------------------------ | ------------------------- |
| `accepted_offer/audio`             | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `aside_while_waiting/audio`        | 0/1                | 0/1                  | 0/1             | 0/1                      | 1/1                       |
| `cancel_and_replace/audio`         | 1/1                | 1/1                  | 1/1             | 0/1                      | 1/1                       |
| `cancel_delegation/audio`          | 0/1                | 1/1                  | 1/1             | 0/1                      | 1/1                       |
| `concurrent_delegations/audio`     | 0/1                | 0/1                  | 0/1             | 0/1                      | 1/1                       |
| `delegate_ack/audio`               | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `delegate_with_aside/audio`        | 0/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `frontend_only/audio`              | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `implicit_cancel/audio`            | 0/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `partial_cancel/audio`             | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `progress_inquiry/audio`           | 1/1                | 0/1                  | 1/1             | 1/1                      | 1/1                       |
| `result_during_interruption/audio` | 1/1                | 0/1                  | 1/1             | 1/1                      | 1/1                       |
| `superseding_delegation/audio`     | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| **total**                          | **8/13**           | **9/13**             | **11/13**       | **9/13**                 | **13/13**                 |

One run per scenario is noisy. Across the five bots, 50/65, against 53/65 on 2026-10-14 and 48/65 before the pairing instruction's rewrite. The scenarios' stops and corrections with no `delegate` call: none on any frontend. The failures: the judge answering "continue" to a correct acknowledgement (4), the Claude and OpenAI Realtime frontends adding a line after a stop's acknowledgement, which the absence check catches (3), `aside_while_waiting` on Claude, Gemini and OpenAI Realtime, where the backend's fix outran the scenario's window, `concurrent_delegations`, and `delegate_with_aside` on Claude, where the reply ran past the window. The two speech-to-speech frontends' acknowledgements stay long: Gemini Live's "I'll look into the HTTP client tests to see why that retry test is failing. I will keep you updated on my progress." and OpenAI Realtime's "I'm on it and will update you once I have something solid. If you want, I can also summarize any tradeoffs if the fix touches real retry behavior." The paragraph's examples and limits are tuned on the text frontends.

The same pass on 2026-10-14, with the previous `AFTER DELEGATING` paragraph:

| scenario                           | anthropic-frontend | gemini-live-frontend | google-frontend | openai-realtime-frontend | openai-responses-frontend |
| ---------------------------------- | ------------------ | -------------------- | --------------- | ------------------------ | ------------------------- |
| `accepted_offer/audio`             | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `aside_while_waiting/audio`        | 0/1                | 1/1                  | 1/1             | 0/1                      | 1/1                       |
| `cancel_and_replace/audio`         | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `cancel_delegation/audio`          | 0/1                | 1/1                  | 1/1             | 0/1                      | 1/1                       |
| `concurrent_delegations/audio`     | 1/1                | 0/1                  | 0/1             | 1/1                      | 0/1                       |
| `delegate_ack/audio`               | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `delegate_with_aside/audio`        | 1/1                | 1/1                  | 1/1             | 1/1                      | 0/1                       |
| `frontend_only/audio`              | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `implicit_cancel/audio`            | 0/1                | 1/1                  | 1/1             | 1/1                      | 0/1                       |
| `partial_cancel/audio`             | 1/1                | 0/1                  | 1/1             | 1/1                      | 1/1                       |
| `progress_inquiry/audio`           | 1/1                | 0/1                  | 1/1             | 1/1                      | 1/1                       |
| `result_during_interruption/audio` | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `superseding_delegation/audio`     | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| **total**                          | **10/13**          | **10/13**            | **12/13**       | **11/13**                | **10/13**                 |

And the same pass on 2026-10-14 on the pairing instruction before this one:

| scenario                           | anthropic-frontend | gemini-live-frontend | google-frontend | openai-realtime-frontend | openai-responses-frontend |
| ---------------------------------- | ------------------ | -------------------- | --------------- | ------------------------ | ------------------------- |
| `accepted_offer/audio`             | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `aside_while_waiting/audio`        | 0/1                | 1/1                  | 0/1             | 1/1                      | 1/1                       |
| `cancel_and_replace/audio`         | 0/1                | 1/1                  | 1/1             | 0/1                      | 1/1                       |
| `cancel_delegation/audio`          | 0/1                | 0/1                  | 1/1             | 0/1                      | 0/1                       |
| `concurrent_delegations/audio`     | 1/1                | 1/1                  | 1/1             | 0/1                      | 0/1                       |
| `delegate_ack/audio`               | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `delegate_with_aside/audio`        | 0/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `frontend_only/audio`              | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `implicit_cancel/audio`            | 0/1                | 1/1                  | 1/1             | 1/1                      | 0/1                       |
| `partial_cancel/audio`             | 1/1                | 1/1                  | 1/1             | 1/1                      | 0/1                       |
| `progress_inquiry/audio`           | 0/1                | 0/1                  | 1/1             | 1/1                      | 1/1                       |
| `result_during_interruption/audio` | 1/1                | 1/1                  | 1/1             | 1/1                      | 1/1                       |
| `superseding_delegation/audio`     | 1/1                | 1/1                  | 1/1             | 1/1                      | 0/1                       |
| **total**                          | **7/13**           | **11/13**            | **12/13**       | **10/13**                | **8/13**                  |

On 2026-10-14, 53/65 against 48/65 before. The scenarios' stops and corrections with no `delegate` call: gpt-4.1 0 of 4 (4 of 4 before), Gemini Live 1 (`partial_cancel`; 1 before, `cancel_delegation`), the others 0. The rest are the judge answering "continue" to a correct acknowledgement, the Claude and OpenAI Realtime frontends adding a line after a stop's acknowledgement, which the absence check catches, and `aside_while_waiting` on Claude, where the backend's fix outran the scenario's window. `delegate_with_aside/audio` on the gpt-4.1 bot, five more runs a side: the `delegate` call was made in 5 of 5 (4 of 5 before, the miss the fix promised with no call). The scenario itself passed 1 of 5 and 2 of 5: in audio the user's sentence splits into two turns, the second cutting the joke off mid-sentence, and the judge sees the truncated reply.

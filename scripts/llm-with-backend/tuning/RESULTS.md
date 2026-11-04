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

## Wait reassurance prompt

`WAIT_REASSURANCE_PROMPT` (`pipecat.processors.wait_reassurance`) asked of the three frontend models in seven mid-delegation contexts, five runs each (2026-11-02). Each context is the example's conversation after "Fix the flaky retry test in the HTTP client" was delegated and acknowledged, with the backend's silent notes for a `search_codebase` and a `read_file` call. Five end with the prompt as the processor appends it (ten seconds quiet): the request as is; the request with "no need to keep me posted, just tell me when it's done"; the same decline given after a first remark, with an `apply_patch` note since; the request with "keep me posted with the details as you go"; and a second prompt after a first remark, with a note since in which the backend names the cause and says it will fix it, then `apply_patch` and `run_tests` calls. Two end with the user speaking after a prompt and a remark, the decline and an aside ("which time zone is the London office on?"), to check that the prompt left in the conversation does not shape the answer. A reply is silent when it opens with `∅`, a remark otherwise; the words counted are the remark's.

| context                       | expected                              | anthropic-frontend | google-frontend          | openai-responses-frontend |
| ----------------------------- | ------------------------------------- | ------------------ | ------------------------ | ------------------------- |
| as is                         | a brief remark, no file names         | 5/5 (17–21 words)  | 5/5 (10–20)              | 5/5 (12–21)               |
| declined up front             | silent                                | 5/5                | 5/5                      | 5/5                       |
| declined after one            | silent                                | 5/5                | 5/5                      | 5/5                       |
| detail asked for              | a remark naming the step              | 5/5 (17–22)        | 5/5 (13–17, names the file) | 5/5 (19–27)            |
| second remark, finding in the notes | a remark naming the step, not the finding | 0/5          | 5/5                      | 4/5                       |
| the user declines after a remark | answered                           | 5/5                | 5/5                      | 5/5                       |
| the user asks an aside after a remark | answered                      | 5/5                | 5/5                      | 5/5                       |

Claude's second remark is "The fix is in and the tests are running now" in 5 of 5, the patch reported as a step done; Gemini says "I'm applying the fix and running the tests now", gpt-4.1 "I'm updating the retry logic and running the tests", with one "I spotted the issue". gpt-4.1 still ends some plain remarks with a promise to report back ("I'll update you as soon as I have news"), which the prompt asks it not to.

Three earlier wordings, measured the same way, shaped the prompt. With the first, which gave the silence rule without "first check what the user asked for", gpt-4.1 spoke through an up-front decline in 4 of 5 runs. With the second, which asked for a remark "brief and unspecific unless they asked for detail", every frontend passed the finding on in the second remark ("I've found the cause and have made a change"), and gpt-4.1 answered the user's own decline with `∅` in 5 of 5, the message left in the conversation read as a rule for answering them; a closing "this applies to this reply only" brought that to 2 and 4 of 10 in two wordings, and "`∅` is only ever the answer to this message: whatever the user says next, answer them" to 0 of 10. "At most what kind of step is under way now, with no file names, commands or code" is what keeps Gemini's plain remarks from naming the file (3 of 5 did with the wording before it) and makes gpt-4.1's and Gemini's second remark a step.

The same runs with the turn-completion instructions appended to the system instruction, as `filter_incomplete_user_turns` appends them: every remark opens with `●` on all three frontends, every declined reply is silent (30 of 30), Gemini's as `● ∅` in 10 of 10, gpt-4.1's as `● ∅` in 2 and `∅` in 8, Claude's as `∅` alone. A bare `∅` reaches the processor through the turn-completion filter's fallback, which passes a response that ended without a marker, so the reply is dropped either way. Claude's filter judged the user's decline itself incomplete (`○`) in 5 of 5, the turn-completion feature's reading of a short utterance with no audio behind it, not the prompt's doing.

The prompt was reworded on 2026-11-03 (the wording now in `WAIT_REASSURANCE_PROMPT`: a shorter message that names the user's preferences as the thing to weigh, "answer with exactly ∅ to respond with silence", and "make no offers or promises, not even to keep the user posted"), and replayed the same way in nine contexts: the seven above, plus the user's decline sent before any of the remark was spoken, and the decline after the backend's reasoning notes (several hundred words of `Backend (thinking):`) sit between the remark and it. Five runs each.

| context                                  | expected                                      | anthropic-frontend | google-frontend | openai-responses-frontend |
| ---------------------------------------- | --------------------------------------------- | ------------------ | --------------- | ------------------------- |
| as is                                    | a brief remark, no promise                    | 5/5                | 5/5             | 5/5                       |
| declined up front                        | silent                                        | 5/5                | 5/5             | 5/5                       |
| declined after one                       | silent                                        | 5/5                | 5/5             | 5/5                       |
| detail asked for                         | a remark naming the step                      | 5/5                | 5/5             | 5/5                       |
| second remark, finding in the notes      | a remark naming the step, not the finding     | 0/5                | 4/5             | 5/5                       |
| the user declines after a remark         | answered                                      | 5/5                | 5/5             | 5/5                       |
| the decline before the remark was spoken | answered                                      | 0/5                | 5/5             | 5/5                       |
| the decline after the reasoning notes    | answered                                      | 5/5                | 5/5             | 5/5                       |
| the user asks an aside after a remark    | answered                                      | 5/5                | 5/5             | 5/5                       |

Claude's second remark passes the finding on in 3 of 5 ("The fix is in and the tests are running now") and is silent in 2; Gemini's names the patch once ("applied a patch to implement exponential backoff"). Claude closes four plain remarks with "won't be long". gpt-4.1 promises to keep the user posted in 3 of 5 detail remarks, where the user asked to be kept posted. The wording "not even to keep the user posted" is what removed gpt-4.1's promises from the plain remark: without it (the same prompt otherwise), 2 of 5 and 3 of 10 ended with one; a version that restated the earlier prompt's content clause instead made Claude silent in 5 of 5 plain contexts.

With the turn-completion instructions appended, as `filter_incomplete_user_turns` appends them: gpt-4.1 and Gemini answer every declined context with `● ∅` or `∅` (20 of 20); Claude answers the up-front decline with a bare `○` in 4 of 5, the turn-completion filter's "the user needs more time" marker, which the filter drops unspoken but also reports upstream as an incomplete turn. The filtering bot in the manifest runs gpt-4.1.

The prompt was reworded again on 2026-11-04 (the wording now in `WAIT_REASSURANCE_PROMPT`): a closing sentence, "`∅` is only ever the answer to this message. Whatever the user says next, answer them, and when the work reports back, tell the user, however often you have chosen silence before", and the detail case as an exception to "keep it general": "unless the user asked for detail: then name the step, drawing on the notes the work has produced". Replayed the same way in eleven contexts: the nine above, plus the backend's final report arriving after three silent decisions, once with the decline given after a remark and once with it given up front. Five runs each.

| context                                   | expected                                  | anthropic-frontend | google-frontend             | openai-responses-frontend |
| ----------------------------------------- | ----------------------------------------- | ------------------ | --------------------------- | ------------------------- |
| as is                                     | a brief remark, no promise                | 5/5                | 5/5                         | 5/5                       |
| declined up front                         | silent                                    | 5/5                | 5/5                         | 5/5                       |
| declined after one                        | silent                                    | 5/5                | 5/5                         | 5/5                       |
| detail asked for                          | a remark naming the step                  | 5/5                | 5/5 (names the file)        | 5/5                       |
| second remark, finding in the notes       | a remark naming the step, not the finding | 0/5                | 5/5                         | 5/5                       |
| the user declines after a remark          | answered                                  | 5/5                | 5/5                         | 5/5                       |
| the decline before the remark was spoken  | answered                                  | 0/5                | 3/5                         | 5/5                       |
| the decline after the reasoning notes     | answered                                  | 5/5                | 5/5                         | 5/5                       |
| the report after three silences, declined after a remark | relayed                    | 5/5                | 5/5                         | 5/5                       |
| the report after three silences, declined up front | relayed                          | 5/5                | 5/5                         | 5/5                       |
| the user asks an aside after a remark     | answered                                  | 5/5                | 5/5                         | 5/5                       |

Claude's second remark is "The fix is patched in and the tests are running now" in 5 of 5, a step reported as done; live it says "Still running the tests" in most runs (see the evals below). Gemini answers the decline that lands before its remark was spoken in 3 of 5 and with `∅` in 2, where the 2026-11-03 wording answered 5 of 5: the closing sentence's cost, paid for the report after silences, which the replay does not show (every frontend relays it 5 of 5 here, and Gemini lost it live in 6 of 10 runs without the sentence). gpt-4.1's silence on the up-front decline turns on the detail clause's form: the same words as a second clause after a semicolon ("; if the user asked for detail, name the step itself, drawing on the notes the work has produced, files included") had gpt-4.1 speak through the decline in 14 of 15 replays, which a first replay of 5 had not shown; as an exception it is silent in 15 of 15. "Files included" is what had Gemini name the test file in its plain remarks live (3 of 5 runs of `wait_reassurance`), so the clause leaves the files to the notes, and gpt-4.1's detail remarks name the test and the client without a file.

With the turn-completion instructions appended: Gemini answers the up-front decline with `● ∅` in 5 of 5, gpt-4.1 with `∅` in 5 of 5, Claude with `∅` in 4 and `● ∅` in 1, and no bare `○`.

## Wait reassurance evals

The five `wait_reassurance*` text scenarios on the three cascade bots, the predictive-delegation bot and the filtering bot (`openai-responses-frontend-filter-incomplete-turns.py`), five runs each (2026-11-04, the prompt as now in `WAIT_REASSURANCE_PROMPT`, the mark read by the LLM service, the waiting timer started over when the last pending function call settles), the judge the release suite's text judge with `allow_continue: false` on the remark expectations, and the declined scenarios asserting the silent decision as `event: llm_marker` with `marker: silent`:

| scenario                                | anthropic-frontend | google-frontend | openai-responses-frontend | predictive-delegation | filter-incomplete-turns |
| --------------------------------------- | ------------------ | --------------- | ------------------------- | --------------------- | ----------------------- |
| `wait_reassurance/text`                 | 5/5                | 4/5             | 5/5                       | 5/5                   | 5/5                     |
| `wait_reassurance_repeats/text`         | 4/5                | 5/5             | 5/5                       | 5/5                   |                         |
| `wait_reassurance_detail_requested/text` | 5/5               | 5/5             | 3/5                       | 3/5                   |                         |
| `wait_reassurance_declined/text`        | 5/5                | 3/5             | 5/5                       | 5/5                   |                         |
| `wait_reassurance_declined_upfront/text` | 5/5               | 4/5             | 5/5                       | 3/5                   | 5/5                     |

99 of 110. No run is silent where a remark is expected, the shape of 8 misses with the 2026-11-03 wording below: every first and second decision in `wait_reassurance`, `wait_reassurance_repeats` and `wait_reassurance_declined` is a remark, every decline after a remark is answered (20 of 20), and a silent decision follows it (20 of 20). The misses:

- Gemini answers the backend's final report with `∅`, after three or more silent decisions, in 3 of its 10 declined runs (2 of `wait_reassurance_declined`, 1 of `wait_reassurance_declined_upfront`): the backend reports, the bot says nothing, and the report expectation times out at 150 seconds. The closing sentence is what holds this down: without it Gemini lost the report in 6 of 10 declined runs (2026-11-03 night, the full run before this one), with it in 9 of 40 across this run and the two live rounds before it. The replay does not reproduce the decision (`∅` in 2 of 40 with a failing run's context replayed as it was): a live Gemini request carries a `thought_signature` on every model turn, so the decision rides on reasoning state the replay cannot supply, and a wording's effect on Gemini has to be checked live.
- The predictive-delegation bot speaks through the up-front decline in 2 of 5 ("I'm still working on fixing that test for you"), where the plain gpt-4.1 frontend is silent in 10 of 10 (the filtering bot included) and the replay of its context in 15 of 15. The two bots' contexts differ in the predictive-delegation instructions in the system prompt; the predictive bot was 3 of 5, 5 of 5 and 3 of 5 over the last three live rounds.
- gpt-4.1's `detail_requested` remarks name the HTTP client and its retry test and no file ("I'm still looking into the test and the HTTP client code to see what's causing the flaky retries"), and the judge, asked for a file or function name where there is one, rejects 4 of 10. The 2026-11-03 wording's remarks were the same shape and passed 9 of 10 ("I'm digging into the HTTP client and its tests right now"), and the detail clause ending "files included" passed 9 of 10 live, at the cost of Gemini naming the test file in its plain remarks (above). Claude and Gemini name the file in 10 of 10.
- Single runs: Gemini's one `wait_reassurance` remark names a file ("the test_http_client_live.py file"), and Claude's one `wait_reassurance_repeats` second remark reports the patch as done ("The fix is in and the tests are running now"; its other four say "Still running the tests").

Claude's `wait_reassurance_declined` passes 5 of 5 once the controller starts the waiting timer over when the last pending call settles: Claude answers the decline by speaking and then making a `delegate` call that passes the preference on, the call's start cancels the timer, and with nothing to start it over no silent decision followed (1 of 5 live before the fix, 5 of 5 after).

The filtering bot passes `frontend_only/text` 5/5, `wait_reassurance/text` 5/5 and `wait_reassurance_declined_upfront/text` 5/5, gpt-4.1 writing its silent replies as `∅` alone. The release scenario `async_tool_wait_reassurance/text` on `function-calling-openai-async.py` (a 15-second weather lookup): 5/5.

The twelve existing text scenarios on the three cascade bots, one run each, with the processor in the pipeline: 32/36. `concurrent_delegations` fails on all three bots, as it does on every baseline. The fourth is Claude's `delegate_ack`, where its second remark, "The fix is in and the tests are running now", arrived where the scenario expects the report, and the judge read it as a report missing its details; the scenario's report expectation does not allow for a remark that sounds like a step done. Earlier runs on this branch failed Claude's `cancel_delegation` instead (it confirms the stop before its `delegate` call, and the scenario judges after it).

Audio, two runs per cascade bot: `wait_reassurance/audio` 6/6, `wait_reassurance_declined_upfront/audio` 5/6. In the Claude miss the turn analyzer split the user's request at its full stop into two turns, the bot acknowledged each ("Sure, I'll get that fixed!", "I'm on it — I'll let you know when it's done!") and finished speaking 16 seconds into the 25 seconds the scenario allows for the silent decision, so the first decision was due after the window closed.

Between the wordings, live checks of the scenarios each change bears on. The closing sentence alone (2026-11-03 night, the detail case "unless the user asked for detail" and nothing more): Gemini `wait_reassurance_declined_upfront` 4/5 then 9/10 and `wait_reassurance_declined` 4/5 then 7/10, Claude `wait_reassurance_declined` 5/5, the predictive bot `wait_reassurance_declined_upfront` 5/5. The detail clause ending "files included": `wait_reassurance_detail_requested` Claude 4/5, Gemini 5/5, gpt-4.1 5/5, the predictive bot 4/5, and gpt-4.1 `wait_reassurance` 5/5.

### The 2026-11-03 wording

The same scenarios on 2026-11-03, with the prompt's 2026-11-03 wording (no closing sentence, the detail case "unless the user asked for detail" and nothing more), the mark read by the LLM service, and the controller before the fix above:

| scenario                                | anthropic-frontend | google-frontend | openai-responses-frontend | predictive-delegation | filter-incomplete-turns |
| --------------------------------------- | ------------------ | --------------- | ------------------------- | --------------------- | ----------------------- |
| `wait_reassurance/text`                 | 5/5                | 3/5             | 5/5                       | 4/5                   | 5/5                     |
| `wait_reassurance_repeats/text`         | 0/5                | 4/5             | 5/5                       | 5/5                   |                         |
| `wait_reassurance_detail_requested/text` | 5/5               | 5/5             | 4/5                       | 5/5                   |                         |
| `wait_reassurance_declined/text`        | 3/5                | 2/5             | 5/5                       | 5/5                   |                         |
| `wait_reassurance_declined_upfront/text` | 5/5               | 4/5             | 5/5                       | 5/5                   | 5/5                     |

On `main`, where no processor runs, the same scenarios fail in 45 of 45 runs (three per bot and scenario, 2026-11-02): the remark turn times out, and the silent decision the declined scenarios wait for never comes.

The user's decline after a remark (`wait_reassurance_declined`) is answered in every run that reaches it, 12 of 12 on the three cascade bots and 5 of 5 with predictive delegation; one Claude answer repeats the backend's fix and is rejected by the judge. The misses are elsewhere. Gemini's first decision is silent, with no remark in the 20 seconds allowed, in 4 of its 15 runs of `wait_reassurance`, `wait_reassurance_repeats` and `wait_reassurance_declined`, where the replay's plain context gets a remark 5 of 5; live, the backend's working notes sit between the acknowledgement and the prompt. Claude's second decision in `wait_reassurance_repeats` is silent in 4 of 5 and passes the finding on in the fifth ("reports that the work is done and the tests are running"). The single failures: a Gemini remark naming a file, a predictive-delegation remark reporting findings, a gpt-4.1 `detail_requested` remark the judge read as general ("its general intent to begin investigating"), and two Gemini runs, one of `declined` and one of `declined_upfront`, in which Gemini answered the backend's final report with `∅` after a run of silent decisions, the frontend silent throughout and the report expectation timing out at 150 seconds.

The filtering bot passes `frontend_only/text` 5/5, `wait_reassurance/text` 5/5 and `wait_reassurance_declined_upfront/text` 5/5. In the last, gpt-4.1 wrote every one of its 21 silent replies as `∅` alone, without the turn marker; the filter buffers an unmarked reply until the response ends, so the service's reading of it has to come before the filter reports the response, or the harness sees an unmarked-turn report first and fails (1 of 5 before that order was fixed).

The release scenario `async_tool_wait_reassurance/text` on `function-calling-openai-async.py` (a 15-second weather lookup): 5/5.

The twelve existing text scenarios on the three cascade bots, one run each, with the processor in the pipeline: 34/36. The two failures are ones every baseline below has: `concurrent_delegations` on the Gemini frontend and Claude's `cancel_delegation` acknowledgement in the same response as its `delegate` call.

Audio, two runs per cascade bot: `wait_reassurance/audio` 6/6; `wait_reassurance_declined_upfront/audio` 6/6. In audio mode the harness waits for the bot's spoken turn to end before the next user turn, so the remark is heard out without a hold.

### The earlier prompt

The same scenarios on 2026-11-02, with the prompt before its 2026-11-03 rewording, the mark read by `WaitReassurance` itself, and the declined scenarios asserting the mark in the LLM's raw text:

| scenario                                | anthropic-frontend | google-frontend | openai-responses-frontend | predictive-delegation | filter-incomplete-turns |
| --------------------------------------- | ------------------ | --------------- | ------------------------- | --------------------- | ----------------------- |
| `wait_reassurance/text`                 | 5/5                | 5/5             | 5/5                       | 5/5                   | 4/5                     |
| `wait_reassurance_repeats/text`         | 1/5                | 5/5             | 5/5                       | 5/5                   |                         |
| `wait_reassurance_detail_requested/text` | 5/5               | 5/5             | 5/5                       | 3/5                   |                         |
| `wait_reassurance_declined/text`        | 2/5                | 5/5             | 1/5                       | 5/5                   |                         |
| `wait_reassurance_declined_upfront/text` | 5/5               | 4/5             | 5/5                       | 5/5                   | 5/5                     |

Claude's second remark in `wait_reassurance_repeats` passes the backend's finding on in 4 of 5 runs ("I've found the issue and applied a fix — now running the tests to confirm everything passes"), as the replay shows, and the judge rejects it as a finding reported. The predictive-delegation bot's two `detail_requested` failures are gpt-4.1 remarks the judge read as general ("I'm digging into the HTTP client and its retry test now"). The filtering bot's one `wait_reassurance` failure is a remark closing with "I'll let you know what turns up", which the judge took as a promise. Gemini's one `declined_upfront` failure is a run whose backend was still patching at the 150-second limit for the result, with the frontend silent throughout (six `∅` decisions). The `wait_reassurance_declined` failures on the three cascade bots are one shape, six of seven: the user's decline, spoken after the remark, is answered with `∅`, and the bot says nothing; the seventh is a Claude run that acknowledged with a `delegate` call and whose next decision fell outside the window. The predictive-delegation bot answers every time, with a `delegate` call that passes the preference on.

In `wait_reassurance_declined` the user's decline is sent eight seconds after the remark's text arrives. Sent at once, as the harness does by default in text mode, it reaches the bot before any of the remark has been spoken: the remark is interrupted and left out of the context, and the model sees the reassurance prompt followed directly by the decline; all three cascade frontends then went silent in 14 of 20 runs, and Claude does so in 10 of 10 replays of that shape with either closing sentence tried.

With the remark spoken and recorded, what makes the decline go unanswered is what the backend puts between the two. The context of one failing gpt-4.1 run, replayed as it was, gets `∅` in 6 of 6; the same context with its `Backend (thinking):` notes (the backend's reasoning summaries, several hundred words here) blanked gets an answer in 4 of 4, and with every note between the remark and the decline removed in 4 of 4. Neither wording tried for the prompt's closing sentence changes the live context's outcome (0 and 1 of 6), and a standing system instruction saying the same gets an answer in 4 of 6. The prompt stays in the conversation after its run, and its rule for the mark is read onto the user's own words once enough developer text separates them; a prompt that is not kept, or a connector that keeps reasoning summaries out of the frontend's context, is where a fix lies.

The filtering bot, with the same prompt: `frontend_only/text` 5/5, `wait_reassurance/text` 4/5 and `wait_reassurance_declined_upfront/text` 5/5; the existing text scenarios 32/36 (the four failures `concurrent_delegations` on all three bots and Claude's `cancel_delegation`); the release scenario 5/5; audio 6/6 and 6/6.

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

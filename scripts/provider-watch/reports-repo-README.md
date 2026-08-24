<!-- Seed README for pipecat-ai/provider-watch-reports: copied there when the repo is
     created; after that, the copy in that repo is the live one. -->

# provider-watch-reports

Research reports on the providers behind [Pipecat](https://github.com/pipecat-ai/pipecat)'s
services: new models, retired models, and API affordances Pipecat does not expose yet. A bot
writes them weekly; maintainers work from the digest issue it opens here.

## The weekly loop

1. **Monday:** the bot researches every service unit, pushes reports here, opens draft PRs on
   pipecat (label `provider-watch`) for clear-cut updates, and files one issue here — "Provider
   watch YYYY-MM-DD" — listing PRs to review and changes to consider, ranked by priority.
2. **Review the draft PRs:** merge what's right, close what isn't — a closed PR is recorded as
   a decision against the change, so the bot won't propose it again.
3. **For each "to consider" item,** either make the change (no reply needed — the next run sees
   the code), or record a decision by replying on the issue, one per line, naming the unit and
   enough of the item to identify it:

   ```
   deepgram/stt, diarize_model: skip, the extra= workaround is fine
   openai/realtime, tool_choice: tracked in pipecat-ai/pipecat#5400, stop reporting
   cartesia/tts, emotion controls: later, revisit after 2026-11-01
   ```

4. **Close the issue** when triaged. Open or closed makes no difference to the bot — closing is
   for humans.

The next run reads the comments on the three most recent digest issues and files each decision
in the unit's `decisions.md`, where it is permanent. So the only deadline is soft: triage
within about three weeks, or reply on the newest digest issue instead — a decision names its
unit and item, so it works from any of them.

## Layout

```
reports/<provider>/<unit>/YYYY-MM-DD.md   one report per service unit per run
reports/<provider>/<unit>/decisions.md    decisions currently in force for the unit
digests/YYYY-MM-DD.md                     one page per run: what to look at first
```

A *unit* is one provider × one service type — `openai/llm`, `openai/realtime`, `cartesia/tts`,
`deepgram/flux-stt` — matching `src/pipecat/services/<provider>/` in the Pipecat repo.

## Reading a report

A report is a snapshot of the gap between what the provider offers and what Pipecat handles on
that date — complete on its own, not a diff against the previous one. It opens with a one-line
verdict, then **What's new for Pipecat** in two buckets:

- **PRs** — a draft PR (or, on a dry run, a local branch with its review command)
- **To consider** — worth doing, but needs a maintainer's call; each shows how long it has been open

followed by **Verification** (the probes that ran, with TTFB — TTFAT for LLMs) and **Sources**.
The YAML frontmatter is machine-read by the next run: gaps keep their `first_seen` date.

Items the team has ruled out live in the unit's `decisions.md`, one bullet per decision in
force. The bot stops raising an item once a decision covers it, and deletes entries that lapse
(a revisit date passing reopens the gap; git history is the archive).

## How it runs

The bot is the `/provider-watch` skill in the Pipecat repo (`.claude/skills/provider-watch/`),
run weekly by `.github/workflows/provider-watch.yml` and on demand locally. Draft PRs it opens
on pipecat carry the `provider-watch` label.

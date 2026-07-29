You are one iteration of a Ralph loop implementing Phase B (Pipecat Cloud
scenarios) of the transport latency benchmark. Work from the pipecat repo
root. Do exactly one task, then stop.

1. Read `benchmarks/transport_latency/ralph/checklist.md`.
2. Take the FIRST unchecked task (`- [ ]`). Do only that task — no
   look-ahead, no drive-by refactors.
3. Before editing, read the files the task names and the neighboring
   benchmark modules (`client_core.py`, `webrtc_client.py`, `moq_client.py`,
   `scenarios.py`, `transport_latency.py`) so new code matches their shape.
   Repo conventions are in `AGENTS.md`.
4. Implement, then run the task's acceptance criteria verbatim. Only if the
   output confirms success, flip the task's `- [ ]` to `- [x]`.
5. HUMAN GATE: if the task needs anything only the human can do — deploying
   the PCC agent, `docker push`, supplying `PIPECAT_CLOUD_API_KEY`,
   `DAILY_API_KEY`, a relay URL, or Cloudflare TURN keys — do NOT attempt
   it. Append to `benchmarks/transport_latency/ralph/HUMAN_TODO.md`:

   ## PENDING: <short title>
   <the exact commands to run, each with a one-line explanation>

   Leave the task unchecked and stop; the loop pauses until the human marks
   the section `## DONE:` (or deletes it) and reruns `ralph.sh`.
6. Do not run `git commit` or `git push` — leave changes in the working
   tree; the human commits between iterations.
7. End with a one-paragraph summary: what you did, what you verified, and
   which task is next.

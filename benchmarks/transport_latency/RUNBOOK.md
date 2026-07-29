# Campaign runbook — phase-A scenarios

Step-by-step procedure for one full benchmark campaign over the six local +
deployed scenarios and their six browser twins. Anyone with this repo, the local MoQ relay bits (below),
and the deployed-relay values (below) should be able to run it end to end and
get comparable tables. Background, methodology, and fairness rules:
[README.md](README.md). This file is the only place run commands live; ad-hoc
variants and agent notes are at the end.

**Rules of the road**

- Run every command yourself, in order. Each step has a **gate** — do not
  advance past a failed gate; diagnose first.
- All commands run from the **pipecat repo root** unless the step says
  otherwise.
- Relay processes run in the **foreground in their own terminal tab** so you
  can watch their logs. The harness never starts them for you.
- One campaign = one commit, one machine, one sitting per tier. Trials are
  3 × 60 s everywhere (harness defaults).

**Placeholders** (values provided out of band by whoever operates the relay
box — substitute them everywhere they appear)

| placeholder | meaning | example |
|---|---|---|
| `RELAY_HOST` | DNS name of the deployed relay box | `relay.example.com` |
| `PUBLIC_IP` | the box's static public IP | `3.101.44.7` |
| `TURN_PASSWORD` | coturn long-term credential (username is `pipecat`) | — |

**Prerequisites**

- macOS with Docker Desktop running; `uv`; Homebrew `openssl` (for the QUIC
  check: `brew install openssl`).
- For the local MoQ relay (Phase 2.1): the `pipecat-moq-relay` repo — it
  carries the relay Dockerfile, configs, and the `moq-relay-dev.sh` helper.
  Internal; where to get it is provided out of band, like the relay-box
  values. Clone it next to this repo — the helper finds the pipecat checkout
  beside it (`PIPECAT_DIR` overrides). (The local coturn kit needs no setup —
  it's in-repo at `benchmarks/transport_latency/coturn-local/`.)
- `uv sync --group bench --extra moq --extra webrtc --extra runner` completed
  once (the runner extra carries the bot's FastAPI dev server; the bench group
  carries matplotlib, playwright, and sounddevice).
- Browser scenarios: Google Chrome (`uv run playwright install chrome` if
  missing) and the bench page built once:
  ```bash
  cd benchmarks/transport_latency/web && npm install && npm run build && cd ../../..
  ```
- Browser method A only (glass-to-glass): two BlackHole virtual audio
  devices — `brew install blackhole-2ch blackhole-16ch` (the harness's
  `--mic-label`/`--spk-label` defaults match these device names).
- The deployed relay box values above. The box runs moq-relay and coturn;
  provisioning and operating it are outside this runbook.

---

## Phase 0 — Preflight + clean slate + floors

- [ ] **0.1 Record the code state.**
  ```bash
  git status --short && git rev-parse --short HEAD
  ```
  Gate: working tree clean (runbook/doc edits are fine); note the commit —
  every trial JSON must carry it in `environment.pipecat_commit`.

- [ ] **0.2 Clean slate.** Archive any previous results into a subdirectory
  (still inside `results/`, so it stays gitignored); the campaign starts with
  no loose files so floors re-run fresh and `summary.md` only ever shows this
  campaign. Summary/chart rendering reads only top-level files, never
  subdirectories.
  ```bash
  cd benchmarks/transport_latency/results
  mkdir archive-$(date +%Y%m%d-%H%M)
  mv *.json *.png *.log *.md archive-*/ 2>/dev/null; cd ../../..
  ```
  Gate: `ls -p benchmarks/transport_latency/results/ | grep -v /` prints
  nothing — no loose files at the top level, archives only.

- [ ] **0.3 Floors** (in-process client-stack baselines, no bot, no network):
  ```bash
  uv run python benchmarks/transport_latency/transport_latency.py --floors
  ```
  Gate: `results/floor-moq.json` p50 ≈ 1–3 ms and `results/floor-webrtc.json`
  p50 ≈ 170–190 ms. A webrtc floor far off ~176 ms means the aiortc
  jitter-buffer behavior changed — stop and investigate before burning trials.

## Phase 1 — Local direct (no relays)

Tab layout: one tab for the harness. The bot is spawned per-trial by the
harness; its logs land in `results/bot-<scenario>-<trial>.log`.

- [ ] **1.1 webrtc-local** (~4 min):
  ```bash
  uv run python benchmarks/transport_latency/transport_latency.py --scenario webrtc-local
  ```
  Gate, per trial (printed line + `results/webrtc-local-<n>.json`):
  n ≥ 55, drops ≤ 1, and `ice_pair` shows `host/host`.

- [ ] **1.2 moq-serve** (~4 min):
  ```bash
  uv run python benchmarks/transport_latency/transport_latency.py --scenario moq-serve
  ```
  Gate: n ≥ 55, drops = 0 per trial.

- [ ] **1.3 Browser twins** (~8 min; Chrome runs headless, driven by
  Playwright):
  ```bash
  uv run python benchmarks/transport_latency/transport_latency.py \
      --scenario browser-webrtc-local --scenario browser-moq-serve
  ```
  Gate: n ≥ 55 per trial; `browser-webrtc-local` JSON `ice_pair` =
  `host/host`. On failure, read `results/page-<scenario>.log` (the page's
  console) before the bot log.

- [ ] **1.4 Tier review.** Open `results/summary.md` and
  `results/chart-local-direct.png`. Sanity: `bot-internal p50` ≈ 0.1–0.3 ms
  for **all** scenarios (the pipeline-symmetry canary); browser bars sit
  above their python twins (Chrome's adaptive jitter buffers are part of the
  measured path).

## Phase 2 — Local relay (docker)

Tab layout: tab 1 = relay (foreground, logs), tab 2 = harness.

- [ ] **2.1 moq-relay-local.**
  Tab 1 (leave running; adjust the path to your `pipecat-moq-relay` checkout):
  ```bash
  ../pipecat-moq-relay/moq-relay-dev.sh relay
  ```
  Tab 2, after the relay reports up:
  ```bash
  uv run python benchmarks/transport_latency/transport_latency.py --scenario moq-relay-local
  ```
  Gate: n ≥ 55, drops = 0 per trial; JSON `moq_relay_url` =
  `https://localhost:4443/anon`; sessions visible in the relay logs.
  Then, with the relay still up, the browser twin:
  ```bash
  uv run python benchmarks/transport_latency/transport_latency.py --scenario browser-moq-relay-local
  ```
  Gate: n ≥ 55 per trial. Then ctrl-c the relay.

- [ ] **2.2 webrtc-turn-local.**
  Tab 1 (leave running):
  ```bash
  benchmarks/transport_latency/coturn-local/run.sh
  ```
  Tab 2:
  ```bash
  uv run python benchmarks/transport_latency/transport_latency.py --scenario webrtc-turn-local
  ```
  Gate: n ≥ 55 per trial; JSON `ice_pair` = `["relay", "host"]` — the client
  side is TURN-forced, the bot keeps host candidates on loopback, and all
  media crosses coturn once; allocation lines in the coturn logs.
  Then, with coturn still up, the browser twin:
  ```bash
  uv run python benchmarks/transport_latency/transport_latency.py --scenario browser-webrtc-turn-local
  ```
  Gate: n ≥ 55 per trial; `ice_pair` local side `relay` (Chrome is pinned to
  `iceTransportPolicy: relay`). Then ctrl-c coturn.

- [ ] **2.3 Tier review.** `results/chart-local-relay.png` renders; excess
  p50 for each transport is within a few ms of its Phase-1 sibling (a relay
  hop on loopback is nearly free — a big jump means something else moved).

## Phase 3 — Deployed relay preflight (laptop only)

The deployed tier uses one cloud box running **both** moq-relay and coturn,
so the two deployed scenarios traverse the same box, region, and network
path. Verify from the laptop that your client stack and the endpoints line
up before burning trials.

- [ ] **3.1 Version pin.** The relay and the client must speak the same MoQ
  revision — a skew silently drops streams (everything connects, no audio
  flows).
  ```bash
  uv run python -c "import importlib.metadata as m; print(m.version('moq-ffi'))"
  ```
  Gate: the printed moq-ffi version pairs with the relay version the box
  operator reports (moq-ffi `0.2.30` ↔ relay `moq-relay-v0.13.5`). If it
  doesn't, stop and coordinate — do not run the deployed tier.

- [ ] **3.2 Reachability.**
  ```bash
  dig +short RELAY_HOST
  curl -s -o /dev/null -w '%{http_code}\n' http://RELAY_HOST/   # any HTTP status = TCP up
  /opt/homebrew/bin/openssl s_client -connect RELAY_HOST:443 -quic -alpn h3 </dev/null
  ```
  Gate: dig returns `PUBLIC_IP`; the openssl output shows a CA-signed chain
  (`issuer=...Let's Encrypt`).

## Phase 4 — Deployed relay tier

Tab layout: one tab for the harness. (Tailing the box's `moq-relay` /
`coturn` container logs in a second tab is useful if you have SSH access to
the box, but no gate requires it.)

- [ ] **4.1 moq-relay-deployed.**
  Smoke (1 × 15 s), then the real run:
  ```bash
  uv run python benchmarks/transport_latency/transport_latency.py \
      --scenario moq-relay-deployed --relay-url https://RELAY_HOST/anon \
      --trials 1 --duration 15
  uv run python benchmarks/transport_latency/transport_latency.py \
      --scenario moq-relay-deployed --relay-url https://RELAY_HOST/anon
  ```
  Gate: smoke completes with n > 0 (else check
  `results/bot-moq-relay-deployed-1.log`); real run n ≥ 55 per trial.
  Note: the 15 s smoke JSON is overwritten by the real trial 1 — run the
  real thing immediately so the campaign keeps only 60 s trials.

- [ ] **4.2 webrtc-turn-deployed.**
  Smoke, then real:
  ```bash
  uv run python benchmarks/transport_latency/transport_latency.py \
      --scenario webrtc-turn-deployed \
      --turn-url turn:PUBLIC_IP:3478 --turn-username pipecat --turn-credential TURN_PASSWORD \
      --trials 1 --duration 15
  uv run python benchmarks/transport_latency/transport_latency.py \
      --scenario webrtc-turn-deployed \
      --turn-url turn:PUBLIC_IP:3478 --turn-username pipecat --turn-credential TURN_PASSWORD
  ```
  Gate: JSON `ice_pair` = `relay/relay` (both sides through the cloud coturn);
  n ≥ 55 per trial. Nonzero drops are **expected** on a real TURN path —
  count them, don't chase them.

- [ ] **4.3 Browser twins** (same endpoints; run right after their python
  siblings so network conditions are comparable):
  ```bash
  uv run python benchmarks/transport_latency/transport_latency.py \
      --scenario browser-moq-relay-deployed --relay-url https://RELAY_HOST/anon
  uv run python benchmarks/transport_latency/transport_latency.py \
      --scenario browser-webrtc-turn-deployed \
      --turn-url turn:PUBLIC_IP:3478 --turn-username pipecat --turn-credential TURN_PASSWORD
  ```
  Gate: n ≥ 55 per trial; browser TURN JSON `ice_pair` local side `relay`.

- [ ] **4.4 Tier review.** `results/chart-deployed-relay.png` renders; both
  scenarios' excess includes one real internet round trip to the box (they
  should differ from each other by protocol behavior, not by tens of ms of
  geography — they share the box).

## Phase 5 — Analysis + record keeping

- [ ] **5.1 Final render** (idempotent; regenerates summary + all charts):
  ```bash
  uv run python benchmarks/transport_latency/charts.py
  ```
  Gate: `results/summary.md` has 12 scenario rows (6 python + 6 browser)
  × 3 trials + 2 floor rows; `chart-all.png`, the three tier sub-charts, and
  the `chart-series-*.png` time-series charts exist.

- [ ] **5.2 Cross-checks** before believing the numbers:
  - `bot-internal p50` comparable across all six scenarios (pipeline canary);
  - each transport's floor unchanged from Phase 0 if re-run;
  - within each tier, compare **excess over own floor**, not raw p50
    (see "Reading the numbers" in the README);
  - drops concentrated where expected (deployed TURN), ~0 elsewhere.

- [ ] **5.3 Share the campaign.** Results are never checked in — the repo
  carries only the procedure. `results/` holds everything a write-up needs
  (`summary.md`, chart PNGs, per-trial JSONs with the environment block);
  share them out-of-band (PR description, doc, dashboard) alongside the
  commit hash from 0.1 so numbers stay tied to the code that produced them.
  Tell the relay-box operator the deployed tier is done so the box can be
  parked.

---

## Ad-hoc variants (outside a recorded campaign)

- **All four local scenarios back-to-back** (relays must already be up):
  ```bash
  uv run python benchmarks/transport_latency/transport_latency.py --scenario all-local
  uv run python benchmarks/transport_latency/transport_latency.py --scenario all-browser-local
  ```
- **Knobs**: `--trials N --duration S` for quick smokes (e.g. `1` × `15`),
  `--jitter-ms` (default 60) for the pinned MoQ buffers, `--save-wav` to keep
  the probe audio, `--floors` to re-run floors explicitly, `--headed` to
  watch the Chrome window on browser scenarios.
- **Talk-spurt onset latency**: add `--gap-every 20` to any scenario — the
  probe pauses for 2 s every 20 s, and the first chirp after each gap is
  reported separately (`spurt p50` in the summary). Use the same flag on
  scenarios you intend to compare.
- **Glass-to-glass (browser method A)**: with both BlackHole devices
  installed:
  ```bash
  uv run python benchmarks/transport_latency/transport_latency.py \
      --scenario browser-webrtc-local --browser-method a
  ```
  Results are tagged `browser_method: "a"` and excluded from the comparison
  charts — they answer "what does a human hear", not "which transport is
  faster". Compare A−B on the same scenario to isolate the fixed browser/OS
  audio cost.
- **Impairment runs** (local relay tiers): apply a profile, tag the trials,
  reset:
  ```bash
  sudo benchmarks/transport_latency/impair/impair.sh loss1
  uv run python benchmarks/transport_latency/transport_latency.py \
      --scenario moq-relay-local --impairment loss1
  sudo benchmarks/transport_latency/impair/impair.sh clean
  ```
  `impair.sh burst <profile> <seconds>` holds a profile mid-trial for
  recovery measurements — read the recovery from the `chart-series-*.png`
  time-series (time for RTT to return to its pre-burst band).

## Agent notes

This runbook is agent-runnable as written; the differences from a human run:

- No terminal tabs: start relay containers detached (`docker run -d ...`,
  the same images/mounts as the foreground scripts) and read
  `docker logs`; remove them at the end of the tier.
- Verify every gate from the artifacts, not scrollback: per-trial stats,
  `ice_pair`, and `moq_relay_url` are in `results/<scenario>-<trial>.json`;
  bot output is in `results/bot-<scenario>-<trial>.log`.
- macOS may deny the docker engine access to protected folders (Documents),
  which breaks volume mounts; copy configs to `/tmp` and mount from there.
- Hard human gates — never do these autonomously: anything involving cloud
  spend, DNS, or credentials, and all git operations.
- Log every deviation from this runbook and report it with the results.

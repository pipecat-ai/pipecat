# Campaign runbook — 6 phase-A scenarios

Step-by-step procedure for one full benchmark campaign over the six local +
deployed scenarios. Anyone with this repo (and its two sibling repos, below)
should be able to run it end to end and get comparable tables. Background,
methodology, and fairness rules: [README.md](README.md). Bare command list
for repeat runs: [COMMANDS.md](COMMANDS.md).

**Rules of the road**

- Run every command yourself, in order. Each step has a **gate** — do not
  advance past a failed gate; diagnose first.
- All commands run from the **pipecat repo root** unless the step says
  otherwise (SSH steps run on the cloud box).
- Relay processes run in the **foreground in their own terminal tab** so you
  can watch their logs. The harness never starts them for you.
- One campaign = one commit, one machine, one sitting per tier. Trials are
  3 × 60 s everywhere (harness defaults).

**Placeholders** (substitute your values everywhere they appear)

| placeholder | meaning | example |
|---|---|---|
| `RELAY_HOST` | DNS name you control, A-record → the box's public IP | `relay.example.com` |
| `PUBLIC_IP` | the instance's static public IP (AWS: Elastic IP; OCI: reserved public IP) | `3.101.44.7` |
| `PRIVATE_IP` | the instance's private IP (`hostname -I` on the box) | `172.31.5.20` |
| `TURN_PASSWORD` | long-term credential you choose for coturn | (strong, no shell metachars) |

**Prerequisites**

- macOS with Docker Desktop running; `uv`; Homebrew `openssl` (for the QUIC
  check: `brew install openssl`).
- Sibling repos checked out next to this one:
  - `../../daily-co/pipecat-moq-relay` (relay Dockerfile + configs)
  - `../../daily-co/pipecat-coturn` (local coturn kit)
  - `../moq-relay-dev.sh` (local relay helper, in the parent dir)
- `uv sync --group bench --extra moq --extra webrtc --extra runner` completed
  once (the runner extra carries the bot's FastAPI dev server).
- A cloud account for the relay box — AWS (Phase 3 below) or OCI
  ([deploy-oci.md](deploy-oci.md)) — and control of `RELAY_HOST` DNS.

---

## Phase 0 — Preflight + clean slate + floors

- [ ] **0.1 Record the code state.**
  ```bash
  git status --short && git rev-parse --short HEAD
  ```
  Gate: working tree clean (runbook/doc edits are fine); note the commit —
  every trial JSON must carry it in `environment.pipecat_commit`.

- [ ] **0.2 Clean slate.** Archive any previous results; the campaign starts
  from an empty dir so floors re-run fresh and `summary.md` only ever shows
  this campaign.
  ```bash
  cd benchmarks/transport_latency
  [ -d results ] && mv results results-archive-$(date +%Y%m%d-%H%M)
  mkdir results && cd ../..
  ```
  Gate: `ls benchmarks/transport_latency/results/` is empty.

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

- [ ] **1.3 Tier review.** Open `results/summary.md` and
  `results/chart-local-direct.png`. Sanity: `bot-internal p50` ≈ 0.1–0.3 ms
  for **both** scenarios (the pipeline-symmetry canary).

## Phase 2 — Local relay (docker)

Tab layout: tab 1 = relay (foreground, logs), tab 2 = harness.

- [ ] **2.1 moq-relay-local.**
  Tab 1 (leave running):
  ```bash
  ../moq-relay-dev.sh relay
  ```
  Tab 2, after the relay reports up:
  ```bash
  uv run python benchmarks/transport_latency/transport_latency.py --scenario moq-relay-local
  ```
  Gate: n ≥ 55, drops = 0 per trial; JSON `moq_relay_url` =
  `https://localhost:4443/anon`; sessions visible in the relay logs.
  Then ctrl-c the relay.

- [ ] **2.2 webrtc-turn-local.**
  Tab 1 (leave running):
  ```bash
  ../../daily-co/pipecat-coturn/run.sh
  ```
  Tab 2:
  ```bash
  uv run python benchmarks/transport_latency/transport_latency.py --scenario webrtc-turn-local
  ```
  Gate: n ≥ 55 per trial; JSON `ice_pair` = `["relay", "host"]` — the client
  side is TURN-forced, the bot keeps host candidates on loopback, and all
  media crosses coturn once; allocation lines in the coturn logs.
  Then ctrl-c coturn.

- [ ] **2.3 Tier review.** `results/chart-local-relay.png` renders; excess
  p50 for each transport is within a few ms of its Phase-1 sibling (a relay
  hop on loopback is nearly free — a big jump means something else moved).

## Phase 3 — AWS box (one-time): moq-relay + coturn on one instance

One small EC2 instance runs **both** relays so the two deployed scenarios
traverse the same box, region, and network path. You run every command;
steps 3.4+ run **on the box** over SSH. Deploying on OCI instead: follow
[deploy-oci.md](deploy-oci.md) for 3.1–3.2 and its host-network relay
variant of 3.5; everything else here applies unchanged.

- [ ] **3.1 Launch the instance** (AWS console or CLI): Ubuntu 24.04 LTS,
  `t3.small`, your nearest region, your SSH key pair. Allocate and associate
  an **Elastic IP**. Security group inbound rules:

  | port | proto | purpose |
  |---|---|---|
  | 22 | TCP | SSH (restrict to your IP) |
  | 80 | TCP | certbot HTTP challenge |
  | 443 | TCP | relay HTTP/WebSocket fallback |
  | 443 | UDP | relay QUIC/WebTransport |
  | 3478 | UDP | coturn STUN/TURN |
  | 49160–49200 | UDP | coturn relay allocations |

- [ ] **3.2 DNS.** Create an A record: `RELAY_HOST` → `PUBLIC_IP`. Gate (laptop):
  `dig +short RELAY_HOST` returns `PUBLIC_IP`.

- [ ] **3.3 Pin the relay version to your client stack.** On the laptop:
  ```bash
  uv run python -c "import importlib.metadata as m; print(m.version('moq-ffi'))"
  ```
  moq-ffi `0.2.30` → relay tag `moq-relay-v0.13.5` (the Dockerfile default).
  A skew silently drops streams — if your moq-ffi differs, look up the
  matching `moq-relay-v*` release and pass it as `MOQ_TAG` in 3.5.

- [ ] **3.4 Box setup** (SSH: `ssh ubuntu@PUBLIC_IP`):
  ```bash
  sudo apt-get update && sudo apt-get install -y docker.io certbot
  sudo usermod -aG docker ubuntu && exit   # re-SSH so the group applies
  ```

- [ ] **3.5 moq-relay.** On the box:
  ```bash
  git clone git@github.com:daily-co/pipecat-moq-relay.git && cd pipecat-moq-relay
  docker build -t pipecat-moq-relay .        # add --build-arg MOQ_TAG=... if 3.3 said so
  sudo certbot certonly --standalone -d RELAY_HOST
  # The relay config reads /etc/moq/tls/{fullchain,privkey}.pem; copy the
  # cert there (-L dereferences certbot's symlinks). Re-run after renewals.
  sudo mkdir -p /etc/moq/tls
  sudo cp -L /etc/letsencrypt/live/RELAY_HOST/fullchain.pem /etc/letsencrypt/live/RELAY_HOST/privkey.pem /etc/moq/tls/
  sudo chmod 644 /etc/moq/tls/*.pem
  docker run -d --restart unless-stopped --name moq-relay \
    -p 443:443/udp -p 443:443/tcp \
    -v "$PWD/moq-relay.toml:/etc/moq/moq-relay.toml:ro" \
    -v /etc/moq/tls:/etc/moq/tls:ro \
    pipecat-moq-relay
  ```
  Gate (laptop):
  ```bash
  curl -s -o /dev/null -w '%{http_code}\n' http://RELAY_HOST/        # any HTTP status = TCP up
  /opt/homebrew/bin/openssl s_client -connect RELAY_HOST:443 -quic -alpn h3 </dev/null
  ```
  The openssl output must show the Let's Encrypt chain (`issuer=...Let's Encrypt`).

- [ ] **3.6 coturn.** On the box — write the benchmark TURN config
  (production variant of the local kit: real credential, no loopback peers):
  ```bash
  sudo mkdir -p /etc/coturn-bench && sudo tee /etc/coturn-bench/turnserver.conf >/dev/null <<'EOF'
  listening-port=3478
  min-port=49160
  max-port=49200
  lt-cred-mech
  user=pipecat:TURN_PASSWORD
  realm=RELAY_HOST
  fingerprint
  no-multicast-peers
  no-cli
  no-tls
  no-dtls
  verbose
  EOF
  docker run -d --restart unless-stopped --name coturn --network host \
    -v /etc/coturn-bench/turnserver.conf:/etc/coturn/turnserver.conf:ro \
    coturn/coturn:4.7 --external-ip='PUBLIC_IP/PRIVATE_IP'
  ```
  (`--external-ip` as an argument, not in the conf — the image auto-injects
  one otherwise, and EC2's NAT needs the `public/private` form.
  `PRIVATE_IP` = first address from `hostname -I`.)
  Gate: `docker logs coturn` shows listeners on 3478 with no config errors.
  Real verification is the smoke run in 4.2.

- [ ] **3.7 Record the deployment** (goes into the campaign notes):
  region, instance type, AMI, `PUBLIC_IP`, `RELAY_HOST`, relay `MOQ_TAG`,
  `coturn/coturn:4.7`.

## Phase 4 — Deployed relay tier

Tab layout: tab 1 = SSH tailing a relay's logs, tab 2 = harness.

- [ ] **4.1 moq-relay-deployed.**
  Tab 1: `ssh ubuntu@PUBLIC_IP docker logs -f moq-relay`
  Tab 2 — smoke (1 × 15 s), then the real run:
  ```bash
  uv run python benchmarks/transport_latency/transport_latency.py \
      --scenario moq-relay-deployed --relay-url https://RELAY_HOST/anon \
      --trials 1 --duration 15
  uv run python benchmarks/transport_latency/transport_latency.py \
      --scenario moq-relay-deployed --relay-url https://RELAY_HOST/anon
  ```
  Gate: smoke completes with n > 0 (else check `results/bot-moq-relay-deployed-1.log`
  and the relay logs); real run n ≥ 55 per trial; sessions visible in tab 1.
  Note: the 15 s smoke JSON is overwritten by the real trial 1 — run the
  real thing immediately so the campaign keeps only 60 s trials.

- [ ] **4.2 webrtc-turn-deployed.**
  Tab 1: `ssh ubuntu@PUBLIC_IP docker logs -f coturn`
  Tab 2 — smoke, then real:
  ```bash
  uv run python benchmarks/transport_latency/transport_latency.py \
      --scenario webrtc-turn-deployed \
      --turn-url turn:PUBLIC_IP:3478 --turn-username pipecat --turn-credential TURN_PASSWORD \
      --trials 1 --duration 15
  uv run python benchmarks/transport_latency/transport_latency.py \
      --scenario webrtc-turn-deployed \
      --turn-url turn:PUBLIC_IP:3478 --turn-username pipecat --turn-credential TURN_PASSWORD
  ```
  Gate: JSON `ice_pair` = `relay/relay` (both sides through the AWS coturn);
  n ≥ 55 per trial. Nonzero drops are **expected** on a real TURN path —
  count them, don't chase them.

- [ ] **4.3 Tier review.** `results/chart-deployed-relay.png` renders; both
  scenarios' excess includes one real internet round trip to the box (they
  should differ from each other by protocol behavior, not by tens of ms of
  geography — they share the box).

## Phase 5 — Analysis + record keeping

- [ ] **5.1 Final render** (idempotent; regenerates summary + all charts):
  ```bash
  uv run python benchmarks/transport_latency/charts.py
  ```
  Gate: `results/summary.md` has 6 scenario rows × 3 trials + 2 floor rows;
  `chart-all.png` plus the three tier sub-charts exist.

- [ ] **5.2 Cross-checks** before believing the numbers:
  - `bot-internal p50` comparable across all six scenarios (pipeline canary);
  - each transport's floor unchanged from Phase 0 if re-run;
  - within each tier, compare **excess over own floor**, not raw p50
    (see "Reading the numbers" in the README);
  - drops concentrated where expected (deployed TURN), ~0 elsewhere.

- [ ] **5.3 Preserve the campaign.** `results/` is gitignored; copy the
  outputs to a tracked campaign dir and commit:
  ```bash
  export CAMPAIGN=benchmarks/transport_latency/campaigns/$(date +%Y-%m-%d)
  mkdir -p $CAMPAIGN
  cp benchmarks/transport_latency/results/*.json \
     benchmarks/transport_latency/results/summary.md \
     benchmarks/transport_latency/results/chart-*.png $CAMPAIGN/
  ```

- [ ] **5.4 Write up.** Update [RESULTS.md](RESULTS.md): headline table
  (p50 RTT, excess over own floor, jitter, drops per scenario), environment
  block from any trial JSON, findings narrative. Commit `RESULTS.md` +
  `campaigns/<date>/` together with the code state from 0.1.

- [ ] **5.5 Park the AWS box.** Between campaigns, stop the instance
  (Elastic IP persists; note it bills a small hourly fee while the instance
  is stopped). Terminate + release when the deployed tier is done for good.

# Campaign commands, in order

Every command for one full campaign, bare. Gates, tab layout, and
troubleshooting live in [RUNBOOK.md](RUNBOOK.md) — read it first; use this
sheet on later runs. All commands from the pipecat repo root; `# tab N`
means a separate foreground terminal. Placeholders (`RELAY_HOST`, `PUBLIC_IP`,
`PRIVATE_IP`, `TURN_PASSWORD`) are defined in the runbook.

```bash
# ---- Phase 0: preflight + clean slate + floors -----------------------------
uv sync --group bench --extra moq --extra webrtc --extra runner    # one-time
git status --short && git rev-parse --short HEAD
cd benchmarks/transport_latency && [ -d results ] && mv results results-archive-$(date +%Y%m%d-%H%M); mkdir -p results && cd ../..
uv run python benchmarks/transport_latency/transport_latency.py --floors

# ---- Phase 1: local direct -------------------------------------------------
uv run python benchmarks/transport_latency/transport_latency.py --scenario webrtc-local
uv run python benchmarks/transport_latency/transport_latency.py --scenario moq-serve

# ---- Phase 2: local relay (docker) ----------------------------------------
../moq-relay-dev.sh relay                                                # tab 1
uv run python benchmarks/transport_latency/transport_latency.py --scenario moq-relay-local
# ctrl-c tab 1
../../daily-co/pipecat-coturn/run.sh                                     # tab 1
uv run python benchmarks/transport_latency/transport_latency.py --scenario webrtc-turn-local
# ctrl-c tab 1

# ---- Phase 3: cloud box (one-time) ----------------------------------------
# Provision instance + open ports first — cloud-specific, console/CLI:
#   AWS: RUNBOOK.md §3.1–3.2      OCI: deploy-oci.md
# Then DNS: A record RELAY_HOST -> PUBLIC_IP; verify:
dig +short RELAY_HOST
# Version pin check (laptop) — moq-ffi 0.2.30 -> MOQ_TAG moq-relay-v0.13.5:
uv run python -c "import importlib.metadata as m; print(m.version('moq-ffi'))"

# On the box (identical for AWS/OCI Ubuntu 24.04) — ssh ubuntu@PUBLIC_IP:
sudo apt-get update && sudo apt-get install -y docker.io certbot
sudo usermod -aG docker ubuntu && exit    # re-SSH
git clone git@github.com:daily-co/pipecat-moq-relay.git && cd pipecat-moq-relay
docker build -t pipecat-moq-relay .       # --build-arg MOQ_TAG=... if pin differs
sudo certbot certonly --standalone -d RELAY_HOST
sudo mkdir -p /etc/moq/tls
sudo cp -L /etc/letsencrypt/live/RELAY_HOST/fullchain.pem /etc/letsencrypt/live/RELAY_HOST/privkey.pem /etc/moq/tls/
sudo chmod 644 /etc/moq/tls/*.pem
docker run -d --restart unless-stopped --name moq-relay \
  -p 443:443/udp -p 443:443/tcp \
  -v "$PWD/moq-relay.toml:/etc/moq/moq-relay.toml:ro" \
  -v /etc/moq/tls:/etc/moq/tls:ro \
  pipecat-moq-relay
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

# Verify from the laptop:
curl -s -o /dev/null -w '%{http_code}\n' http://RELAY_HOST/
/opt/homebrew/bin/openssl s_client -connect RELAY_HOST:443 -quic -alpn h3 </dev/null

# ---- Phase 4: deployed relay tier -----------------------------------------
ssh ubuntu@PUBLIC_IP docker logs -f moq-relay                            # tab 1
uv run python benchmarks/transport_latency/transport_latency.py \
    --scenario moq-relay-deployed --relay-url https://RELAY_HOST/anon \
    --trials 1 --duration 15                                             # smoke
uv run python benchmarks/transport_latency/transport_latency.py \
    --scenario moq-relay-deployed --relay-url https://RELAY_HOST/anon
ssh ubuntu@PUBLIC_IP docker logs -f coturn                               # tab 1
uv run python benchmarks/transport_latency/transport_latency.py \
    --scenario webrtc-turn-deployed \
    --turn-url turn:PUBLIC_IP:3478 --turn-username pipecat --turn-credential TURN_PASSWORD \
    --trials 1 --duration 15                                             # smoke
uv run python benchmarks/transport_latency/transport_latency.py \
    --scenario webrtc-turn-deployed \
    --turn-url turn:PUBLIC_IP:3478 --turn-username pipecat --turn-credential TURN_PASSWORD

# ---- Phase 5: analysis + record keeping -----------------------------------
uv run python benchmarks/transport_latency/charts.py
export CAMPAIGN=benchmarks/transport_latency/campaigns/$(date +%Y-%m-%d)
mkdir -p $CAMPAIGN
cp benchmarks/transport_latency/results/*.json \
   benchmarks/transport_latency/results/summary.md \
   benchmarks/transport_latency/results/chart-*.png $CAMPAIGN/
# then: update RESULTS.md and commit it with campaigns/<date>/
```

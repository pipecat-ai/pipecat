# Relay box on OCI (alternative to RUNBOOK Phase 3.1–3.2)

Same plan as the AWS path: **one instance runs both moq-relay and coturn**, so
the two deployed scenarios share a box, region, and network path. Only the
provisioning differs; once you can SSH in, RUNBOOK steps 3.3–3.7 apply
unchanged except where noted below. Placeholders as in
[RUNBOOK.md](RUNBOOK.md) (`PUBLIC_IP` here = the OCI reserved public IP).

## 1. Instance

- **Shape**: `VM.Standard.A1.Flex`, 1 OCPU / 6 GB (Ampere arm64 — Always Free
  eligible). Everything this box runs is multi-arch: the relay builds natively
  on ARM (`rust:1-bookworm` build stage) and `coturn/coturn:4.7` publishes
  arm64 images. If A1 capacity is exhausted in your availability domain
  (common — "Out of capacity" at launch), retry another AD or fall back to a
  small x86 `VM.Standard.E4.Flex`.
- **Image**: Canonical Ubuntu 24.04 (aarch64 build for A1). SSH user: `ubuntu`.
- **Region**: nearest to the measuring laptop.

## 2. Public IP

Instances get an *ephemeral* public IP by default, which is lost on stop.
Assign a **reserved** public IP so `RELAY_HOST` DNS survives instance stops:
Networking → IP management → Reserved public IPs → create, then edit the
instance VNIC's IPv4 address and attach it. That address is `PUBLIC_IP`.

## 3. VCN + security list

The instance-creation wizard's default VCN (public subnet + internet gateway)
is fine. Add **stateful ingress rules** to the subnet's security list
(source `0.0.0.0/0` except SSH, which you should restrict to your IP):

| port | proto | purpose |
|---|---|---|
| 22 | TCP | SSH (your IP only) |
| 80 | TCP | certbot challenge |
| 443 | TCP | relay HTTP/WebSocket fallback |
| 443 | UDP | relay QUIC/WebTransport |
| 3478 | UDP | coturn STUN/TURN |
| 49160–49200 | UDP | coturn relay allocations |

## 4. OS firewall — the OCI gotcha

Oracle's Ubuntu images ship **iptables rules that reject everything except
SSH**, independent of the security list — opening ports in the console alone
is not enough. On the box, before starting services:

```bash
sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 443 -j ACCEPT
sudo iptables -I INPUT -p udp --dport 443 -j ACCEPT
sudo iptables -I INPUT -p udp --dport 3478 -j ACCEPT
sudo iptables -I INPUT -p udp --dport 49160:49200 -j ACCEPT
sudo netfilter-persistent save
```

## 5. Services — run both on the host network

To keep the baked-in firewall rules the only iptables variable, run the relay
with `--network host` on OCI (it listens on `[::]:443` per its config; coturn
already uses host networking in the RUNBOOK). In RUNBOOK 3.5, replace the
relay's `docker run` with:

```bash
docker run -d --restart unless-stopped --name moq-relay --network host \
  -v "$PWD/moq-relay.toml:/etc/moq/moq-relay.toml:ro" \
  -v /etc/moq/tls:/etc/moq/tls:ro \
  pipecat-moq-relay
```

Everything else in 3.3–3.7 (version pin, certbot, cert copy, coturn config
with `--external-ip='PUBLIC_IP/PRIVATE_IP'`, verification from the laptop) is
identical. `PRIVATE_IP` comes from `hostname -I` (a `10.0.x.x` address on the
default VCN).

## 6. Cost + parking

A1.Flex within the Always-Free limits costs nothing while running; egress at
benchmark volumes is far inside the free 10 TB/month. Parking between
campaigns: stop the instance (the reserved public IP persists). Terminate the
instance and release the reserved IP when the deployed tier is done for good.

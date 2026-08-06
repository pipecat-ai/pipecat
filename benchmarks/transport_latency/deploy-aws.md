# Relay box on AWS (provisioning for RUNBOOK Phase 3)

One EC2 instance runs **both moq-relay and coturn**, so the two deployed
scenarios share a box, region, and network path. This doc covers AWS
provisioning only; once you can SSH in, RUNBOOK steps 3.2–3.7 apply
unchanged. OCI counterpart: [deploy-oci.md](deploy-oci.md). Placeholders as
in [RUNBOOK.md](RUNBOOK.md) (`PUBLIC_IP` here = the Elastic IP).

## 1. Instance

- **Type**: `t3.small` (any small x86 shape works; the relay is one Rust
  binary and coturn is I/O-bound).
- **AMI**: Ubuntu Server 24.04 LTS. SSH user: `ubuntu`.
- **Region**: nearest to the measuring laptop.
- Launch with your SSH key pair.

## 2. Public IP

Allocate an **Elastic IP** and associate it with the instance — the default
auto-assigned public IP changes on stop/start, which would break the
`RELAY_HOST` DNS record and the Let's Encrypt cert. That address is
`PUBLIC_IP`.

## 3. Security group

Inbound rules (source `0.0.0.0/0` except SSH, which you should restrict to
your IP):

| port | proto | purpose |
|---|---|---|
| 22 | TCP | SSH (your IP only) |
| 80 | TCP | certbot challenge |
| 443 | TCP | relay HTTP/WebSocket fallback |
| 443 | UDP | relay QUIC/WebTransport |
| 3478 | UDP | coturn STUN/TURN |
| 49160–49200 | UDP | coturn relay allocations |

Outbound: the default allow-all is fine.

## 4. OS firewall

Nothing to do — Ubuntu AMIs ship with no restrictive iptables rules and ufw
disabled, so the security group is the only packet filter. (This is the main
provisioning difference from OCI, whose images firewall themselves — see
deploy-oci.md §4.)

## 5. Services

The RUNBOOK's docker commands (3.5 moq-relay with `-p` port mappings, 3.6
coturn with `--network host`) work as written. `PRIVATE_IP` for coturn's
`--external-ip='PUBLIC_IP/PRIVATE_IP'` comes from `hostname -I`
(a `172.31.x.x` address on the default VPC).

## 6. Cost + parking

Between campaigns, stop the instance — the Elastic IP persists but bills a
small hourly fee while unattached to a running instance. Terminate the
instance and release the Elastic IP when the deployed tier is done for good.

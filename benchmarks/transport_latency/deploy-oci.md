# Relay box on OCI (provisioning for RUNBOOK Phase 3)

One instance runs **both moq-relay and coturn**, so the two deployed
scenarios share a box, region, and network path. This doc covers OCI
provisioning only; once you can SSH in, RUNBOOK steps 3.2–3.7 apply
unchanged except where noted below. AWS counterpart:
[deploy-aws.md](deploy-aws.md). Placeholders as in [RUNBOOK.md](RUNBOOK.md)
(`PUBLIC_IP` here = the OCI reserved public IP).

Provisioning is written for the `oci` CLI (`brew install oci-cli`, then
`oci setup config` for persistent API-key auth; smoke test with
`oci os ns get`). Every step has a console equivalent if you prefer
clicking. The CLI's region comes from the `~/.oci/config` profile — use
the region nearest the measuring laptop.

## 0. Session variables

```bash
export C=ocid1.compartment.oc1..aaaaaaaarifmqaluicalbafsish2a3yqzvx6ueozrj6v25th3xqsnjrpfyia  # tenancy root OCID (ocid1.tenancy.oc1..…) works
oci iam availability-domain list -c $C --query 'data[].name'
export AD='<one name from the list>'   # e.g. 'Xxxx:US-SANJOSE-1-AD-1'
```

## 1. Network — VCN, internet gateway, subnet, security list

```bash
VCN=$(oci network vcn create -c $C --cidr-blocks '["10.0.0.0/16"]' \
--display-name relay-vcn --wait-for-state AVAILABLE \
--query data.id --raw-output)
RT=$(oci network vcn get --vcn-id $VCN --query 'data."default-route-table-id"' --raw-output)
SL=$(oci network vcn get --vcn-id $VCN --query 'data."default-security-list-id"' --raw-output)

IGW=$(oci network internet-gateway create -c $C --vcn-id $VCN --is-enabled true \
--display-name relay-igw --wait-for-state AVAILABLE --query data.id --raw-output)
oci network route-table update --rt-id $RT --force --route-rules \
'[{"destination":"0.0.0.0/0","destinationType":"CIDR_BLOCK","networkEntityId":"'$IGW'"}]'

SUBNET=$(oci network subnet create -c $C --vcn-id $VCN --cidr-block 10.0.0.0/24 \
--display-name relay-subnet --wait-for-state AVAILABLE \
--query data.id --raw-output)
```

Replace the default security list's ingress rules (stateful; egress
allow-all is left untouched). Ports, with SSH restricted to your IP
(`YOUR_IP`) and the ICMP path-MTU rules the console default ships:

| port | proto | purpose |
|---|---|---|
| 22 | TCP | SSH (your IP only) |
| 80 | TCP | certbot challenge |
| 443 | TCP | relay HTTP/WebSocket fallback |
| 443 | UDP | relay QUIC/WebTransport |
| 3478 | UDP | coturn STUN/TURN |
| 49160–49200 | UDP | coturn relay allocations |

```bash
oci network security-list update --security-list-id $SL --force \
  --ingress-security-rules '[
  {"protocol":"6","source":"184.186.29.106/32","tcpOptions":{"destinationPortRange":{"min":22,"max":22}}},
  {"protocol":"6","source":"0.0.0.0/0","tcpOptions":{"destinationPortRange":{"min":80,"max":80}}},
  {"protocol":"6","source":"0.0.0.0/0","tcpOptions":{"destinationPortRange":{"min":443,"max":443}}},
  {"protocol":"17","source":"0.0.0.0/0","udpOptions":{"destinationPortRange":{"min":443,"max":443}}},
  {"protocol":"17","source":"0.0.0.0/0","udpOptions":{"destinationPortRange":{"min":3478,"max":3478}}},
  {"protocol":"17","source":"0.0.0.0/0","udpOptions":{"destinationPortRange":{"min":49160,"max":49200}}},
  {"protocol":"1","source":"0.0.0.0/0","icmpOptions":{"type":3,"code":4}},
  {"protocol":"1","source":"10.0.0.0/16","icmpOptions":{"type":3}}
]'
```

## 2. Instance

**Shape**: `VM.Standard.A1.Flex`, 1 OCPU / 6 GB (Ampere arm64 — Always Free
eligible). Everything this box runs is multi-arch: the relay builds natively
on ARM (`rust:1-bookworm` build stage) and `coturn/coturn:4.7` publishes
arm64 images. **Image**: latest Canonical Ubuntu 24.04 aarch64 build
compatible with A1 (the `--shape` filter below selects for that). SSH user:
`ubuntu`.

```bash
IMAGE=$(oci compute image list -c $C \
  --operating-system "Canonical Ubuntu" --operating-system-version 24.04 \
  --shape VM.Standard.A1.Flex --sort-by TIMECREATED --sort-order DESC \
  --query 'data[0].id' --raw-output)

INSTANCE=$(oci compute instance launch -c $C --availability-domain "$AD" \
  --shape VM.Standard.A1.Flex --shape-config '{"ocpus":1,"memoryInGBs":6}' \
  --image-id $IMAGE --subnet-id $SUBNET --assign-public-ip false \
  --ssh-authorized-keys-file ~/.ssh/id_ed25519.pub \
  --display-name moq-relay --wait-for-state RUNNING \
  --query data.id --raw-output)
```

`--assign-public-ip false` because §3 attaches a reserved IP instead — an
ephemeral one would be lost on stop and take `RELAY_HOST` DNS with it.

If the launch fails with "Out of capacity" (common for A1), retry with
another `$AD` from the §0 list, or fall back to a small x86
`VM.Standard.E4.Flex` (drop the `--shape` filter from the image lookup and
pick an x86_64 build).

## 3. Reserved public IP

```bash
VNIC=$(oci compute instance list-vnics --instance-id $INSTANCE \
  --query 'data[0].id' --raw-output)
PRIVIP=$(oci network private-ip list --vnic-id $VNIC \
  --query 'data[0].id' --raw-output)
oci network public-ip create -c $C --lifetime RESERVED --private-ip-id $PRIVIP \
  --display-name relay-ip --query 'data."ip-address"' --raw-output
```

The printed address is `PUBLIC_IP` — it survives instance stops. Gate:
`ssh -o StrictHostKeyChecking=accept-new ubuntu@PUBLIC_IP` works.
Then `ssh ubuntu@PUBLIC_IP` works.

## 4. OS firewall — the OCI gotcha

Oracle's Ubuntu images ship **iptables rules that reject everything except
SSH**, independent of the security list — opening ports in the security
list alone is not enough. On the box, before starting services:

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
campaigns — stop the instance (the reserved public IP persists):

```bash
oci compute instance action --instance-id $INSTANCE --action STOP
```

When the deployed tier is done for good, terminate the instance and release
the reserved IP:

```bash
oci compute instance terminate --instance-id $INSTANCE   # add --preserve-boot-volume false to also drop the disk
oci network public-ip list --scope RESERVED -c $C --query 'data[].{ip:"ip-address",id:id}'
oci network public-ip delete --public-ip-id <id from above>
```

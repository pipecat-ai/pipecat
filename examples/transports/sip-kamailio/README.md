# SIP bot registered through Kamailio

This example pairs the native SIP transport with a minimal Kamailio
registrar: the bot sends its REGISTER to Kamailio, Kamailio stores the
registration and routes inbound calls to the bot, and RTP media flows
directly between caller and bot (Kamailio is signaling-only).

```
caller ──INVITE──> Kamailio ──lookup(location)──> bot
                     │                              │
                     └────── signaling only ────────┘
media: caller <────────── direct RTP ──────────────> bot
```

This is the carrier-style topology: the registration edge is separate
from the PBX. FreeSWITCH (or any PBX) can sit behind Kamailio for
transcoding, recording, conferencing, and transfer handling while
Kamailio owns the registration table and call routing.

## Requirements

- `pipecat-ai[sip,kokoro]` (bot; see `bot.py`)
- Docker (for the Kamailio registrar)

## Run Kamailio

```bash
cd examples/transports/sip-kamailio
docker run -d --name kamailio -p 5060:5060/udp -p 5060:5060/tcp \
  -v "$(pwd)/kamailio.cfg:/etc/kamailio/kamailio.cfg:ro" \
  kamailio/kamailio:latest
```

The config uses only stable core registrar/nathelper directives and works
with any 5.x/6.x release.

The bundled `kamailio.cfg` is a minimal registrar: REGISTERs are stored
in memory, INVITEs for registered users are relayed to the bot's
NAT-received address, unregistered targets get 480. No digest auth —
fine for the lab, add the auth module for anything reachable beyond
localhost.

## Run the bot

```bash
export SIP_USER=5001
export SIP_DOMAIN=kamailio.lab          # AOR domain (realm)
export SIP_PASSWORD=bot-password        # unused here; cfg has no auth
export KAMAILIO_REGISTRAR=<kamailio-host>:5060
python bot.py
```

Wait for `Registered to SIP server, AOR = sip:5001@kamailio.lab`.

## Call the bot

Register any SIP phone (e.g. Linphone) to the same Kamailio as user
`1001` and dial `5001`. The INVITE hits the location table, Kamailio
relays to the bot's NAT-received address, the bot answers, speaks its
greeting, and hangs up.

To verify routing without a phone, a scripted INVITE works too — point
it at the registrar with R-URI `sip:5001@kamailio.lab` and an
`X-AICC-Leg: agent` style header only if your config dispatches on one;
the bundled cfg routes on the R-URI user alone.

## NAT

The bot re-REGISTERs every 25 seconds (`reg_interval`) so the NAT's UDP
flow stays open, and sends `+sip.instance` (RFC 5626) so a restarted bot
replaces its stale binding instead of stacking contacts. On the server
side, `force_rport()` + `fix_nated_register()` record the NAT-received
address and `handle_ruri_alias()` relays the INVITE there.

## Scaling out

One bot process handles one call at a time. For multiple concurrent
calls run several bot processes: multiple `SIPConnection`s sharing one
AOR in a single process auto-dispatch inbound calls to the first idle
connection (486 when all are busy); across hosts, put a dispatcher in
front (or the `dmq_usrloc` module to replicate the location table for
multi-active registrars).

## Production hardening

- digest auth (`auth` + `pv_auth_check`), TLS (WSS for browser agents)
- `dispatcher` for load-balancing bot farms, `dmq_usrloc` for HA
- a media component (`rtpengine`) when media must be anchored (NAT/LBG,
  recording, codec transcoding)

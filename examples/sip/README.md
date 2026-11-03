# Testing the SIP transport

## What this is

A dev-facing test harness for the baresip-backed `SIPTransport`. It provisions
everything a SIP call needs over REST, runs one bot that works in both
directions, and prints numbered call-progress milestones at INFO so a working
call is visible rather than silent. Run it to hear the transport work, and to
learn the telephony topology it sits in — SIP clients, a registrar, a trunk,
the PSTN.

Two tiers, each self-contained:

| Tier | Needs | Costs | Call clients | Proves |
|---|---|---|---|---|
| 0 | `DAILY_API_KEY` + Deepgram/OpenAI/Cartesia keys | nothing | macOS Telephone.app (or Linphone/Zoiper) registered as a Daily SIP client | REGISTER, INVITE, two-way RTP, DTMF, dial-in and dial-out — pure SIP, no PSTN |
| 1 | Tier 0 plus `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` | a number is ~$1.15/month, opt-in | your cell phone over the PSTN, plus a Twilio REST auto-caller with no human on the line | the same, across a real Elastic SIP trunk: origination, termination auth, IP ACL |

Files here:

| File | What it is |
|---|---|
| `bot.py` | the harness bot, dial-in and dial-out in one file |
| `milestones.py` | the numbered milestone event handlers, reusable in your own bot |
| `harness.py` | `provision` / `status` / `call` / `teardown`, owns `.state.json` |
| `twilio_api.py` | thin async httpx wrapper over the Twilio REST endpoints used here |
| `env.example` | the env surface, per tier |

Deploying these bots to Pipecat Cloud is out of scope here; this document
covers running them locally.

## The topology

```
                 TIER 1 (PSTN)                              TIER 0 (SIP only)
┌───────────┐        ┌─────────────────┐        ┌───────────────┐
│ your cell │        │  Twilio Elastic │        │   Daily SIP   │   REGISTER/INVITE (UDP)
│   phone   │──PSTN──│   SIP trunk     │──SIP───│   registrar   │◄───── Telephone.app
└───────────┘  [A]   └─────────────────┘  [B]   └───────┬───────┘       (softphone) [F]
                      origination ──►                   │ [C] SIP + RTP
  Twilio REST ────────┘ auto-caller [E]         ┌───────▼───────┐      ┌─────────────────┐
  (no human, inline TwiML <Say>)                │  baresip SIP  │──────│ pipecat pipeline│
                                                │   transport   │ [D]  │ STT → LLM → TTS │
                                                └───────────────┘      └─────────────────┘
```

The hops:

| Hop | What it is |
|---|---|
| A | PSTN: a real phone call reaching (or leaving) a Twilio number |
| B | Trunk ↔ Daily: the trunk's origination URI sends the call to the bot client's SIP URI; termination carries dial-out the other way |
| C | Registrar ↔ bot: REGISTER keeps the bot reachable at its AOR; the INVITE and the RTP stream ride this hop |
| D | Transport ↔ pipeline: decoded audio frames into STT, TTS audio back out |
| E | Twilio REST: a call created from the API with inline TwiML, no phone and no webhook |
| F | Softphone ↔ registrar: a second Daily SIP client registered from your desktop |

Which hops each path exercises — this is the point of running more than one
path:

| Path | Hops |
|---|---|
| Cell phone dials the trunk number (Tier 1) | A+B+C+D |
| Auto-caller `call dial-in --via sip` (Tier 1) | E+C+D |
| Auto-caller `call dial-in --via pstn` (Tier 1) | E+B+C+D |
| Softphone dials the bot (Tier 0) | F+C+D |
| Bot dials the softphone (Tier 0) | D+C+F |
| Bot dials a cell phone (Tier 1) | D+C+B+A |

A path that works tells you its hops are sound; a path that fails next to a
path that works localizes the fault to the hops they differ on. One-way audio
on A+B+C+D while F+C+D is clean, for instance, points at B — the trunk's
media path.

## Tier 0 quickstart

Softphone ↔ bot, Daily only. Fifteen minutes from a checkout, no money, no
ngrok.

### 1. Credentials

```bash
cp examples/sip/env.example examples/sip/.env
```

Fill `DAILY_API_KEY`, `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, `CARTESIA_API_KEY`.
Leave the Twilio block empty — Tier 0 never reads it.

### 2. Provision

```bash
uv run python examples/sip/harness.py provision
```

This creates two persistent Daily SIP clients — `pipecat-harness-bot`, which
the bot registers as, and `pipecat-harness-dev`, which your softphone
registers as — and prints the softphone's account block plus the bot's
dialable URI. Re-run it any time: an existing client is left alone, and a
client whose password is no longer in `.state.json` gets its password
rotated, so deleting the state file is recoverable. See the planned REST calls
without sending them with `--dry-run`.

```bash
uv run python examples/sip/harness.py status
```

prints the same values again as an OK/FIX checklist, fetched live.

### 3. Register the softphone

In Telephone.app: **Settings → Accounts → +**, then the four values
`provision` printed —

- **Username**: `pipecat-harness-dev`
- **Domain**: your Daily SIP domain, `<something>.sip.daily.co`
- **Password**: the printed password
- **Transport**: UDP

Daily accepts UDP for these accounts; Telephone.app's TLS support is not
something to rely on. If it will not register, use
[Linphone](https://www.linphone.org/) or
[Zoiper](https://www.zoiper.com/) instead — same four fields, same result.

### 4. Run the bot and call it

```bash
uv run python examples/sip/bot.py -t sip -d
```

Wait for `[1/6] REGISTER ok`, then dial the bot's URI from the softphone
exactly as `provision` printed it — `pipecat-harness-bot@<domain>.sip.daily.co`.
Expect ringing, a greeting, and a conversation. Press a dial-pad digit to
exercise DTMF; the bot logs `DTMF <tone>`. Hang up and the bot logs `[5/6]`
and `[6/6]`.

### 5. The other direction

Keep the softphone registered and generate the dial-out body:

```bash
uv run python examples/sip/harness.py call dial-out --to softphone
```

It writes `examples/sip/dialout-body.json` and prints the exact command to
run. The bot dials as soon as it is ready; answer in Telephone.app.

## Tier 1: real phones via Twilio

Adds a fully REST-provisioned Twilio Elastic SIP trunk, so calls reach the
bot from the PSTN and the bot can call a real phone.

### 1. Credentials

Add `TWILIO_ACCOUNT_SID` and `TWILIO_AUTH_TOKEN` to `examples/sip/.env`. Set
`TWILIO_CALLER_ID` as well if you want the auto-caller to use a number of
yours that is not the one attached to the trunk.

### 2. Provision the trunk

```bash
uv run python examples/sip/harness.py provision --tier 1
```

Tier 1 runs Tier 0 first, then upserts, all by the `pipecat-harness` naming
convention:

- a **trunk** `pipecat-harness` at `pipecat-harness-<8 hex>.pstn.twilio.com`
  (termination domains are globally unique, hence the suffix), with
  `Secure=false`
- an **origination URI** pointing at the bot client's SIP URI with
  `;transport=tls`, priority 10, weight 10 — this is what makes a call to the
  trunk land on the registered bot
- an **IP ACL** `pipecat-harness` holding this machine's current public IP,
  refreshed on every run because developer IPs churn
- a **credential list** `pipecat-harness` with a generated password, the
  primary dial-out auth path

Without a number flag it finishes and says `no number attached; dial-in from
PSTN unavailable` — the SIP-direct auto-caller still works.

### 3. A phone number (this is the part that costs money)

Attach one your account already owns:

```bash
uv run python examples/sip/harness.py provision --tier 1 --number +1XXXXXXXXXX
```

Or buy one. **This spends money** — roughly $1.15 per month until the number
is released. The command prints the price and asks for a `y` before
purchasing:

```bash
uv run python examples/sip/harness.py provision --tier 1 --buy-number --area-code 415
```

Teardown keeps the number by default; only `--release-number` gives it back.

### 4. Cell-phone dial-in

```bash
uv run python examples/sip/bot.py -t sip -d
```

Call the trunk's number (`status` prints it) from your phone. Hops A+B+C+D.

### 5. The auto-caller

No phone, no ears. Twilio places a call that reads one sentence to the bot:

```bash
uv run python examples/sip/harness.py call dial-in --via sip --duration 30
```

The call carries inline TwiML — `<Pause>`, then `<Say>This is the pipecat SIP
harness test call. Please say something back.</Say>` — so no webhook is
involved. The harness polls the call once a second, prints each status
transition with the matching bot milestone as a hint, hangs up at
`--duration`, and exits 0 only if the call reached `in-progress` and held at
least 5 seconds. The bot's own log should show that sentence transcribed.

`--via pstn` dials the trunk's number instead, exercising origination end to
end (hops E+B+C+D). It is best-effort: with a single number the `From` and
`To` are the same and Twilio rejects the call, which the harness warns about
before trying.

### 6. Cell-phone dial-out

```bash
uv run python examples/sip/harness.py call dial-out --to +1XXXXXXXXXX
```

Run the printed command. Dial-out goes direct to Twilio over the credential
list: the generated command sets `SIP_DOMAIN` to the trunk's termination
domain, `SIP_REG_INTERVAL=0` so the bot registers nowhere, and
`SIP_USER`/`SIP_PASS`/`SIP_AUTH_USER` to the credential, authenticating per
INVITE. The body carries a `P-Asserted-Identity` header naming the trunk
number, so the call presents a caller ID the PSTN accepts.

The command also sets `SIP_STUN_SERVER=off` — a registration-less account
cannot run `medianat=stun` (the dial fails instantly; see troubleshooting).
With STUN off the bot advertises its LAN address, so **turn on the trunk's
Symmetric RTP first** (Console > Elastic SIP Trunking > Trunks >
`pipecat-harness` > General) or the bot will not hear the callee.

## What you should see

Every milestone is one grep-able INFO line, `[N/6]`, emitted by
`milestones.py` except `[4/6]`, which `bot.py` logs where it queues the
greeting. A Tier 0 dial-in run reads:

```
[1/6] REGISTER ok — reachable at sip:pipecat-harness-bot@<domain>.sip.daily.co
[2/6] INVITE — call ringing
[3/6] answered, media flowing (first RTP received)
[4/6] pipeline started — bot is listening
DTMF 5
[5/6] call ended
[6/6] quality: duration=12s loss=0.0% jitter=3ms rtt=41ms
```

What each one means about the SIP machinery underneath:

| Line | What just happened |
|---|---|
| `[1/6] REGISTER ok` | The bot's REGISTER was accepted and the registrar now maps its AOR to this host's contact. Until this lands nothing can reach the bot by name. In trunk mode (`SIP_REG_INTERVAL=0`) there is no registration at all and the line reads `trunk mode, skipping REGISTER` — the peer reaches the bot directly instead. |
| `[2/6] INVITE — call ringing` | An INVITE arrived (or went out) and the call is in its provisional 180 Ringing state. Signaling works; media has not started. An auto-answered dial-in can jump straight to established, skipping this line. |
| `[3/6] answered, media flowing` | 200 OK, SDP negotiated, and the first RTP packet actually arrived. In this transport "connected" means answered *and* receiving audio, so this line is the one that rules out a NAT media failure. |
| `[4/6] pipeline started` | `on_client_connected` fired; the pipeline is running and the greeting `LLMRunFrame` is queued. Audio frames now reach STT. |
| `DTMF <tone>` | A dial-pad digit arrived as RFC 4733 telephone-event (or as SIP INFO, with `dtmf_mode="info"`). |
| `[5/6] call ended` | BYE in either direction, logged once even though the transport reports the end on up to three separate events. |
| `[6/6] quality` | The close-time stats from the media stack: duration, packet loss, jitter, round-trip time. Loss above a few percent or jitter in the hundreds of milliseconds explains choppy audio. |

Dial-out reads the same, with `[2/6]` naming the dialed target and `[3/6]`
coming from the answer instead of the INVITE.

Lines that are not milestones: `SIP audio warning: ...` for a stalling or
degraded stream, and `SIP error: ... — likely cause: ...`, which names the
most likely of bad credentials, an unreachable registrar, and NAT.

To see this list without any credentials at all:

```bash
uv run python examples/sip/milestones.py
```

## Troubleshooting

**The bot greets you, then greets you again a few seconds later.** The bot is
hearing its own voice. Watch the `heard:` lines in the bot terminal: the bot's
own words showing up there about a second after it spoke them means the far
end is feeding its audio back. On the softphone that is acoustic echo —
Telephone.app playing through the Mac's speakers while the built-in mic picks
them up, with no echo cancellation anywhere on the SIP leg (a WebRTC browser
would cancel this; a plain SIP client does not). The bot's VAD then treats the
echo as the user talking, interrupts the greeting, and the LLM greets again.
Fix: wear headphones on the softphone machine. To tell acoustic echo from
echo in the media path itself, mute the softphone's microphone — if the
repeats continue with the mic muted, the echo is upstream of you and worth a
bug report with the `heard:` lines.

**No ringing before the bot answers.** Expected on dial-in: the bot runs with
`auto_answer`, so it accepts the INVITE as soon as it arrives and the call can
jump straight to established — the caller hears no ringback and the `[2/6]`
milestone may never print. Ringing is a dial-out thing, where the far end
decides when to pick up.

**Dial-out through the trunk fails instantly with `dial failed: Address
family not supported by protocol family`.** `medianat=stun` cannot run on a
registration-less account (`SIP_REG_INTERVAL=0`): the STUN server lookup runs
at dial time, before any DNS has fixed the account's address family, and the
media session aborts before an INVITE is sent. Registered accounts (Tier 0,
softphone dial-out) are unaffected. `call dial-out` therefore puts
`SIP_STUN_SERVER=off` in the command it prints; with STUN off, return media
needs the trunk's Symmetric RTP (next entry).

**The call connects but audio flows only one way.** The far end is sending
RTP to an address the bot is not reachable at: its LAN address when STUN is
off (every trunk dial-out), or the STUN-reported address when the bot sits
behind a symmetric NAT. Symmetric RTP makes the trunk latch onto the bot's
real RTP source address instead. It is off by default and the Twilio REST API
cannot set it — `symmetric_rtp_enabled` is a read-only Trunk property, so
turn it on by hand at Console > Elastic SIP Trunking > Trunks >
`pipecat-harness` > General > Symmetric RTP. `status` reports which way it is
set. A TURN relay (`SIP_EXTRA_PARAMS=medianat=turn,...`) is the alternative
when the far end cannot latch.

**Nothing registers, or the INVITE goes unanswered.** Trace the signaling:

```bash
SIP_TRACE=1 SIP_NATIVE_LOG_LEVEL=debug uv run python examples/sip/bot.py -t sip -d
```

`SIP_TRACE=1` logs every SIP message verbatim, SDP included. **Those messages
carry authentication material** — never paste a trace into an issue or commit
one, and never enable it in production.

**The bot registers, then calls ring somewhere else.** Two processes
registered the same account; the registrar is holding two bindings and
picking the wrong one. A cloud deployment left running does this to a local
bot. Set a stable `SIP_INSTANCE_ID` (any fixed UUID) so a restarted bot
*replaces* its own binding (RFC 5626) instead of stacking a stale one, or
shut the other process down.

**Tier 1 dial-out fails with a 403 after it worked yesterday.** Your public IP
moved and the trunk's IP ACL no longer lists it. `status` prints the address
it sees next to the authorized one; `provision --tier 1` refreshes the entry.
The credential list is why dial-out usually survives this — it authenticates
per INVITE and does not care about addresses.

**Telephone.app quirks.** It registers over UDP, not TLS; dial the bare
`user@domain` URI, with no `sip:` prefix and no port; and it sometimes keeps
showing "registered" after the binding has lapsed — remove and re-add the
account to force a fresh REGISTER. Linphone and Zoiper are the fallbacks and
both show the raw SIP exchange, which makes them better for debugging.

**`status` says a credential password is not cached.** Twilio never returns a
credential password after creation. `provision --tier 1` resets it to a fresh
one and caches that.

## Advanced: the webhook trunk path

The harness uses two *persistent* Daily SIP clients because that needs no
public URL. Production works the other way: a Daily SIP trunk of type
`sip_client` provisions a SIP client *per inbound call* and POSTs its
credentials to the trunk's `webhook_url`, which keeps no long-lived
credential anywhere and scales to concurrent calls.

The dev runner stands in for the cloud handler at
`POST /daily-sip-trunk-webhook`, so you can run that path locally:

1. Create a Daily SIP trunk of type `sip_client` via the Daily REST API, with
   its `webhook_url` pointing at a public tunnel to your machine (ngrok, say)
   and the path `/daily-sip-trunk-webhook`. `DailyRESTHelper` has no trunk
   methods — this is a plain REST call, and it is why the automated tiers do
   not use this path.
2. Run the bot with the runner's server mode: `uv run python
   examples/sip/bot.py -t sip` (no `-d`).
3. Point a phone number at the trunk and call it.

Daily acknowledges trunk creation with a `sip_trunk.test` probe, which the
route answers without starting anything. For a real `sip_trunk.incoming`
notification it starts the bot on the per-call account and answers at once,
while the registrar holds the caller until the bot's REGISTER lands. The whole
notification reaches the bot as `runner_args.body`, and
`create_transport` picks the account out of `body["sip_client"]` with no code
change in the bot.

## Teardown and env reference

Delete what the harness created, in reverse order, treating anything already
gone as done:

```bash
uv run python examples/sip/harness.py teardown --tier all
```

`--tier 0` deletes only the two Daily SIP clients; `--tier 1` only the Twilio
resources (and warns if Tier 0 is left behind, since origination points at
the bot client). The phone number is detached and kept unless you pass
`--release-number`, which is what ends the monthly charge. `--dry-run` lists
the deletions first.

Harness env vars:

| Variable | Used by | Meaning |
|---|---|---|
| `DAILY_API_KEY` | both tiers | SIP-client CRUD, and the runner's temporary client when no account is set |
| `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, `CARTESIA_API_KEY` | `bot.py` | the STT → LLM → TTS cascade |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` | Tier 1 | Twilio REST basic auth |
| `TWILIO_CALLER_ID` | `call dial-in` | the `From` number, when no number is attached to the trunk |

SIP transport env vars the harness touches (the full surface is documented in
`pipecat.runner.sip`):

| Variable | Default | Meaning |
|---|---|---|
| `SIP_USER` / `SIP_PASS` / `SIP_DOMAIN` | `.state.json`'s `pipecat-harness-bot`, else unset | the account to register or authenticate as; `bot.py` exports the provisioned harness account when these are unset, and the runner provisions a temporary Daily client when there is no state either |
| `SIP_AUTH_USER` | the user part of the AOR | digest username when it differs — the credential-list trunk path |
| `SIP_REG_INTERVAL` | 600 | seconds between registration refreshes; `0` is trunk mode, no registration |
| `SIP_TRANSPORT` | `tls` for the harness account or a provisioned client, `udp` for any other env account | SIP signaling transport |
| `SIP_DIALOUT_URI` | unset | dial this URI on start; the alternative to a `dialout_settings` runner body |
| `SIP_INSTANCE_ID` | unset | stable RFC 5626 `+sip.instance`, so a restart replaces its own binding |
| `SIP_STUN_SERVER` | `stun:stun.l.google.com:19302` | STUN server for `medianat=stun`; `off` disables media-NAT traversal |
| `SIP_EXTRA_PARAMS` | unset | raw baresip account parameters, comma-separated `key=value`; wins over `SIP_STUN_SERVER` |
| `SIP_NATIVE_LOG_LEVEL` | `warning` | lowest severity captured from the native stack |
| `SIP_TRACE` | unset | `1` logs every SIP message verbatim; contains auth material |

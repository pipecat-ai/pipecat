# Pipecat SIP Examples

Voice bots on a phone call over the baresip-backed `SIPTransport`. Each
example is a complete, runnable bot: Dial-in & Dial-out.

## Setup

From the repo root:

```bash
uv sync --all-extras
brew install baresip
```
or
```bash
uv sync --all-extras
apt install baresip
```

Copy the env template and fill in your API keys:

```bash
cp examples/sip/env.example examples/sip/.env
```

## Environment variables

| Variable | Required by |
|---|---|
| `DAILY_API_KEY` | both call types: SIP-client CRUD, and the runner's temporary client when no account is set |
| `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, `CARTESIA_API_KEY` | the bots' STT → LLM → TTS cascade |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` | PSTN: Twilio REST basic auth |
| `TWILIO_CALLER_ID` | `call dial-in`: the `From` number, when no number is attached to the trunk |

## Provisioning

`provision.py` owns the auxiliary infrastructure. Every resource it creates
carries the `pipecat-harness` name prefix, so `status` can find them live and
`teardown` never touches anything else. See the planned REST calls without
sending them with `--dry-run` on any command.

Two call types: [Softphone calls](#softphone-calls-two-daily-sip-clients) and [PSTN calls](#PSTN-calls-a-Twilio-Elastic-SIP-trunk)

### Softphone calls: two Daily SIP clients

> [!NOTE]
> The softphone is deliberately a separate install rather than the
> [`baresip-python`](https://github.com/daily-co/baresip-python) package the transport itself runs on (whose wheels do ship
> hardware audio drivers). Two copies of one stack can cancel out their own
> bugs; keeping the far end an independently built client means a fault on a
> call points at one side.

#### 1. Provision the SIP clients

```bash
uv run python examples/sip/util/provision.py provision
```

This creates two persistent Daily SIP clients:
- `pipecat-harness-bot` which the bots register as
- `pipecat-harness-dev` which your softphone registers as

It also prints:
- the softphone's account block
- the bot's dialable URI

Re-run it any time as it is idempotent and deleting the state file 
`.state.json` is recoverable.

```bash
uv run python examples/sip/util/provision.py status
```

Prints the same values again as an OK/FIX checklist, fetched live.

#### 2. Launch the softphone

```bash
uv run python examples/sip/util/provision.py softphone
```

This writes a [baresip](https://github.com/baresip/baresip) config registered as
the dev client and launches it. The terminal it runs in becomes the softphone.

`baresip` crash-course:
- `d <uri>` dials
- `a` answers
- `b` hangs up
- digits send DTMF
- `q` quits

### PSTN calls: a Twilio Elastic SIP trunk

Adds a fully REST-provisioned trunk, so calls reach the bot from the PSTN and
the bot can call a real phone.

#### 1. Provision PSTN

```bash
uv run python examples/sip/util/provision.py provision --pstn
```

`--pstn` provisions the Daily clients first, then upserts, all by the
`pipecat-harness` naming convention:

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

#### 2. Attach phone number

Attach a phone number owned by your Twilio account:

```bash
uv run python examples/sip/util/provision.py provision --pstn --number +1XXXXXXXXXX
```

### Run an example

| Example                  | What it shows                                                          |
| ------------------------ | ---------------------------------------------------------------------- |
| [`dial-in/`](dial-in/)   | The bot registers and answers calls — from a softphone or a real phone. |
| [`dial-out/`](dial-out/) | The bot places a call — to a softphone or a real phone.               |

Both run over either of two call types, each self-contained:

| Call type | Needs | Call clients | Proves |
|---|---|---|---|
| **💻 Softphone** on the same Daily domain | `DAILY_API_KEY` + Deepgram/OpenAI/Cartesia keys | a softphone registered as a Daily SIP client — baresip via `provision.py softphone`, or Telephone.app/Linphone/Zoiper | REGISTER, INVITE, two-way RTP, DTMF, dial-in and dial-out — pure SIP, no PSTN |
| **📞 PSTN** via a Twilio SIP trunk | the softphone row plus `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` | your cell phone over the PSTN, plus a Twilio REST auto-caller with no human on the line | the same, across a real Elastic SIP trunk: origination, termination auth, IP ACL |

### Provisioning teardown

Delete what provisioning created, in reverse order, treating anything already
gone as done:

```bash
uv run python examples/sip/util/provision.py teardown --scope all
```

`--scope daily` deletes only the two Daily SIP clients; `--scope pstn` only
the Twilio resources (and warns if the Daily clients are left behind, since
origination points at the bot client). The phone number is detached and kept unless you pass
`--release-number`, which is what ends the monthly charge. `--dry-run` lists
the deletions first.


### baresip alternative option: GUI softphone

[Telephone.app](https://apps.apple.com/us/app/telephone/id406825478) (macOS), [Linphone](https://www.linphone.org/), or [Zoiper](https://www.zoiper.com/) replaces baresip as the softphone. In the app's account settings (Telephone.app: **Settings → Accounts → +**), enter the four values `provision` printed:

- **Username**: `pipecat-harness-dev`
- **Domain**: your Daily SIP domain, `<something>.sip.daily.co`
- **Password**: the printed password
- **Transport**: UDP

Daily accepts UDP for these accounts. 
Telephone.app's TLS support is not something to rely on.

## Terminology & Topology

| Term | Meaning |
|---|---|
| **SIP** | the protocol phone systems use to set up and end calls — who is calling whom, ringing, answer, hangup; the audio itself flows separately (as RTP) |
| **softphone** | a desktop app that acts as a phone: mic, speakers, dial pad — but the network instead of a phone line |
| **registrar** | the server a SIP client signs in to (a SIP `REGISTER`), so that calls to its name can find it |
| **PSTN** | the public switched telephone network — the thing that makes real phones ring |
| **SIP trunk** | a bridge between SIP and the PSTN: it holds phone numbers on one side and converts calls in both directions |

Four manual call paths follow, one chart each. Every chart reads top to
bottom in the same order — far end first, bot's pipeline last — so the four
are easy to compare. Arrows show who places the call; once answered, audio
flows both ways on every leg. Each leg carries a letter, `[A]`–`[G]`, used
by the fault-localization table at the end.

### Softphone dial-in — you dial the bot

```
softphone — baresip or Telephone.app, registered as pipecat-harness-dev
     │
     │  [F] SIP INVITE (UDP), then RTP audio
     ▼
Daily SIP registrar
     │
     │  [C] SIP + RTP — the bot's REGISTER makes it reachable here
     ▼
baresip SIP transport — the bot, registered as pipecat-harness-bot
     │
     │  [D] decoded audio frames, in-process (both ways)
     ▼
pipecat pipeline — STT → LLM → TTS
```

### Softphone dial-out — the bot dials you

Same three legs, INVITE in the other direction. The bot's own registration
carries the outbound call.

```
softphone — rings; answer with `a` in baresip
     ▲
     │  [F] SIP INVITE, then RTP audio
     │
Daily SIP registrar
     ▲
     │  [C] the registered bot sends the INVITE here
     │
baresip SIP transport — dials on start (`SIP_DIALOUT_URI` or runner body)
     │
     │  [D] decoded audio frames, in-process (both ways)
     ▼
pipecat pipeline — STT → LLM → TTS
```

### PSTN dial-in — your cell phone dials the trunk's number

```
your cell phone
     │
     │  [A] PSTN — a real call to the trunk's Twilio number
     ▼
Twilio Elastic SIP trunk
     │
     │  [B] origination URI — the trunk forwards the call to the bot client's SIP URI
     ▼
Daily SIP registrar
     │
     │  [C] SIP + RTP — the bot's REGISTER makes it reachable here
     ▼
baresip SIP transport — the bot, registered as pipecat-harness-bot
     │
     │  [D] decoded audio frames, in-process (both ways)
     ▼
pipecat pipeline — STT → LLM → TTS
```

### PSTN dial-out — the bot dials a cell phone

No Daily registrar on this path: the bot INVITEs the trunk's termination
domain directly, authenticating each INVITE against the trunk's credential
list (`SIP_REG_INTERVAL=0`, no registration).

```
your cell phone — rings
     ▲
     │  [A] PSTN — the trunk places a real call, presenting the trunk's number
     │
Twilio Elastic SIP trunk
     ▲
     │  [G] termination — SIP INVITE straight to <trunk>.pstn.twilio.com, credential auth
     │
baresip SIP transport — dials on start, registers nowhere
     │
     │  [D] decoded audio frames, in-process (both ways)
     ▼
pipecat pipeline — STT → LLM → TTS
```

### The legs

| Leg | Protocol | What it is |
|---|---|---|
| A | PSTN (cellular/landline) | a real phone call reaching (or leaving) the trunk's Twilio number |
| B | SIP over TLS + RTP | origination: the trunk forwards an inbound call to the bot client's SIP URI at the registrar |
| C | SIP over TLS + RTP | registrar ↔ bot: REGISTER keeps the bot reachable at its AOR; INVITEs and RTP ride this leg |
| D | none — in-process Python | transport ↔ pipeline: decoded audio frames into STT, TTS audio back out |
| E | HTTPS (Twilio REST API) | auto-caller: a call created from the API with inline TwiML `<Say>`, no phone and no webhook |
| F | SIP over UDP + RTP | softphone ↔ registrar: a second Daily SIP client registered from your desktop |
| G | SIP over UDP + RTP | termination: the bot INVITEs the trunk's termination domain directly, credential-list auth, no registration |

Which legs each path exercises — this is the point of running more than one
path:

| Path | Legs |
|---|---|
| Softphone dials the bot | F+C+D |
| Bot dials the softphone | D+C+F |
| Cell phone dials the trunk number | A+B+C+D |
| Bot dials a cell phone | D+G+A |
| Auto-caller `call dial-in --via sip` (no human) | E+C+D |
| Auto-caller `call dial-in --via pstn` (no human) | E+B+C+D |

A path that works tells you its legs are sound; a path that fails next to a
path that works localizes the fault to the legs they differ on. One-way audio
on A+B+C+D while F+C+D is clean, for instance, points at B — the trunk's
media path.

## Call milestones

Every milestone is one grep-able INFO line, `[N/6]`, emitted by
`milestones.py` except `[4/6]`, which each bot logs where it queues the
greeting. What each one means about the SIP machinery underneath:

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

Lines that are not milestones: `heard: "..."` for every final transcription
that reached the LLM, `SIP audio warning: ...` for a stalling or degraded
stream, and `SIP error: ... — likely cause: ...`, which names the most likely
of bad credentials, an unreachable registrar, and NAT.

To see this list without any credentials at all:

```bash
uv run python examples/sip/util/milestones.py
```

## Troubleshooting

**The bot greets you, then greets again a few seconds later.**
Cause: the far end echoes the bot's audio back — the bot's own words appear
as `heard:` lines about a second after it speaks, and the VAD treats them as
the user. On a softphone this is speakers-into-mic echo; a SIP leg has no
echo cancellation.
Fix: wear headphones on the softphone machine. Echo that continues with the
softphone mic muted is in the media path — report it with the `heard:` lines.

**No ringing before the bot answers.**
Expected on dial-in: `auto_answer` accepts the INVITE immediately, so the
caller hears no ringback and `[2/6]` may never print.

**A captured log ends at `[4/6]` — no `[5/6]` or `[6/6]`.**
Cause: Ctrl-C killed the bot (and a plain `tee`) before the hangup was
processed.
Fix: hang up and wait — the bot logs `[5/6]`/`[6/6]` and exits on its own.
When teeing, use `tee -i`.

**Trunk dial-out fails instantly with `dial failed: Address family not
supported by protocol family`.**
Cause: `medianat=stun` cannot run on a registration-less account
(`SIP_REG_INTERVAL=0`). Registered accounts are unaffected.
Fix: use the command `call dial-out` prints — it sets `SIP_STUN_SERVER=off`
— and turn on the trunk's Symmetric RTP (next entry).

**The call connects but audio flows only one way.**
Cause: the far end sends RTP to an address the bot is not reachable at —
its LAN address when STUN is off (every trunk dial-out), or behind a
symmetric NAT.
Fix: turn on the trunk's Symmetric RTP at Twilio Console > Elastic SIP
Trunking > Trunks > `pipecat-harness` > General. The REST API cannot set it;
`status` reports it. Alternative: a TURN relay,
`SIP_EXTRA_PARAMS=medianat=turn,...`.

**Nothing registers, or the INVITE goes unanswered.**
Trace the signaling:

```bash
SIP_TRACE=1 SIP_NATIVE_LOG_LEVEL=debug uv run python examples/sip/dial-in/bot.py -t sip -d
```

The trace carries authentication material — never paste it into an issue or
commit it.

**The bot registers, then calls ring somewhere else.**
Cause: two processes registered the same account (a second copy of the bot
left running elsewhere); the registrar picked the other binding.
Fix: shut the other process down, or set a stable `SIP_INSTANCE_ID` (any
fixed UUID) so a restarted bot replaces its own binding.

**Trunk dial-out fails with a 403 after it worked yesterday.**
Cause: your public IP moved; the trunk's IP ACL is stale. `status` prints
the current address next to the authorized one.
Fix:

```bash
uv run python examples/sip/util/provision.py provision --pstn
```

**Telephone.app quirks.**
Registers over UDP, not TLS. Dial the bare `user@domain` — no `sip:` prefix,
no port. It can keep showing "registered" after the binding lapsed — remove
and re-add the account. Linphone and Zoiper show the raw SIP exchange and
debug better.

**`status` says a credential password is not cached.**
Cause: Twilio never returns a credential password after creation.
Fix:

```bash
uv run python examples/sip/util/provision.py provision --pstn
```

## Advanced: the webhook trunk path

Provisioning uses two *persistent* Daily SIP clients because that needs no
public URL. A Daily SIP trunk of type `sip_client` works the other way: it
provisions a SIP client *per inbound call* and POSTs its
credentials to the trunk's `webhook_url`, which keeps no long-lived
credential anywhere and scales to concurrent calls.

The dev runner stands in for the cloud handler at
`POST /daily-sip-trunk-webhook`, so you can run that path locally:

1. Create a Daily SIP trunk of type `sip_client` via the Daily REST API, with
   its `webhook_url` pointing at a public tunnel to your machine (ngrok, say)
   and the path `/daily-sip-trunk-webhook`. `DailyRESTHelper` has no trunk
   methods — this is a plain REST call, and it is why `provision.py` does not
   use this path.
2. Run the dial-in bot with the runner's server mode: `uv run python
   examples/sip/dial-in/bot.py -t sip` (no `-d`).
3. Point a phone number at the trunk and call it.

Daily acknowledges trunk creation with a `sip_trunk.test` probe, which the
route answers without starting anything. For a real `sip_trunk.incoming`
notification it starts the bot on the per-call account and answers at once,
while the registrar holds the caller until the bot's REGISTER lands. The whole
notification reaches the bot as `runner_args.body`, and
`create_transport` picks the account out of `body["sip_client"]` with no code
change in the bot.

## SIP env reference

SIP transport env vars these examples touch (the full surface is documented
in `pipecat.runner.sip`):

| Variable | Default | Meaning |
|---|---|---|
| `SIP_USER` / `SIP_PASS` / `SIP_DOMAIN` | `.state.json`'s `pipecat-harness-bot`, else unset | the account to register or authenticate as; the bots export the provisioned account when these are unset, and the runner provisions a temporary Daily client when there is no state either |
| `SIP_AUTH_USER` | the user part of the AOR | digest username when it differs — the credential-list trunk path |
| `SIP_REG_INTERVAL` | 600 | seconds between registration refreshes; `0` is trunk mode, no registration |
| `SIP_TRANSPORT` | `tls` for the provisioned account or a runner-provisioned client, `udp` for any other env account | SIP signaling transport |
| `SIP_DIALOUT_URI` | unset | dial this URI on start; the alternative to a `dialout_settings` runner body |
| `SIP_INSTANCE_ID` | unset | stable RFC 5626 `+sip.instance`, so a restart replaces its own binding |
| `SIP_STUN_SERVER` | `stun:stun.l.google.com:19302` | STUN server for `medianat=stun`; `off` disables media-NAT traversal |
| `SIP_EXTRA_PARAMS` | unset | raw baresip account parameters, comma-separated `key=value`; wins over `SIP_STUN_SERVER` |
| `SIP_NATIVE_LOG_LEVEL` | `warning` | lowest severity captured from the native stack |
| `SIP_TRACE` | unset | `1` logs every SIP message verbatim; contains auth material |

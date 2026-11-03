# Pipecat SIP Examples

Voice bots on a phone call, over the baresip-backed `SIPTransport`. Each
example is a complete, runnable bot; a shared `provision.py` CLI creates
everything a SIP call needs over REST, and numbered call-progress milestones
at INFO make a working call visible rather than silent. Run them to hear the
transport work, and to learn the telephony topology it sits in.

No telephony background is assumed. The five terms these documents lean on:

| Term | Meaning |
|---|---|
| **SIP** | the protocol phone systems use to set up and end calls — who is calling whom, ringing, answer, hangup; the audio itself flows separately (as RTP) |
| **softphone** | a desktop app that acts as a phone: mic, speakers, dial pad — but the network instead of a phone line |
| **registrar** | the server a SIP client signs in to (a SIP `REGISTER`), so that calls to its name can find it |
| **PSTN** | the public switched telephone network — the thing that makes real phones ring |
| **SIP trunk** | a bridge between SIP and the PSTN: it holds phone numbers on one side and converts calls in both directions |

## Examples

| Example                  | What it shows                                                          |
| ------------------------ | ---------------------------------------------------------------------- |
| [`dial-in/`](dial-in/)   | The bot registers and answers calls — from a softphone, or a real phone. |
| [`dial-out/`](dial-out/) | The bot places a call — to a softphone, or a real phone.               |

Both run over either of two call types, each self-contained:

| Call type | Needs | Costs | Call clients | Proves |
|---|---|---|---|---|
| **💻 Softphone** on the same Daily domain | `DAILY_API_KEY` + Deepgram/OpenAI/Cartesia keys | nothing | a softphone registered as a Daily SIP client — baresip via `provision.py softphone`, or Telephone.app/Linphone/Zoiper | REGISTER, INVITE, two-way RTP, DTMF, dial-in and dial-out — pure SIP, no PSTN |
| **📞 PSTN** via a Twilio SIP trunk | the softphone row plus `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` | a number is ~$1.15/month, opt-in | your cell phone over the PSTN, plus a Twilio REST auto-caller with no human on the line | the same, across a real Elastic SIP trunk: origination, termination auth, IP ACL |

Start with the softphone call type: it costs nothing and needs nothing
beyond the `.env` keys. Add PSTN when you want real phones.

Deploying these bots to Pipecat Cloud is out of scope here; this document
covers running them locally.

## Setup

From the repo root:

```bash
uv sync --all-extras
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

The SIP transport's own env vars (`SIP_USER`, `SIP_TRANSPORT`, …) are listed
under [SIP env reference](#sip-env-reference); the full surface is documented
in `pipecat.runner.sip`.

## Provisioning

`provision.py` owns the auxiliary infrastructure. Every resource it creates
carries the `pipecat-harness` name prefix, so `status` can find them live and
`teardown` never touches anything else. See the planned REST calls without
sending them with `--dry-run` on any command.

### Softphone calls: two Daily SIP clients

```bash
uv run python examples/sip/util/provision.py provision
```

This creates two persistent Daily SIP clients — `pipecat-harness-bot`, which
the bots register as, and `pipecat-harness-dev`, which your softphone
registers as — and prints the softphone's account block plus the bot's
dialable URI. Re-run it any time: an existing client is left alone, and a
client whose password is no longer in `.state.json` gets its password
rotated, so deleting the state file is recoverable.

```bash
uv run python examples/sip/util/provision.py status
```

prints the same values again as an OK/FIX checklist, fetched live.

```bash
uv run python examples/sip/util/provision.py softphone
```

skips manual softphone setup entirely: it writes a
[baresip](https://github.com/baresip/baresip) config registered as the dev
client and launches it (`brew install baresip` / `apt install baresip`
first). In baresip: `d <uri>` dials, `a` answers, `b` hangs up, digits send
DTMF, `q` quits.

Prefer a phone-shaped app instead? In Telephone.app on macOS
(**Settings → Accounts → +**), or Linphone/Zoiper on any OS, enter the four
values `provision` printed:

- **Username**: `pipecat-harness-dev`
- **Domain**: your Daily SIP domain, `<something>.sip.daily.co`
- **Password**: the printed password
- **Transport**: UDP

Daily accepts UDP for these accounts; Telephone.app's TLS support is not
something to rely on.

The softphone is deliberately a separate install rather than the
`baresip-python` package the transport itself runs on (whose wheels do ship
hardware audio drivers). Two copies of one stack can cancel out their own
bugs; keeping the far end an independently built client — a system
`baresip`, or a different SIP implementation entirely in Telephone.app,
Linphone, or Zoiper — means a fault on a call points at one side.

### PSTN calls: a Twilio Elastic SIP trunk

Adds a fully REST-provisioned trunk, so calls reach the bot from the PSTN and
the bot can call a real phone.

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

### A phone number (this is the part that costs money)

Attach one your account already owns:

```bash
uv run python examples/sip/util/provision.py provision --pstn --number +1XXXXXXXXXX
```

Or buy one. **This spends money** — roughly $1.15 per month until the number
is released. The command prints the price and asks for a `y` before
purchasing:

```bash
uv run python examples/sip/util/provision.py provision --pstn --buy-number --area-code 415
```

Teardown keeps the number by default; only `--release-number` gives it back.

### Teardown

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

## The topology

```
              PSTN (TWILIO TRUNK)                       SOFTPHONE (DAILY ONLY)
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
| Cell phone dials the trunk number (PSTN) | A+B+C+D |
| Auto-caller `call dial-in --via sip` (PSTN) | E+C+D |
| Auto-caller `call dial-in --via pstn` (PSTN) | E+B+C+D |
| Softphone dials the bot (softphone) | F+C+D |
| Bot dials the softphone (softphone) | D+C+F |
| Bot dials a cell phone (PSTN) | D+C+B+A |

A path that works tells you its hops are sound; a path that fails next to a
path that works localizes the fault to the hops they differ on. One-way audio
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

**A captured log ends at `[4/6]` — no `[5/6]` or `[6/6]`.** The bot never got
to process the hangup: Ctrl-C arrived while the call was still up (or within
the moment it takes to process the BYE), and in a `bot | tee` pipeline the
same Ctrl-C kills `tee`, so the lines the bot logs while shutting down never
reach the file. The bot ends its run by itself once the call ends — hang up,
watch for `[5/6]`/`[6/6]`, and skip Ctrl-C entirely. When teeing, use
`tee -i`, which ignores the interrupt and keeps the shutdown lines even if
Ctrl-C comes early.

**Dial-out through the trunk fails instantly with `dial failed: Address
family not supported by protocol family`.** `medianat=stun` cannot run on a
registration-less account (`SIP_REG_INTERVAL=0`): the STUN server lookup runs
at dial time, before any DNS has fixed the account's address family, and the
media session aborts before an INVITE is sent. Registered accounts (softphone
dial-out) are unaffected. `call dial-out` therefore puts
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
SIP_TRACE=1 SIP_NATIVE_LOG_LEVEL=debug uv run python examples/sip/dial-in/bot.py -t sip -d
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

**Trunk dial-out fails with a 403 after it worked yesterday.** Your public IP
moved and the trunk's IP ACL no longer lists it. `status` prints the address
it sees next to the authorized one; `provision --pstn` refreshes the entry.
The credential list is why dial-out usually survives this — it authenticates
per INVITE and does not care about addresses.

**Telephone.app quirks.** It registers over UDP, not TLS; dial the bare
`user@domain` URI, with no `sip:` prefix and no port; and it sometimes keeps
showing "registered" after the binding has lapsed — remove and re-add the
account to force a fresh REGISTER. Linphone and Zoiper are the fallbacks and
both show the raw SIP exchange, which makes them better for debugging.

**`status` says a credential password is not cached.** Twilio never returns a
credential password after creation. `provision --pstn` resets it to a fresh
one and caches that.

## Advanced: the webhook trunk path

Provisioning uses two *persistent* Daily SIP clients because that needs no
public URL. Production works the other way: a Daily SIP trunk of type
`sip_client` provisions a SIP client *per inbound call* and POSTs its
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

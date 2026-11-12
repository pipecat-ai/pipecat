# Pipecat SIP Examples

Voice bots on a phone call over the baresip-backed `SIPTransport`.

Never placed a SIP call before? Go straight to
[Tier 1](#tier-1--your-first-call-5-minutes): one provisioning command, a free
softphone, and a bot that answers. Unfamiliar words are tabulated under
[Terminology & topology](#terminology--topology).

## Which tier

Each tier stands alone and adds one layer of real telephony. Stop at whichever
one answers your question.

| Tier | Proves | Needs | Time |
|---|---|---|---|
| [0](#tier-0--no-credentials-no-calls-1-minute) | the code imports, the units pass, the log vocabulary | nothing | 1 min |
| [1](#tier-1--your-first-call-5-minutes) | REGISTER, INVITE, two-way RTP, DTMF — a real call to a real bot | Daily + Deepgram/OpenAI/Cartesia keys, a softphone | 5 min |
| [2](#tier-2--dial-out-and-video-10-minutes) | the bot places calls; video | tier 1 | 10 min |
| [3](#tier-3--pstn-via-a-twilio-trunk-optional) | a real phone number, over a carrier trunk | tier 1 + Twilio credentials | 20 min |
| [4](#tier-4--the-webhook-trunk-path-optional) | per-call credentials, signed notifications, concurrency | tier 1 + a public tunnel | 20 min |

Tiers 3 and 4 are optional. Tier 3 costs money and needs a Twilio account;
tier 4 needs a publicly reachable URL. Neither is needed to exercise the
transport.

## Tier 0 — no credentials, no calls (1 minute)

The unit tests and the log vocabulary, with no account anywhere.

```bash
uv run pytest tests/test_sip_connection.py tests/test_sip_transport.py \
    tests/test_sip_video.py tests/test_runner_sip.py \
    tests/test_daily_sip_clients.py -n 0
```

All mock-based — nothing opens a socket, and the run takes a few seconds.

> [!IMPORTANT]
> `test_sip_connection.py`, `test_sip_transport.py` and `test_sip_video.py`
> begin with `pytest.importorskip("baresip")`. Without `uv sync --all-extras`
> they **skip silently**: most of the suite skips and the run still reports
> success. Read the skip count, not the exit code.

Print every milestone line a call can emit, with no credentials and no call:

```bash
uv run python examples/sip/util/milestones.py
```

What each line means is tabulated under [Call milestones](#call-milestones).

## Tier 1 — your first call (5 minutes)

### Setup

From the repo root:

```bash
uv sync --group dev --extra sip --extra sip-video --extra daily --extra deepgram --extra cartesia --extra openai --extra silero --extra runner
```

That installs the transport's SIP stack. The `baresip-python` wheel bundles the
native stack, so no system `baresip` is needed to **run a bot** — only to use
baresip as the far-end softphone ([tier 1](#softphone-choice)).

Copy the env template and fill in your API keys:

```bash
cp examples/sip/env.example examples/sip/.env
```

| Variable | Required by |
|---|---|
| `DAILY_API_KEY` | all tiers: SIP-client CRUD, and the runner's temporary client when no account is set |
| `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`, `CARTESIA_API_KEY` | the bots' STT → LLM → TTS cascade |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN` | tier 3 only: Twilio REST basic auth |
| `TWILIO_CALLER_ID` | tier 3 only, and only when no number is attached to the trunk |

A softphone dials the bot; the bot answers and holds a conversation.

### 1. Provision a softphone account

```bash
uv run python examples/sip/util/provision.py provision
```

Creates two Daily SIP clients — `pipecat-harness-bot` and
`pipecat-harness-dev` — and prints the dev client's username, domain and
password plus the bot's dialable URI. No Twilio, no system packages, no cost.

Idempotent, so re-run it any time. Re-running **rotates the password** when
none is cached, which breaks a softphone already configured with the old one —
re-enter the printed password after a rotation.

```bash
uv run python examples/sip/util/provision.py status
```

Prints the same values again as a live OK/FIX checklist.

### 2. Register a softphone

Install [Linphone](https://www.linphone.org/) (macOS, Windows, Linux) and add
an account with the values `provision` printed:

- **Username**: `pipecat-harness-dev`
- **Domain**: your Daily SIP domain exactly as `provision` printed it — region-scoped, like `cloud-<id>.sip-us.daily.co`
- **Password**: the printed password
- **Transport**: UDP

Daily accepts UDP for these accounts.

#### Softphone choice

[Telephone.app](https://apps.apple.com/us/app/telephone/id406825478) (macOS)
and [Zoiper](https://www.zoiper.com/) work the same way. Linphone and Zoiper
show the raw SIP exchange, which debugs better.

For a terminal softphone, install baresip and let `provision.py` configure it:

```bash
brew install baresip     # apt install baresip on Linux
uv run python examples/sip/util/provision.py softphone
```

The terminal it runs in becomes the softphone: `d <uri>` dials, `a` answers,
`b` hangs up, digits send DTMF, `q` quits.

> [!NOTE]
> The softphone is deliberately a separate install rather than the
> [`baresip-python`](https://github.com/daily-co/baresip-python) package the
> transport itself runs on. Two copies of one stack can cancel out their own
> bugs; keeping the far end an independently built client means a fault on a
> call points at one side.

### 3. Run the bot

```bash
uv run python examples/transports/transports-sip-dialin.py -t sip -d
```

No SIP account needed on the command line: the runner provisions a temporary
Daily SIP client, prints its dialable URI, and deletes it again on exit.

### 4. Dial it

Dial the URI the bot printed. In Linphone and Telephone.app, dial the bare
`user@domain` — no `sip:` prefix, no port. In baresip, `d` then the full
`sip:user@domain`.

Expect no ringback — the bot auto-answers — then a greeting and a
conversation. Press a dial-pad digit to exercise DTMF. Hang up and the bot
logs its closing milestones and exits: **one call per run**.

**Wear headphones.** Speakers into mic makes the bot hear itself and greet
twice; a SIP leg has no echo cancellation.

A healthy run is captured in [`dial-in/README.md`](dial-in/README.md#what-you-should-see)
— zero packet loss, single-digit jitter. What each line means is under
[Call milestones](#call-milestones).

> [!NOTE]
> Daily accepts an INVITE to a registered client from anywhere, so a softphone
> that dials without any account configured can skip steps 1 and 2 and dial the
> printed URI directly. Whether a given softphone *can* dial accountless is up
> to that softphone — Telephone.app cannot, Linphone can be configured to.
> Steps 1 and 2 work on every client, and tier 2 needs the account anyway.

### The harness bot

[`dial-in/bot.py`](dial-in/) is the same bot wired to the provisioned
`pipecat-harness-bot` account and the numbered milestone logger:

```bash
uv run python examples/sip/dial-in/bot.py -t sip -d
```

Use it when you want a stable URI across runs, or the `[1/6]`–`[6/6]` lines.

## Tier 2 — dial-out and video (10 minutes)

### Bot → softphone

Keep the softphone of tier 1 registered. Generate the dial-out body and the
command that uses it:

```bash
uv run python examples/sip/util/provision.py call dial-out --to softphone
```

Run what it prints:

```bash
uv run python examples/sip/dial-out/bot.py -t sip -d \
    --runner-body examples/sip/dial-out/dialout-body.json
```

The bot dials as soon as it is ready. Answer in the softphone. Full
walkthrough: [`dial-out/README.md`](dial-out/).

A registered softphone account is **required** in this direction — the bot's
INVITE needs an address of record to find. Dial-in does not.

### Video

Echoes the caller's audio and video back, with no AI services and no API keys:

```bash
uv run python examples/transports/transports-sip-video.py -t sip -d
```

Call it from a video-capable client (Linphone) to see and hear yourself.
Needs the `sip-video` extra, which `uv sync --all-extras` already installed.

## Tier 3 — PSTN via a Twilio trunk (optional)

> [!IMPORTANT]
> Skip this tier unless you have Twilio credentials. It provisions billable
> resources and needs one manual console click. Tiers 1 and 2 already prove
> REGISTER, INVITE, two-way RTP, DTMF and both call directions.

Adds a fully REST-provisioned Elastic SIP trunk, so calls reach the bot from
the PSTN and the bot can call a real phone.

### 1. Provision the trunk

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

### 2. Attach a phone number

Attach a number your Twilio account already owns:

```bash
uv run python examples/sip/util/provision.py provision --pstn --number +1XXXXXXXXXX
```

`--buy-number` buys one instead, and **charges roughly $1.15 per month until
released**. `teardown --release-number` is what ends that charge; plain
teardown detaches the number and keeps it.

### 3. Turn on Symmetric RTP — manual

> [!WARNING]
> Twilio Console > Elastic SIP Trunking > Trunks > `pipecat-harness` >
> General > **Symmetric RTP**. The REST API cannot set it, so `provision`
> cannot either; `status` reports it as a warning. Trunk dial-out has no
> working media path until it is on, because the bot advertises its LAN
> address with STUN off.

### 4. Call, both directions

Run the bot, then call the trunk's number from your phone (`status` prints
the number):

```bash
uv run python examples/sip/dial-in/bot.py -t sip -d
```

The bot calls a phone with the generated command:

```bash
uv run python examples/sip/util/provision.py call dial-out --to +1XXXXXXXXXX
```

### The auto-caller — no human on the line

Twilio places the call from the REST API with inline TwiML and reads a
sentence to the bot. Terminal 1 runs the bot; terminal 2 places the call:

```bash
uv run python examples/sip/util/provision.py call dial-in --via sip --duration 30
```

It follows the call's status, hangs up at `--duration`, and exits 0 only if
the call connected and held at least 5 seconds. `--via pstn` rings the
trunk's number instead of the bot's SIP URI — best-effort, since Twilio
rejects a call whose `From` and `To` are the same number.

A `queued → failed` in under a second means nothing was registered at the bot
URI. Start the bot first; it serves one call per run, so restart it before
each auto-call.

## Tier 4 — the webhook trunk path (optional)

Tiers 1–3 use two *persistent* Daily SIP clients, because that needs no public
URL. A Daily SIP trunk of type `sip_client` works the other way: it provisions
a SIP client **per inbound call** and POSTs its credentials to the trunk's
`webhook_url`. No long-lived credential exists anywhere, and concurrent calls
each get their own account. This is the shape Pipecat Cloud runs.

Two endpoints implement the same route, and they differ in one way that
matters:

| Endpoint | Signature | Use for |
|---|---|---|
| the dev runner's built-in `POST /daily-sip-trunk-webhook`, active under `-t sip` without `-d` | **not verified** — logs `signed`/`unsigned` only | trying the path out |
| [`examples/transports/transports-sip-trunk-webhook.py`](../transports/transports-sip-trunk-webhook.py) | verified | reviewing what production does |

### Run it

Start an endpoint and expose it. With the standalone example:

```bash
SIP_TRUNK_HMAC=placeholder uv run python \
    examples/transports/transports-sip-trunk-webhook.py --port 7860
ngrok http 7860
```

Or with the dev runner's stand-in, which needs no secret:

```bash
uv run python examples/sip/dial-in/bot.py -t sip
```

Create the trunk against it. `DailyRESTHelper` has no trunk methods, so this
is a plain REST call — which is why `provision.py` does not own this path:

```bash
curl -s -X POST "$DAILY_API_URL/sip-trunk" \
  -H "Authorization: Bearer $DAILY_API_KEY" -H "Content-Type: application/json" \
  -d '{"trunk_name": "mytrunk", "trunk_type": "sip_client",
       "trunk_config": {"is_open": true, "exp_offset": 3600},
       "sip_client_template": {"transport": "tls"},
       "notification": {"webhook_url": "https://<host>/daily-sip-trunk-webhook"}}'
```

Then point a phone number at the trunk and call it.

### What happens, and why

- **The create fails unless your endpoint answers.** Daily acknowledges trunk
  creation with a `sip_trunk.test` probe and needs a 2xx.
- **The secret does not exist until the create returns it.** The response
  carries `config.notification.hmac` — the value of `SIP_TRUNK_HMAC` — and
  `sip_uri`, the host your carrier's trunk originates calls to. So start the
  standalone server with any value, create the trunk, then restart it with the
  real secret. The probe is answered without verification either way. Restart
  the same way after a rotation.
- **The signature covers the bytes Daily sent.** `X-Siptrunk-Signature` is the
  base64 HMAC-SHA256, keyed with the base64-*decoded* secret, of
  `"<X-Siptrunk-Timestamp>.<request body>"`. A re-serialized body only matches
  while both ends happen to serialize alike.
- **A 2xx is a commitment.** The endpoint answers before the bot is up; Daily
  then holds the caller until the bot's REGISTER lands at the registrar.
  Anything else is retried inside Daily's budget — about 12 seconds from the
  INVITE — and the call is then rejected with `480`.
- **Duplicates are served, not dropped.** Daily marks a carrier failover
  between its SIP nodes with `call.duplicate`: a second branch with the same
  Call-ID and its own account. Both must be answered and served, since one of
  the two carries the caller and it is often the flagged one. A handler that
  keeps state keys it on `(call.sipCallIdHeader, sip_client.username)`.

The whole notification reaches the bot as `runner_args.body`, and
`create_transport` picks the account out of `body["sip_client"]` with no code
change in the bot.

## Provisioning reference

`provision.py` owns the auxiliary infrastructure. Every resource it creates
carries the `pipecat-harness` name prefix, so `status` can find them live and
`teardown` never touches anything else. Every command that changes something
takes `--dry-run`, which prints the REST calls it would send and touches
nothing; `status` only ever reads.

Provisioned resources are cached in `.state.json` beside `.env`, mode 0600 —
it holds SIP passwords, which Daily and Twilio return only once. Both files
are gitignored. Deleting the cache is recoverable: `status` re-reads
everything live, and `provision` rotates a password it cannot recover.

### Teardown

Delete what provisioning created, in reverse order, treating anything already
gone as done:

```bash
uv run python examples/sip/util/provision.py teardown --scope all
```

`--scope daily` deletes only the two Daily SIP clients; `--scope pstn` only
the Twilio resources (and warns if the Daily clients are left behind, since
origination points at the bot client). The phone number is detached and kept
unless you pass `--release-number`, which is what ends the monthly charge.

## Terminology & topology

| Term | Meaning |
|---|---|
| **SIP** | the protocol phone systems use to set up and end calls — who is calling whom, ringing, answer, hangup; the audio itself flows separately (as RTP) |
| **softphone** | a desktop app that acts as a phone: mic, speakers, dial pad — but the network instead of a phone line |
| **registrar** | the server a SIP client signs in to (a SIP `REGISTER`), so that calls to its name can find it |
| **AOR** | address of record — the name a registrar maps to wherever the client currently is |
| **PSTN** | the public switched telephone network — the thing that makes real phones ring |
| **SIP trunk** | a bridge between SIP and the PSTN: it holds phone numbers on one side and converts calls in both directions |

Four manual call paths follow, one chart each. Every chart reads top to
bottom in the same order — far end first, bot's pipeline last — so the four
are easy to compare. Arrows show who places the call; once answered, audio
flows both ways on every leg. Each leg carries a letter, `[A]`–`[G]`, used
by the fault-localization table at the end.

### Softphone dial-in — you dial the bot

```
softphone — Linphone or Telephone.app, registered as pipecat-harness-dev
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
softphone — rings; answer in the app
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
greeting. The harness bots in [`dial-in/`](dial-in/) and
[`dial-out/`](dial-out/) attach it; the `examples/transports/` bots log the
same events in their own words. What each one means about the SIP machinery
underneath:

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

**The softphone was working, now it will not register.**
Cause: `provision` rotated the account password, which it does whenever none
is cached.
Fix: re-run `status` and re-enter the printed password in the softphone.

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

**The auto-caller reports `queued → failed` in under a second.**
Cause: nothing was registered at the bot URI — the bot was not running, or it
had already served its one call for the run.
Fix: start the bot, wait for `[1/6]`, then place the call.

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
and re-add the account. It has no dial field without a configured account.
Linphone and Zoiper show the raw SIP exchange and debug better.

**`status` says a credential password is not cached.**
Cause: Twilio never returns a credential password after creation.
Fix:

```bash
uv run python examples/sip/util/provision.py provision --pstn
```

## SIP env reference

SIP transport env vars these examples touch (the full surface is documented
in `pipecat.runner.sip`):

| Variable | Default | Meaning |
|---|---|---|
| `SIP_USER` / `SIP_PASS` / `SIP_DOMAIN` | `.state.json`'s `pipecat-harness-bot`, else unset | the account to register or authenticate as; the harness bots export the provisioned account when these are unset, and the runner provisions a temporary Daily client when there is no state either |
| `SIP_AUTH_USER` | the user part of the AOR | digest username when it differs — the credential-list trunk path |
| `SIP_REG_INTERVAL` | 600 | seconds between registration refreshes; `0` is trunk mode, no registration |
| `SIP_TRANSPORT` | `tls` for a provisioned client or an account from a request body, `udp` for an account from the environment | SIP signaling transport: `udp`, `tcp` or `tls` |
| `SIP_DIALOUT_URI` | unset | dial this URI on start; the alternative to a `dialout_settings` runner body |
| `SIP_INSTANCE_ID` | unset | stable RFC 5626 `+sip.instance`, so a restart replaces its own binding |
| `SIP_STUN_SERVER` | `stun:stun.l.google.com:19302` | STUN server for `medianat=stun`; `off` disables media-NAT traversal |
| `SIP_EXTRA_PARAMS` | unset | raw baresip account parameters, comma-separated `key=value`; wins over `SIP_STUN_SERVER` |
| `SIP_NATIVE_LOG_LEVEL` | `warning` | lowest severity captured from the native stack |
| `SIP_TRACE` | unset | `1` logs every SIP message verbatim; contains auth material |

# SIP Dial-in

The bot registers as a SIP client and answers whatever calls in — from a
softphone on your Daily domain, or from a real phone over the PSTN.

> [!IMPORTANT]
> Setup instructions — dependencies, API keys, provisioning, softphone
> options, terminology — are in [`examples/sip/README.md`](../README.md).

## Softphone → bot (same Daily domain)

One-time setup (details in [`../README.md`](../README.md)):

```bash
brew install baresip                                  # only for the baresip softphone; apt on Linux
cp examples/sip/env.example examples/sip/.env         # then fill in the keys
uv run python examples/sip/util/provision.py provision
```

Terminal 1 — launch the softphone. This terminal becomes baresip; the `d`
command further below is typed here:

```bash
uv run python examples/sip/util/provision.py softphone
```

Terminal 2 — run the bot:

```bash
uv run python examples/sip/dial-in/bot.py -t sip -d
```

Wait for the bot's `[1/6] REGISTER ok`. Then, in terminal 1 at the baresip
prompt, type `d` (baresip's dial command) followed by the bot's URI —
`provision` and `status` print the exact URI:

```
d sip:pipecat-harness-bot@<domain>.sip-us.daily.co
```

In a GUI softphone, dial the bare `pipecat-harness-bot@<domain>.sip-us.daily.co`
— no `sip:` prefix.

Expect no ringback — the bot auto-answers — then a greeting and a
conversation. Press a dial-pad digit to exercise DTMF; the bot logs
`DTMF <tone>`. Hang up and the bot logs `[5/6]` and `[6/6]`, then exits on
its own — one call per run.

> [!NOTE]
> When capturing a run to a file, pipe through `tee -i`: plain `tee` dies
> with the first Ctrl-C and drops everything the bot logs while shutting
> down.

## PSTN → bot (Twilio SIP trunk)

Provision the trunk and route a number to it (idempotent — already done is
fine):

```bash
uv run python examples/sip/util/provision.py provision --pstn --number +1XXXXXXXXXX
```

### Cell-phone dial-in

```bash
uv run python examples/sip/dial-in/bot.py -t sip -d
```

Call the trunk's number from your phone. It is printed by:

```bash
uv run python examples/sip/util/provision.py status
```

### The auto-caller

Fully automated — you neither speak nor hear anything on this call. Twilio
places it and plays a scripted sentence to the bot; you watch the two
terminals instead.

Terminal 1 — run the bot:

```bash
uv run python examples/sip/dial-in/bot.py -t sip -d
```

Terminal 2 — place the call:

```bash
uv run python examples/sip/util/provision.py call dial-in --via sip --duration 30
```

The CLI follows the call's status until it hangs up at `--duration`, and
exits 0 only if the call connected and held at least 5 seconds. The bot's
log should show the sentence transcribed. `--via pstn` rings the trunk's
number instead of the bot's SIP URI (best-effort: Twilio rejects it when
`From` and `To` are the same number).

## What you should see

A real softphone capture (softphone → bot, two spoken turns, two dial-pad
digits, caller hangs up):

```
00:00:02.793 [1/6] REGISTER ok — reachable at sip:pipecat-harness-bot@cloud-372e5a69aac349f08a312211e75f28c2.sip-us.daily.co
00:00:09.243 [3/6] answered, media flowing (first RTP received)
00:00:09.243 [4/6] pipeline started — bot is listening
00:00:13.776 heard: "Hi."
00:00:15.754 heard: "I'm just testing something."
00:00:18.039 DTMF 3
00:00:19.452 DTMF 4
00:00:30.363 [6/6] quality: CallStats(duration_s=21, setup_s=0, tx_packets=1056, tx_bytes=168960, tx_errors=0, tx_avg_bitrate_bps=63882, rx_packets=1053, rx_bytes=164736, rx_errors=0, rx_avg_bitrate_bps=62400, tx_lost=0, rx_lost=0, tx_jitter_ms=3.25, rx_jitter_ms=2.25, rtt_ms=13.047, jbuf_late=928, jbuf_lost=24, jbuf_overflow=0, jbuf_out_of_sequence=0, jbuf_delay_ms=40, jbuf_jitter_ms=0, jbuf_skew_ms=0, audio_tx_silence_frames=0, audio_tx_starved_frames=0, audio_rx_dropped_bytes=0, audio_rx_discarded_bytes=0)
00:00:30.363 [5/6] call ended
```

`[2/6]` is absent because `auto_answer` accepts the INVITE before any
ringback. Zero loss (`tx_lost=0, rx_lost=0`) and single-digit jitter are
what a healthy same-domain call looks like. What each line means is
tabulated in [Call milestones](../README.md#call-milestones).

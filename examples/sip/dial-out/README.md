# SIP Dial-out

The bot places a call as soon as its transport is ready — to a softphone on
your Daily domain, or to a real phone over the PSTN.

> [!IMPORTANT]
> Setup instructions — dependencies, API keys, provisioning, softphone
> options, terminology — are in [`examples/sip/README.md`](../README.md).

The destination comes from either the `SIP_DIALOUT_URI` env var or a small
JSON file passed as `--runner-body`; `provision.py call dial-out` writes
that file and prints the exact command to run, so start there.

## Bot → softphone (same Daily domain)

With the softphone registered (`provision.py softphone`), generate the
dial-out body and command:

```bash
uv run python examples/sip/util/provision.py call dial-out --to softphone
```

Run the command it prints:

```bash
uv run python examples/sip/dial-out/bot.py -t sip -d \
    --runner-body examples/sip/dial-out/dialout-body.json
```

The bot dials as soon as it is ready; answer in the softphone — `a` in
baresip, the green button in a GUI app.

## Bot → cell phone (PSTN via Twilio trunk)

Provision the trunk (idempotent — already done is fine):

```bash
uv run python examples/sip/util/provision.py provision --pstn
```

> [!WARNING]
> Turn on the trunk's Symmetric RTP first (Twilio Console > Elastic SIP
> Trunking > Trunks > `pipecat-harness` > General) or the bot will not hear
> the callee — see [troubleshooting](../README.md#troubleshooting).

Generate the dial-out command for the number to call, and run what it
prints:

```bash
uv run python examples/sip/util/provision.py call dial-out --to +1XXXXXXXXXX
```

The printed command authenticates straight to the trunk per INVITE (no
registration, `SIP_STUN_SERVER=off`) and presents the trunk's number as
caller ID — the generated env vars and body carry all of it.

## What you should see

The same numbered milestones as [dial-in](../dial-in/README.md), with
`[2/6]` naming the dialed target and `[3/6]` coming from the answer instead
of the INVITE — the far end decides when to pick up, so dial-out is the
direction where you hear ringback.

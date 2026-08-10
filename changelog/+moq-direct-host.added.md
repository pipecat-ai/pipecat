`MOQDirectHost` in `pipecat.runner.moq`: the reusable MoQ direct-mode host —
watch a relay for browsers announcing under the request prefix, run one bot
per browser-minted session id, and enforce the lifecycle guards a deployed
host needs (per-session peer wait and speech-idle limits via the runner
args, plus a host no-calls idle exit so a capped agent instance is always
released). The development runner's `--moq-direct` path now runs on it, and
`MOQDirectHost.from_env()` builds a host from `MOQ_*` environment variables
for platforms that start bots without CLI arguments, making a cloud
direct-mode bot a one-line entry point. Direct-mode sessions started by the
dev runner now also end after five minutes without speech instead of
running indefinitely.

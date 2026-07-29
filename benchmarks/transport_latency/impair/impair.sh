#!/usr/bin/env bash
# Manual network impairment for the local-relay benchmark tiers (macOS).
#
# Applies dummynet pipes (dnctl) via a pf anchor to the local relay ports:
#   moq-relay :4443 (QUIC/UDP + cert endpoint TCP), coturn :3478 (UDP).
# Both transports' client<->relay legs cross these ports, so one profile
# impairs both stacks identically.
#
# The harness NEVER runs this; apply a profile yourself, run trials with
# --impairment <profile> so the tag lands in the results, then reset.
#
# Usage:
#   sudo ./impair.sh <clean|rtt50|loss1|loss5>
#   sudo ./impair.sh burst <profile> <seconds>   # apply, sleep, reset (recovery runs)
#   sudo ./impair.sh status
#
# Profiles:
#   clean   remove all impairment (restore /etc/pf.conf)
#   rtt50   +25 ms each direction (+50 ms RTT), no loss
#   loss1   1% packet loss each direction
#   loss5   5% packet loss each direction

set -euo pipefail

ANCHOR=bench-impair
PORTS_UDP=(4443 3478)

pipe_config() {
  case "$1" in
    rtt50) echo "delay 25" ;;
    loss1) echo "plr 0.01" ;;
    loss5) echo "plr 0.05" ;;
    *) echo "unknown profile: $1" >&2; exit 1 ;;
  esac
}

apply() {
  local profile=$1
  dnctl pipe 1 config $(pipe_config "$profile")
  {
    for port in "${PORTS_UDP[@]}"; do
      echo "dummynet in quick proto udp from any port $port to any pipe 1"
      echo "dummynet out quick proto udp from any to any port $port pipe 1"
    done
  } | pfctl -q -a "$ANCHOR" -f -
  # The anchor must be referenced from the active ruleset to take effect.
  (cat /etc/pf.conf; echo "dummynet-anchor \"$ANCHOR\""; echo "anchor \"$ANCHOR\"") | pfctl -q -f -
  pfctl -q -E
  echo "impairment '$profile' active on udp ports: ${PORTS_UDP[*]}"
}

reset() {
  pfctl -q -a "$ANCHOR" -F all 2>/dev/null || true
  pfctl -q -f /etc/pf.conf
  dnctl -q flush
  echo "impairment cleared"
}

case "${1:-}" in
  clean) reset ;;
  rtt50|loss1|loss5) apply "$1" ;;
  burst)
    profile=${2:?usage: impair.sh burst <profile> <seconds>}
    secs=${3:?usage: impair.sh burst <profile> <seconds>}
    apply "$profile"
    echo "burst: holding '$profile' for ${secs}s…"
    sleep "$secs"
    reset
    ;;
  status)
    dnctl list 2>/dev/null || echo "no pipes"
    pfctl -a "$ANCHOR" -s rules 2>/dev/null || echo "no anchor rules"
    ;;
  *) sed -n '2,25p' "$0"; exit 1 ;;
esac

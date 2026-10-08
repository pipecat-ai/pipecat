#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The SIP account a bot registers with, for the development runner and for bots.

Three ways to use it, from least to most involved:

- **Normal.** Declare a ``"sip"`` factory in ``transport_params`` and call
  :func:`~pipecat.runner.utils.create_transport`::

      transport_params = {
          "sip": lambda: SIPParams(audio_in_enabled=True, audio_out_enabled=True),
      }

      async def bot(runner_args: RunnerArguments):
          transport = await create_transport(runner_args, transport_params)

  Under the development runner the arguments already carry the account. On
  Pipecat Cloud the bot receives plain ``RunnerArguments``, and
  ``create_transport`` resolves the account itself: from
  ``runner_args.body["sip_client"]`` (what a Daily SIP trunk sends for each
  call) or from ``SIP_USER``/``SIP_PASS``/``SIP_DOMAIN``. It never provisions
  a client.
- **Configured.** The same code reads deployment settings from the
  environment, listed below: media-NAT traversal, the registration interval,
  the fallback transport, logging. A bot that needs the account itself, to
  log its URI or to delete a client it provisioned, calls :func:`configure`,
  :func:`sip_runner_arguments` and :func:`cleanup` (example below).
- **Advanced.** Build :class:`~pipecat.runner.types.SIPRunnerArguments` or a
  :class:`~pipecat.transports.sip.connection.SIPConnection` by hand: codecs,
  the jitter buffer, an instance id, a network interface, DTMF mode.

Functions:

- configure(): Resolve the SIP account, in order: a ``sip_client`` block in
  the request body, the environment, or a temporary Daily SIP client it
  creates. Returns a SIPClientConfig.
- sip_runner_arguments(): Turn a resolved account plus the environment's SIP
  settings into the ``SIPRunnerArguments`` that ``create_transport`` takes.
- cleanup(): Delete a SIP client that configure() created.
- resolve_media_nat_params(): Resolve the baresip account ``extra_params``,
  enabling ``medianat=stun`` by default for NAT traversal.
- parse_jitter_buffer(): Parse ``SIP_JITTER_BUFFER`` into the receive
  jitter-buffer arguments.

Environment variables:

- SIP_USER / SIP_DOMAIN (optional) - Existing SIP account to use (along with
  SIP_PASS). A ``sip_client`` block in the request body takes precedence;
  without either, a temporary SIP client is created automatically.
- DAILY_API_KEY - Daily API key for SIP client creation (required when no
  account is provided).
- SIP_TRANSPORT (optional) - SIP transport protocol ("udp", "tcp", or "tls").
  Defaults to "udp" for an account from the environment and "tls" for a
  provisioned Daily SIP client or an account from the request body; a body
  that names its own ``transport`` wins over this variable.
- SIP_REG_INTERVAL (optional) - Seconds between registration refreshes,
  default 600; 0 disables registration (trunk mode).
- SIP_RTP_TIMEOUT (optional) - Seconds without received RTP after which a
  call is closed as dead; default 0 (off).
- SIP_AUDIO_CODECS (optional) - Comma-separated codec preference list by
  stack name, e.g. "opus/48000/2,PCMU/8000/1".
- SIP_AUTH_USER (optional) - Digest username when it differs from the user
  part of the address (credential-list trunks).
- SIP_INSTANCE_ID (optional) - A stable UUID for RFC 5626 ``+sip.instance``.
- SIP_EXTRA_PARAMS (optional) - Comma-separated raw baresip account
  parameters (``key=value``), passed verbatim; wins over SIP_STUN_SERVER.
- SIP_STUN_SERVER (optional) - STUN server for the default media-NAT traversal
  (medianat=stun, default stun.l.google.com); set to "off" to disable. Ignored
  when SIP_EXTRA_PARAMS is set. The address STUN reports does not reach the bot
  through a symmetric NAT; that deployment needs a peer that latches onto the
  bot's RTP source address, or medianat=turn via SIP_EXTRA_PARAMS.
- SIP_NET_INTERFACE (optional) - Restrict the stack to one local interface, by
  name or address: "127.0.0.1" for a registrar on loopback, or one address of
  a multi-homed host. Unset lets the OS pick the source address.
- SIP_JITTER_BUFFER (optional) - The receive jitter buffer: "off",
  "fixed:MIN-MAX" or "adaptive:MIN-MAX" in milliseconds (a bare "MIN-MAX"
  means fixed). Unset selects the transport's default, a fixed 40–60 ms;
  the stack's own setting is "fixed:100-200".
- SIP_NATIVE_LOG_LEVEL (optional) - Lowest severity captured from the native
  SIP stack: "debug", "info", "warning" (default) or "error".
- SIP_TRACE (optional) - "true" logs every SIP message verbatim, auth
  material included. Never in production.

Example, the configured layer::

    import aiohttp
    from pipecat.runner.sip import cleanup, configure, sip_runner_arguments
    from pipecat.runner.utils import create_transport

    async def bot(runner_args):
        async with aiohttp.ClientSession() as session:
            config = await configure(session, body=runner_args.body)
            try:
                sip_args = sip_runner_arguments(config, runner_args)
                transport = await create_transport(sip_args, transport_params)
                ...
            finally:
                await cleanup(session, config)
"""

import os
import uuid
from typing import Any

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.runner.types import RunnerArguments, SIPRunnerArguments
from pipecat.transports.daily.utils import DailyRESTHelper, DailySIPClientParams

#: STUN server baresip uses to discover and advertise its public media address
#: (``medianat=stun``) when the user has not supplied their own
#: ``SIP_EXTRA_PARAMS``. Override with ``SIP_STUN_SERVER``; disable with
#: ``SIP_STUN_SERVER=off``.
DEFAULT_STUN_SERVER = "stun:stun.l.google.com:19302"


class SIPClientConfig(BaseModel):
    """Configuration returned when configuring a SIP account.

    Parameters:
        user: SIP username.
        password: SIP password.
        domain: SIP domain to register with.
        transport: SIP transport protocol ("udp", "tcp", or "tls").
        sip_uri: The account's dialable SIP URI (None when the account came
            from the environment).
        provisioned: True when configure() created the account as an ephemeral
            Daily SIP client — pass the config to cleanup() when done.
    """

    user: str
    password: str
    domain: str
    transport: str
    sip_uri: str | None = None
    provisioned: bool = False


async def configure(
    aiohttp_session: aiohttp.ClientSession,
    *,
    body: Any = None,
    api_key: str | None = None,
    client_exp_duration: float = 2.0,
    prefix: str = "pipecat",
) -> SIPClientConfig:
    """Configure the SIP account for a bot to register with.

    This function will, in order:
    1. Use the account in ``body["sip_client"]`` when the request body carries
       one with a ``username`` and ``domain`` (what a Daily SIP trunk sends for
       each call: ``username``, ``password``, ``domain``, and optionally
       ``transport`` and ``sip_uri``).
    2. Use an existing account from the SIP_USER/SIP_DOMAIN environment variables
    3. Create an ephemeral SIP client on your Daily domain if no account is provided

    An ephemeral client gets a random ``<prefix>-<hex>`` username and expires
    on its own after ``client_exp_duration`` hours; delete it earlier with
    cleanup() when the bot shuts down. It registers over TLS by default, as
    does an account from the body, while an account from the environment
    defaults to UDP; ``SIP_TRANSPORT`` overrides all three, and a body that
    names its own ``transport`` wins over the variable.

    Args:
        aiohttp_session: HTTP session for making API requests.
        body: The session's request body (``runner_args.body``), read only
            for its ``sip_client`` block; anything else is ignored.
        api_key: Daily API key. Defaults to DAILY_API_KEY.
        client_exp_duration: Ephemeral client expiration time in hours.
        prefix: Username prefix for the ephemeral client.

    Returns:
        SIPClientConfig: The account plus, when provisioned through Daily or
        named in the body, its dialable sip_uri.

    Raises:
        Exception: If no account is provided and DAILY_API_KEY is not set,
            or SIP client creation fails.
    """
    client = body.get("sip_client") if isinstance(body, dict) else None
    if isinstance(client, dict) and client.get("username") and client.get("domain"):
        user, domain = client["username"], client["domain"]
        return SIPClientConfig(
            user=user,
            password=client.get("password") or "",
            domain=domain,
            transport=client.get("transport") or os.getenv("SIP_TRANSPORT") or "tls",
            sip_uri=client.get("sip_uri") or f"sip:{user}@{domain}",
        )

    user = os.getenv("SIP_USER")
    domain = os.getenv("SIP_DOMAIN")
    if user and domain:
        return SIPClientConfig(
            user=user,
            password=os.getenv("SIP_PASS", ""),
            domain=domain,
            transport=os.getenv("SIP_TRANSPORT", "udp"),
        )

    api_key = api_key or os.getenv("DAILY_API_KEY")
    if not api_key:
        raise Exception(
            "A SIP account is required: set SIP_USER/SIP_PASS/SIP_DOMAIN, or set "
            "DAILY_API_KEY to provision a temporary Daily SIP client automatically. "
            "Get your API key from https://dashboard.daily.co/developers"
        )

    daily_rest_helper = DailyRESTHelper(
        daily_api_key=api_key,
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )

    username = f"{prefix}-{uuid.uuid4().hex[:8]}"
    logger.info(f"Creating temporary Daily SIP client: {username}")
    client = await daily_rest_helper.create_sip_client(
        DailySIPClientParams(
            username=username,
            expires_in_seconds=int(client_exp_duration * 60 * 60),
        )
    )
    logger.info(f"Created Daily SIP client: {client.sip_uri}")

    return SIPClientConfig(
        user=client.username,
        password=client.password or "",
        domain=client.domain,
        transport=os.getenv("SIP_TRANSPORT", "tls"),
        sip_uri=client.sip_uri,
        provisioned=True,
    )


async def cleanup(
    aiohttp_session: aiohttp.ClientSession,
    config: SIPClientConfig,
    *,
    api_key: str | None = None,
) -> None:
    """Delete the SIP client that configure() created, if any.

    A no-op for accounts that came from the environment.

    Args:
        aiohttp_session: HTTP session for making API requests.
        config: The configuration configure() returned.
        api_key: Daily API key. Defaults to DAILY_API_KEY.
    """
    if not config.provisioned:
        return

    api_key = api_key or os.getenv("DAILY_API_KEY")
    if not api_key:
        return

    daily_rest_helper = DailyRESTHelper(
        daily_api_key=api_key,
        daily_api_url=os.getenv("DAILY_API_URL", "https://api.daily.co/v1"),
        aiohttp_session=aiohttp_session,
    )
    try:
        await daily_rest_helper.delete_sip_client(config.user)
        logger.info(f"Deleted Daily SIP client: {config.user}")
    except Exception as e:
        # The client expires on its own; a failed delete must not turn a
        # clean shutdown into an error.
        logger.warning(f"Failed to delete Daily SIP client {config.user}: {e}")


def parse_jitter_buffer(spec: str | None) -> tuple[str | None, tuple[int, int] | None]:
    """Parse ``SIP_JITTER_BUFFER`` into ``SIPConnection``'s jitter-buffer arguments.

    Accepted forms: ``off``; ``fixed:MIN-MAX`` or ``adaptive:MIN-MAX`` with
    the range in milliseconds; a bare ``MIN-MAX``, which means fixed. Unset
    or empty selects the transport's default (see
    :data:`pipecat.transports.sip.connection.DEFAULT_JITTER_BUFFER`).

    Args:
        spec: The raw ``SIP_JITTER_BUFFER`` value, or None.

    Returns:
        ``(mode, (min, max))``, either or both None when not specified.

    Raises:
        ValueError: The value is not one of the accepted forms.
    """
    if spec is None or not spec.strip():
        return None, None
    text = spec.strip().lower()
    if text == "off":
        return "off", None
    mode, sep, span = text.partition(":")
    if not sep:
        mode, span = "fixed", text
    if mode not in ("fixed", "adaptive"):
        raise ValueError(f"expected off, fixed:MIN-MAX or adaptive:MIN-MAX, got {spec!r}")
    lo, sep, hi = (part.strip() for part in span.partition("-"))
    if not (sep and lo.isdigit() and hi.isdigit()):
        raise ValueError(f"expected a MIN-MAX range in milliseconds, got {spec!r}")
    if not 0 < int(lo) <= int(hi):
        raise ValueError(f"MIN must be greater than 0 and no more than MAX, got {spec!r}")
    return mode, (int(lo), int(hi))


def resolve_media_nat_params(explicit: str | None) -> tuple[str, ...] | None:
    """Resolve the baresip account ``extra_params``, defaulting to STUN.

    ``SIP_EXTRA_PARAMS`` (``explicit``), if set, wins verbatim. Otherwise, unless
    ``SIP_STUN_SERVER`` is "off"/"none"/empty, ``medianat=stun`` is enabled with
    that STUN server (default stun.l.google.com) so baresip advertises the
    address the STUN server reports back.

    That address reaches the bot behind an endpoint-independent NAT. Behind an
    endpoint-dependent ("symmetric") NAT such as an AWS NAT Gateway it does not,
    and the call needs either a peer that latches onto the bot's RTP source
    address or ``medianat=turn``, which only ``SIP_EXTRA_PARAMS`` can express —
    see the NAT notes on
    :class:`~pipecat.transports.sip.connection.SIPConnection`.

    ``ice`` is deliberately not the default: against a non-ICE peer it gates
    media and breaks audio, whereas ``stun`` runs plain RTP. It is on by default
    rather than gated on NAT detection because a dev/deployed bot is behind NAT
    in the common case; on a public-IP host it is harmless beyond a dependency
    on the STUN server, which the logged message notes alongside the
    symmetric-NAT case.

    Args:
        explicit: The raw ``SIP_EXTRA_PARAMS`` value, or None.

    Returns:
        The account parameters as a tuple, or None for no extra params.
    """
    if explicit:
        return tuple(p.strip() for p in explicit.split(",") if p.strip()) or None

    stun = os.getenv("SIP_STUN_SERVER", DEFAULT_STUN_SERVER).strip()
    if stun.lower() in ("", "off", "none", "false", "0"):
        return None

    stun_uri = stun if stun.startswith(("stun:", "stuns:")) else f"stun:{stun}"
    logger.info(
        f"SIP media-NAT: enabling medianat=stun via {stun_uri} by default. If a call has "
        "one-way or no audio, either this STUN server is unreachable (baresip logs "
        "'medianat ... failed'), or a symmetric NAT (an AWS NAT Gateway) made the "
        "advertised port unreachable and the bot received no RTP: the peer has to latch "
        "onto the bot's RTP source address (on Twilio, the trunk's Symmetric RTP "
        "setting), or set SIP_EXTRA_PARAMS=medianat=turn,stunserver=turn:HOST:3478,"
        "stunuser=USER,stunpass=PASS. Disable with SIP_STUN_SERVER=off."
    )
    return ("medianat=stun", f"stunserver={stun_uri}")


def sip_runner_arguments(
    config: SIPClientConfig, runner_args: RunnerArguments | None = None
) -> SIPRunnerArguments:
    """Turn a resolved account into the arguments ``create_transport`` takes.

    Applies the environment's SIP settings (``SIP_AUDIO_CODECS``,
    ``SIP_AUTH_USER``, ``SIP_EXTRA_PARAMS``/``SIP_STUN_SERVER``,
    ``SIP_REG_INTERVAL``, ``SIP_RTP_TIMEOUT``, ``SIP_INSTANCE_ID``,
    ``SIP_NATIVE_LOG_LEVEL``, ``SIP_TRACE``, ``SIP_NET_INTERFACE``,
    ``SIP_JITTER_BUFFER``) on top of the account, and carries the session's
    ``body``, ``session_id``, signal flags and ``cli_args`` over from
    ``runner_args`` when given. The development runner, its routes and
    :func:`~pipecat.runner.utils.create_transport` all build their SIP
    arguments here, so a bot that calls it gets exactly what they would.

    Args:
        config: The account, from :func:`configure`.
        runner_args: The session's arguments to carry over, if any.

    Returns:
        The SIP runner arguments.

    Raises:
        ValueError: ``SIP_REG_INTERVAL`` or ``SIP_RTP_TIMEOUT`` is not an
            integer, or ``SIP_JITTER_BUFFER`` is malformed.
    """
    try:
        reg_interval = int(os.getenv("SIP_REG_INTERVAL", "600"))
        rtp_timeout = int(os.getenv("SIP_RTP_TIMEOUT", "0"))
    except ValueError:
        raise ValueError(
            "SIP_REG_INTERVAL and SIP_RTP_TIMEOUT must be integers (seconds)"
        ) from None
    try:
        jitter_buffer_mode, jitter_buffer_ms = parse_jitter_buffer(os.getenv("SIP_JITTER_BUFFER"))
    except ValueError as e:
        raise ValueError(f"SIP_JITTER_BUFFER: {e}") from None

    codecs = os.getenv("SIP_AUDIO_CODECS")
    args = SIPRunnerArguments(
        user=config.user,
        domain=config.domain,
        password=config.password,
        transport=config.transport,
        audio_codecs=tuple(c.strip() for c in codecs.split(",") if c.strip()) if codecs else None,
        auth_user=os.getenv("SIP_AUTH_USER"),
        extra_params=resolve_media_nat_params(os.getenv("SIP_EXTRA_PARAMS")),
        reg_interval=reg_interval,
        rtp_timeout=rtp_timeout,
        instance_id=os.getenv("SIP_INSTANCE_ID"),
        native_log_level=os.getenv("SIP_NATIVE_LOG_LEVEL", "warning"),
        sip_trace=os.getenv("SIP_TRACE", "").lower() in ("1", "true", "yes"),
        net_interface=os.getenv("SIP_NET_INTERFACE") or None,
        jitter_buffer_mode=jitter_buffer_mode,
        jitter_buffer_ms=jitter_buffer_ms,
        body=runner_args.body if runner_args is not None else {},
        session_id=runner_args.session_id if runner_args is not None else None,
    )
    if runner_args is not None:
        args.handle_sigint = runner_args.handle_sigint
        args.handle_sigterm = runner_args.handle_sigterm
        args.pipeline_idle_timeout_secs = runner_args.pipeline_idle_timeout_secs
        args.cli_args = runner_args.cli_args
    return args

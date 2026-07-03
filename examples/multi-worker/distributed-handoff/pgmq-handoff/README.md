# pgmq-handoff

Same shape as the [Redis handoff](../redis-handoff/), but the bus is backed by [PGMQ](https://github.com/tembo-io/pgmq) on a shared Postgres database (e.g. Supabase). Requires `uv add "pipecat-ai[pgmq]"`.

See the [top-level multi-worker README](../../README.md) for shared environment variables.

## Additional environment variables

| Variable       | Required by                                                          |
| -------------- | -------------------------------------------------------------------- |
| `DATABASE_URL` | PostgreSQL DSN (e.g. Supabase pooled connection string)              |
| `PGMQ_CHANNEL` | Optional, channel prefix for queue names. Defaults to `pipecat_acme` |

## Quick start

_Terminal 1_: start the greeter worker

```bash
uv run distributed-handoff/pgmq-handoff/llm.py greeter --database-url $DATABASE_URL
```

_Terminal 2_: start the support worker

```bash
uv run distributed-handoff/pgmq-handoff/llm.py support --database-url $DATABASE_URL
```

_Terminal 3_: start the main transport worker

```bash
uv run distributed-handoff/pgmq-handoff/main.py --database-url $DATABASE_URL
```

You can also set `DATABASE_URL` in `.env` and omit the `--database-url` flag.

## Architecture

Same as the [Redis handoff](../redis-handoff/); the `RedisBus` is replaced by a `PgmqBus`, and the "pub/sub channel" is a set of PGMQ queues on the shared Postgres instance.

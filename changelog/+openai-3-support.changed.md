- Widened the `openai` dependency to `>=1.74.0,<4` to support the openai 3 SDK,
  which builds on `httpx2` rather than `httpx`. On openai 3, TLS certificates
  verify against the operating system trust store instead of `certifi` — set
  `SSL_CERT_FILE` or `SSL_CERT_DIR` if you rely on a custom CA bundle — and a
  `Timeout` passed to a service's `http_client` must come from the HTTP client
  family that SDK uses.

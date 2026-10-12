- Added `output_config` to `AnthropicLLMService.Settings`, which carries Anthropic's
  `effort` level (`low` through `max`). Anthropic recommends setting it explicitly on
  Claude Sonnet 4.6 to avoid unexpected latency:
  `settings=AnthropicLLMService.Settings(output_config={"effort": "low"})`.

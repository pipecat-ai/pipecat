- Added `provider` to `OpenRouterLLMService.Settings`, which carries OpenRouter's
  provider routing preferences — which upstream providers may serve a request, in
  what order, and under what price, quantization and data-retention constraints:
  `settings=OpenRouterLLMService.Settings(provider={"only": ["azure"], "sort": "latency"})`.

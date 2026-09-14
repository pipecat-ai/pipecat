- `DeepSeekLLMService` defaults to `deepseek-flash`, DeepSeek's current name for
  V4.1 Flash. The previous default, `deepseek-v4-flash`, names a retired model
  and is only temporarily routed to V4.1 Flash. Set
  `settings=DeepSeekLLMService.Settings(model=...)` to use a different model.

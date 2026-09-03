- Widened the `anthropic` dependency to `>=0.49.0,<2` to support the anthropic 1 SDK. `AnthropicLLMService` sends `temperature`, `top_k` and `top_p` through the request's `extra_body`, since the Messages API methods dropped them as parameters in anthropic 1. Requests reach the API unchanged, and the settings keep their names and meaning.

    If you pass your own Bedrock client, anthropic 1 requires an explicit region — `AsyncAnthropicBedrock(aws_region=...)`, or `AWS_REGION` in the environment — where it previously fell back to `us-east-1`.

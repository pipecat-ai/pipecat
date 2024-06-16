import asyncio
import time

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.utils import random_uuid

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=4096
)

prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.<|eot_id|><|start_header_id|>system<|end_header_id|>\n\nPlease introduce yourself to the user.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


async def main():
    print("🥶 cold starting inference")
    start = time.monotonic_ns()

    engine_args = AsyncEngineArgs(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.90,
        enforce_eager=False,        # False means slower starts but faster inference
        disable_log_stats=True,     # disable logging so we can stream tokens
        disable_log_requests=True,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    duration_s = (time.monotonic_ns() - start) / 1e9
    print(f"🏎️ engine started in {duration_s:.0f}s")

    request_id = random_uuid()
    result_generator = engine.generate(
        prompt,
        sampling_params,
        request_id,
    )
    index, num_tokens = 0, 0
    start = time.monotonic_ns()
    async for output in result_generator:
        if (
            output.outputs[0].text
            and "\ufffd" == output.outputs[0].text[-1]
        ):
            continue
        text_delta = output.outputs[0].text[index:]
        index = len(output.outputs[0].text)
        num_tokens = len(output.outputs[0].token_ids)

        print(text_delta)
    duration_s = (time.monotonic_ns() - start) / 1e9

    print(
        f"\n\tGenerated {num_tokens} tokens in {duration_s:.1f}s,"
        f" throughput = {num_tokens / duration_s:.0f} tokens/second.\n"
    )

    return


async def xmain():
    llm = LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        enable_prefix_caching=True
    )

    outputs = llm.generate(prompt, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    outputs = llm.generate(prompt, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    asyncio.run(main())

"""最简单的 OpenRouter 测试 - 直接调用，不用 pipeline."""
import asyncio
import os

from openai import AsyncOpenAI

async def test_openrouter():
    """Run a minimal OpenRouter function-calling round-trip.

    This is an ad-hoc sanity script (kept outside the formal test suite)
    demonstrating direct OpenAI-compatible usage with a tool call.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("请设置 OPENROUTER_API_KEY")
        return
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取城市天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"}
                },
                "required": ["city"]
            }
        }
    }]
    
    print("🚀 测试 OpenRouter function calling...")
    
    try:
        response = await client.chat.completions.create(
            model="openai/gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "你是个助手，需要时调用工具"},
                {"role": "user", "content": "上海天气怎么样？用中文回答"}
            ],
            tools=tools,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        print(f"✅ OpenRouter 响应: {message.content}")
        
        if message.tool_calls:
            print(f"✅ 工具调用: {message.tool_calls[0].function.name}")
            print(f"   参数: {message.tool_calls[0].function.arguments}")
            
            # 模拟工具执行
            tool_result = "上海现在温暖潮湿，有轻微雾霾"
            
            # 第二轮：把工具结果发给模型
            messages = [
                {"role": "system", "content": "你是个助手"},
                {"role": "user", "content": "上海天气怎么样？用中文回答"},
                message,
                {
                    "role": "tool",
                    "tool_call_id": message.tool_calls[0].id,
                    "content": tool_result
                }
            ]
            
            final_response = await client.chat.completions.create(
                model="openai/gpt-4o-2024-11-20",
                messages=messages
            )
            
            print(f"✅ 最终回答: {final_response.choices[0].message.content}")
        else:
            print("❌ 没有调用工具")
            
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    asyncio.run(test_openrouter())
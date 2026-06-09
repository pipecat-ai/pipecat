`LLMSetToolsFrame` now accepts a plain list of direct functions and/or
`FunctionSchema` objects (not just a `ToolsSchema` or provider tool dicts), and
the LLM service auto-registers handlers for any direct functions it advertises.
Tools can now be changed mid-conversation using the same direct-function pattern
as `LLMContext(tools=[...])`, with no separate `register_function()` call.

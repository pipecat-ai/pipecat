Added `NO_RESPONSE` to Pipecat Flows. A consolidated function can return
`(result, NO_RESPONSE)` to finish the call without transitioning to a new node
or running the LLM, for when something else produces the next turn — for
example a function that hands control to another worker, which then responds.

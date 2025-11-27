from bedrock_agentcore import BedrockAgentCoreApp

app = BedrockAgentCoreApp()

@app.entrypoint
def my_agent(payload):
    return {"result": f"Hello {payload.get('name', 'World')}!"}

if __name__ == "__main__":
    app.run()
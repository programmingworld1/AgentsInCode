import asyncio
from agent_framework import tool, AgentSession
from agent_framework.azure import AzureOpenAIResponsesClient
from azure.identity import AzureCliCredential
from pydantic import Field
from typing import Annotated

PROJECT_ENDPOINT = "https://dqwdwfsfsdfewf.services.ai.azure.com/api/projects/proj-default"
DEPLOYMENT_NAME = "gpt-4.1"
AGENT_NAME = "my-agent-framework"
AGENT_INSTRUCTIONS = "You are a helpful assistant."


# Example tool — Agent Framework uses @tool + Annotated[type, Field(...)] for parameter descriptions
@tool
def add_numbers(
    a: Annotated[float, Field(description="First number")],
    b: Annotated[float, Field(description="Second number")],
) -> float:
    """Adds two numbers and returns the result."""
    return a + b


async def main():
    # project_endpoint tells the SDK this is a Foundry project URL,
    # so it handles the correct token scope (https://ai.azure.com) automatically
    credential = AzureCliCredential()
    client = AzureOpenAIResponsesClient(
        project_endpoint=PROJECT_ENDPOINT,
        deployment_name=DEPLOYMENT_NAME,
        credential=credential,
    )

    # Agent runs locally — no cloud agent service needed
    agent = client.as_agent(
        name=AGENT_NAME,
        instructions=AGENT_INSTRUCTIONS,
        tools=[add_numbers],
    )

    print(f"Agent '{AGENT_NAME}' started (Microsoft Agent Framework).\n")
    print("Type 'stop' to quit.\n")

    # AgentSession maintains conversation history across turns
    session = AgentSession()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "stop":
            print("Stopped.")
            break

        response = await agent.run(user_input, session=session)
        print(f"Agent: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())

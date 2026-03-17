import asyncio
from agent_framework import tool, AgentSession
from agent_framework.azure import AzureOpenAIResponsesClient
from agent_framework_orchestrations import GroupChatBuilder
from azure.identity import AzureCliCredential
from pydantic import Field
from typing import Annotated

PROJECT_ENDPOINT = "https://dqwdwfsfsdfewf.services.ai.azure.com/api/projects/proj-default"
DEPLOYMENT_NAME = "gpt-4.1"


@tool
def add_numbers(
    a: Annotated[float, Field(description="First number")],
    b: Annotated[float, Field(description="Second number")],
) -> float:
    """Adds two numbers and returns the result."""
    return a + b


async def main():
    client = AzureOpenAIResponsesClient(
        project_endpoint=PROJECT_ENDPOINT,
        deployment_name=DEPLOYMENT_NAME,
        credential=AzureCliCredential(),
    )

    math_agent = client.as_agent(
        name="math-agent",
        instructions="You are a math assistant. Solve calculations using your tools.",
        tools=[add_numbers],
    )
    general_agent = client.as_agent(
        name="general-agent",
        instructions="You are a helpful general assistant. Answer questions concisely.",
        tools=[],
    )
    creative_agent = client.as_agent(
        name="creative-agent",
        instructions="You are a creative writer. Give imaginative, vivid answers.",
        tools=[],
    )

    orchestrator = client.as_agent(
        name="orchestrator",
        instructions=(
            "You are a group chat orchestrator facilitating a collaborative discussion between specialized agents. "
            "Your role is to guide the conversation toward a well-rounded conclusion through iterative collaboration. "
            "Decide which agent should contribute next based on the flow of the discussion: "
            "'math-agent' for calculations and data, 'general-agent' for factual context, "
            "'creative-agent' for ideation and synthesis. "
            "Encourage agents to build on each other's contributions, challenge assumptions, and refine ideas. "
            "The goal is group discussion and iterative reasoning — not a single answer, but a shared understanding. "
            "Reply with only the name of the agent that should speak next."
        ),
        tools=[],
    )

    workflow = GroupChatBuilder(
        participants=[math_agent, general_agent, creative_agent],
        orchestrator_agent=orchestrator,
    ).build()

    print("Type 'stop' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "stop":
            break

        results = await workflow.run(user_input)

        print("\n--- Results ---")
        for event in results:
            if event.type == "output":
                for message in event.data:
                    if message.role == "assistant":
                        text = "".join(c.text for c in message.contents if c.text)
                        print(f"\n[{message.author_name}]\n{text}")
        print()


if __name__ == "__main__":
    asyncio.run(main())

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

PROJECT_ENDPOINT = "https://dqwdwfsfsdfewf.services.ai.azure.com/api/projects/proj-default"
WORKFLOW_NAME = "mynewworkflow"  # Name of the workflow created in the Foundry portal


def main():
    # Connect to the Foundry project — handles authentication and API access
    project = AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
    )

    openai_client = project.get_openai_client()

    # Create a conversation context to maintain state across the workflow execution
    conversation = openai_client.conversations.create()
    print(f"Conversation started: {conversation.id}\n")

    print("Type 'stop' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "stop":
            print("Stopped.")
            break

        # Execute the workflow by name — the workflow is defined in the Foundry portal
        stream = openai_client.responses.create(
            conversation=conversation.id,
            extra_body={"agent_reference": {"name": WORKFLOW_NAME, "type": "agent_reference"}},
            input=user_input,
            stream=True,
        )

        # Stream the response chunks as they arrive
        print("Agent: ", end="", flush=True)
        for event in stream:
            if event.type == "response.output_text.delta":
                print(event.delta, end="", flush=True)
            elif event.type == "response.failed":
                print(f"\n[Workflow error: {event.response.error.message}]")
        print("\n")


if __name__ == "__main__":
    main()

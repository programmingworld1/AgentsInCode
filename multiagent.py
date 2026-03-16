from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition

PROJECT_ENDPOINT = "https://dqwdwfsfsdfewf.services.ai.azure.com/api/projects/proj-default"
MODEL_NAME = "gpt-4.1"

ORCHESTRATOR_NAME = "orchestrator-agent"
PAYMENTS_AGENT_NAME = "payments-agent"

# In-memory list to store processed orders
orders = []

# Keywords used to detect payment-related requests and route to the payments agent
PAYMENT_KEYWORDS = ["pay", "payment", "order", "invoice", "charge", "transaction"]

# NOTE: ConnectedAgentTool does not exist in azure-ai-projects 2.0.1.
# Instead, routing between agents is done manually in Python using is_payment_request().
# This is functionally equivalent: the orchestrator detects intent and delegates to the correct agent.


def get_or_create_agent(project: AIProjectClient, agent_name: str, definition: PromptAgentDefinition):
    # Try to fetch the existing agent, create it if it doesn't exist yet
    try:
        agent = project.agents.get(agent_name=agent_name)
        print(f"Existing agent found: {agent.name}")
    except:
        agent = project.agents.create_version(
            agent_name=agent_name,
            definition=definition,
        )
        print(f"New agent created: {agent.name}")
    return agent


def is_payment_request(text: str) -> bool:
    # Check if any payment keyword is present in the user input
    return any(keyword in text.lower() for keyword in PAYMENT_KEYWORDS)


def main():
    project = AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
    )

    # Payments agent: handles payment requests and order processing
    payments_agent = get_or_create_agent(
        project,
        PAYMENTS_AGENT_NAME,
        PromptAgentDefinition(
            model=MODEL_NAME,
            instructions=(
                "You are a payments specialist. "
                "When you receive a payment request, extract the order details and confirm the payment was added. "
                "Always respond with: 'Payment processed: <details>'"
            ),
            tools=[],
        ),
    )

    # Orchestrator agent: handles all general questions not related to payments
    orchestrator_agent = get_or_create_agent(
        project,
        ORCHESTRATOR_NAME,
        PromptAgentDefinition(
            model=MODEL_NAME,
            instructions="You are a helpful assistant. Answer the user's questions.",
            tools=[],
        ),
    )

    print("\nMulti-agent app ready. Type 'stop' to quit.\n")

    openai_client = project.get_openai_client()  # client to send messages to the agents

    # Each agent gets its own conversation so their chat histories stay separate
    orchestrator_conversation = openai_client.conversations.create()
    payments_conversation = openai_client.conversations.create()

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "stop":
            print("Stopped.")
            break

        # Route to the payments agent if the input contains payment keywords, otherwise use the orchestrator
        if is_payment_request(user_input):
            print("[Routing to payments agent]")
            agent_ref = {"agent_reference": {"name": payments_agent.name, "type": "agent_reference"}}
            conversation_id = payments_conversation.id
        else:
            agent_ref = {"agent_reference": {"name": orchestrator_agent.name, "type": "agent_reference"}}
            conversation_id = orchestrator_conversation.id

        # Send the user message to the selected agent and get a response
        response = openai_client.responses.create(
            conversation=conversation_id,
            extra_body=agent_ref,
            input=user_input,
        )

        # If the payments agent confirmed a processed payment, save it to the orders array
        if "payment processed" in response.output_text.lower():
            orders.append({"input": user_input, "response": response.output_text})
            print(f"[Order saved] Total orders: {len(orders)}")

        print(f"Agent: {response.output_text}\n")


if __name__ == "__main__":
    main()

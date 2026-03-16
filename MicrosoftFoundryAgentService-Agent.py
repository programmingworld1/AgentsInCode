from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition

PROJECT_ENDPOINT = "https://dqwdwfsfsdfewf.services.ai.azure.com/api/projects/proj-default"
MODEL_NAME = "gpt-4.1"
AGENT_NAME = "my-agent-incodebuild"

def get_or_create_agent(project: AIProjectClient):
    try:
        agent = project.agents.get(agent_name=AGENT_NAME)
        print(f"Bestaande agent gevonden: {agent.name}\n")
    except:
        agent = project.agents.create_version(
            agent_name=AGENT_NAME,
            definition=PromptAgentDefinition(
                model=MODEL_NAME,
                instructions="Je bent een behulpzame assistent.",
                tools=[],
            ),
        )
        print(f"Nieuwe agent aangemaakt: {agent.name}\n")
    return agent


def main():
    project = AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
    )

    agent = get_or_create_agent(project)

    # The client is not local. When you call responses.create, it sends an HTTP request to your Azure endpoint. The
    # agent runs entirely in the cloud on Azure's servers.                                                        
    # Your local code only: Sends the message to Azure,and waits for the response.                                                                                 2. Waits for the response
    openai_client = project.get_openai_client()  # client to send messages to the agent

    # Without this there is no conversation ID to pass to responses.create on line openai_client.responses.create(                           
    # The conversation ID is what tells Azure which chat history to attach the message to. Without it, 
    # the agent has no memory of previous messages in the same session. 
    conversation = openai_client.conversations.create()  # create a new conversation session to maintain chat history
    print(f"Conversatie gestart: {conversation.id}\n")

    # agent_ref tells Azure which agent to use for the response. Without it, responses.create doesn't know which
    # agent should handle the message — you could have multiple agents in the same project.
    agent_ref = {"agent_reference": {"name": agent.name, "type": "agent_reference"}}

    print("Typ 'stop' om te stoppen.\n")
    while True:
        user_input = input("Jij: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "stop":
            print("Gestopt.")
            break

        response = openai_client.responses.create(  # send the user message to the agent and get a response
            conversation=conversation.id,
            extra_body=agent_ref,
            input=user_input,
        )
        print(f"Agent: {response.output_text}\n")

if __name__ == "__main__":
    main()

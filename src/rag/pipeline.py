"""End-to-end RAG pipeline using Azure AI Agent Service.

Creates a persistent agent in Azure AI Foundry with Azure AI Search grounding.
The agent remains available in the Foundry portal across runs.
"""

from __future__ import annotations

import typer
from rich.console import Console

from src.config import get_settings

console = Console()


def _get_project_client():
    """Create an AIProjectClient using managed identity."""
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    settings = get_settings()
    project_endpoint = settings.azure_ai_project.project_endpoint
    if not project_endpoint:
        console.print("[red]AZURE_AI_PROJECT_ENDPOINT not set in .env[/red]")
        raise typer.Exit(1)

    return AIProjectClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(),
    )


def _get_search_connection_id(project_client) -> str:
    """Find the Azure AI Search connection in the Foundry project."""
    from azure.ai.projects.models import ConnectionType

    try:
        conn = project_client.connections.get_default(
            connection_type=ConnectionType.AZURE_AI_SEARCH,
        )
        return conn.id
    except Exception:
        pass

    for conn in project_client.connections.list():
        if conn.type in (ConnectionType.AZURE_AI_SEARCH, "AzureAISearch"):
            return conn.id

    raise RuntimeError(
        "No Azure AI Search connection found in the Foundry project. "
        "Ensure the Bicep deployment created the 'ai-search-connection'."
    )


def _build_search_tool(connection_id: str, settings):
    """Construct the AzureAISearchTool for the agent."""
    from azure.ai.projects.models import (
        AISearchIndexResource,
        AzureAISearchQueryType,
        AzureAISearchTool,
        AzureAISearchToolResource,
    )

    query_type_str = settings.yaml_config.get("search", {}).get(
        "query_type", "vector_semantic_hybrid"
    )
    query_type_map = {
        "simple": AzureAISearchQueryType.SIMPLE,
        "semantic": AzureAISearchQueryType.SEMANTIC,
        "vector": AzureAISearchQueryType.VECTOR,
        "vector_simple_hybrid": AzureAISearchQueryType.VECTOR_SIMPLE_HYBRID,
        "vector_semantic_hybrid": AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID,
    }
    query_type = query_type_map.get(query_type_str, AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID)

    return AzureAISearchTool(
        azure_ai_search=AzureAISearchToolResource(
            indexes=[
                AISearchIndexResource(
                    project_connection_id=connection_id,
                    index_name=settings.azure_search.index,
                    query_type=query_type,
                    top_k=settings.search_top_k,
                )
            ]
        )
    )


def _get_or_create_agent(project_client, settings):
    """Find existing agent by name, or create a new version.

    The agent persists in Azure AI Foundry across runs — it is never deleted
    programmatically. This makes it visible and testable in the Foundry portal.

    Returns (agent_id, created) where created is True if a new version was made.
    """
    from azure.ai.projects.models import PromptAgentDefinition
    from azure.core.exceptions import ResourceNotFoundError

    agent_name = settings.agent_name

    try:
        agent_details = project_client.agents.get(agent_name)
        agent_id = agent_details.versions.latest.id
        console.print(
            f"  [green]Found existing agent '{agent_name}' "
            f"(id: {agent_id})[/green]"
        )
        return agent_id, False
    except (ResourceNotFoundError, Exception):
        pass

    console.print(f"  [yellow]Agent '{agent_name}' not found — creating...[/yellow]")

    search_conn_id = _get_search_connection_id(project_client)
    search_tool = _build_search_tool(search_conn_id, settings)

    instructions = settings.yaml_config.get("agent", {}).get(
        "instructions",
        (
            "You are a medical question-answering assistant. "
            "Search the knowledge base and provide evidence-based answers."
        ),
    )

    definition = PromptAgentDefinition(
        model=settings.agent_model,
        instructions=instructions,
        tools=[search_tool],
        temperature=settings.temperature,
    )

    agent_version = project_client.agents.create_version(
        agent_name=agent_name,
        definition=definition,
        description="PubMedQA RAG agent — RLAIF pipeline",
    )

    console.print(
        f"  [green]Created agent '{agent_name}' "
        f"(id: {agent_version.id})[/green]"
    )
    return agent_version.id, True


def _query_agent(openai_client, agent_id: str, question: str) -> str:
    """Send a question to the agent via the OpenAI Assistants API and return the response."""
    thread = openai_client.beta.threads.create()
    openai_client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=question,
    )
    run = openai_client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=agent_id,
    )
    if run.status == "failed":
        error_msg = getattr(run, "last_error", None)
        raise RuntimeError(f"Agent run failed: {error_msg or run.status}")

    messages = openai_client.beta.threads.messages.list(thread_id=thread.id)
    for msg in messages.data:
        if msg.role == "assistant":
            parts = []
            for block in msg.content:
                if hasattr(block, "text"):
                    parts.append(block.text.value)
            if parts:
                return "\n".join(parts)

    return "No response generated."


def demo() -> None:
    """Run 5 demo queries through the RAG pipeline."""
    settings = get_settings()
    project_client = _get_project_client()

    console.print("[bold]Initializing RAG agent...[/bold]")
    agent_id, created = _get_or_create_agent(project_client, settings)

    openai_client = project_client.get_openai_client()

    demo_questions = [
        "Does aspirin reduce the risk of cardiovascular events?",
        "Is metformin effective for type 2 diabetes management?",
        "Can regular exercise reduce the risk of Alzheimer's disease?",
        "Does vitamin D supplementation prevent fractures in elderly patients?",
        "Is there evidence that statins reduce mortality in heart disease patients?",
    ]

    for i, question in enumerate(demo_questions, 1):
        console.print(f"\n[bold cyan]Question {i}:[/bold cyan] {question}")
        try:
            response = _query_agent(openai_client, agent_id, question)
            console.print(f"[bold green]Response:[/bold green]\n{response}")
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
        console.print("─" * 80)

    console.print(
        f"\n[bold green]Demo complete — agent '{settings.agent_name}' "
        f"persists in Foundry.[/bold green]"
    )


def query() -> None:
    """Run a single interactive query through the RAG pipeline."""
    settings = get_settings()
    project_client = _get_project_client()

    console.print("[bold]Initializing RAG agent...[/bold]")
    agent_id, _created = _get_or_create_agent(project_client, settings)

    openai_client = project_client.get_openai_client()

    question = typer.prompt("Enter your medical question")
    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}")
    response = _query_agent(openai_client, agent_id, question)
    console.print(f"[bold green]Response:[/bold green]\n{response}")

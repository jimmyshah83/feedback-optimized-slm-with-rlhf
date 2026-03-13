"""End-to-end RAG pipeline using Azure AI Search + Azure OpenAI.

Architecture:
- Agent *definition* is stored in Azure AI Foundry (visible in the portal,
  versioned via azure-ai-projects SDK).
- Execution uses AzureOpenAI chat completions with manual retrieval from
  Azure AI Search (vector + semantic hybrid search).
- The Foundry Agent Service SDK (2.0.x) manages definitions but does not yet
  expose a runtime invoke API, so we use the proven search-then-generate
  pattern directly.
"""

from __future__ import annotations

import typer
from rich.console import Console

from src.config import get_settings

console = Console()


# ---------------------------------------------------------------------------
# Foundry agent definition (CRUD only — visible in the portal)
# ---------------------------------------------------------------------------

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


def _ensure_agent_definition(project_client, settings) -> str:
    """Ensure the agent definition exists in Foundry (get-or-create).

    Returns the agent version id (e.g. 'pubmedqa-rag:1').
    """
    from azure.ai.projects.models import (
        AISearchIndexResource,
        AzureAISearchQueryType,
        AzureAISearchTool,
        AzureAISearchToolResource,
        PromptAgentDefinition,
    )
    from azure.core.exceptions import ResourceNotFoundError

    agent_name = settings.agent_name

    try:
        agent_details = project_client.agents.get(agent_name)
        agent_id = agent_details.versions.latest.id
        console.print(
            f"  [green]Found agent definition '{agent_name}' "
            f"(id: {agent_id})[/green]"
        )
        return agent_id
    except (ResourceNotFoundError, Exception):
        pass

    console.print(
        f"  [yellow]Agent '{agent_name}' not found — creating definition...[/yellow]"
    )

    search_conn_id = _get_search_connection_id(project_client)

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
    query_type = query_type_map.get(
        query_type_str, AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID
    )

    search_tool = AzureAISearchTool(
        azure_ai_search=AzureAISearchToolResource(
            indexes=[
                AISearchIndexResource(
                    project_connection_id=search_conn_id,
                    index_name=settings.azure_search.index,
                    query_type=query_type,
                    top_k=settings.search_top_k,
                )
            ]
        )
    )

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
        f"  [green]Created agent definition '{agent_name}' "
        f"(id: {agent_version.id})[/green]"
    )
    return agent_version.id


# ---------------------------------------------------------------------------
# Runtime: search + generate
# ---------------------------------------------------------------------------

def _get_openai_client():
    """Get an AzureOpenAI client at the account level (not project-scoped)."""
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
    from openai import AzureOpenAI

    settings = get_settings()
    project_endpoint = settings.azure_ai_project.project_endpoint
    base_endpoint = project_endpoint.split("/api/projects/")[0]

    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        azure_endpoint=base_endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2025-03-01-preview",
    )


def _get_search_client():
    """Get an Azure AI Search client for the pubmedqa index."""
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient

    settings = get_settings()
    return SearchClient(
        endpoint=settings.azure_search.endpoint,
        index_name=settings.azure_search.index,
        credential=DefaultAzureCredential(),
    )


def _retrieve(openai_client, search_client, question: str, top_k: int = 5) -> list[dict]:
    """Retrieve relevant documents using vector + semantic hybrid search."""
    from azure.search.documents.models import VectorizedQuery

    settings = get_settings()

    emb_resp = openai_client.embeddings.create(
        model=settings.embedding_deployment,
        input=[question],
    )
    q_vector = emb_resp.data[0].embedding

    results = search_client.search(
        search_text=question,
        vector_queries=[
            VectorizedQuery(
                vector=q_vector,
                k_nearest_neighbors=top_k,
                fields="content_vector",
            )
        ],
        query_type="semantic",
        semantic_configuration_name=settings.search_semantic_config,
        top=top_k,
        select=["question", "context", "long_answer", "final_decision"],
    )

    docs = []
    for r in results:
        docs.append({
            "question": r["question"],
            "context": r["context"],
            "long_answer": r.get("long_answer", ""),
            "final_decision": r.get("final_decision", ""),
        })
    return docs


def _generate(openai_client, question: str, evidence: list[dict]) -> str:
    """Generate an answer grounded in the retrieved evidence."""
    settings = get_settings()

    context_parts = []
    for i, doc in enumerate(evidence, 1):
        context_parts.append(
            f"[{i}] Question: {doc['question']}\n"
            f"    Context: {doc['context'][:500]}\n"
            f"    Answer: {doc.get('long_answer', 'N/A')}\n"
            f"    Decision: {doc.get('final_decision', 'N/A')}"
        )
    context_block = "\n\n".join(context_parts)

    instructions = settings.yaml_config.get("agent", {}).get(
        "instructions",
        (
            "You are a medical question-answering assistant. "
            "Use ONLY the provided evidence to answer."
        ),
    )

    response = openai_client.chat.completions.create(
        model=settings.agent_model,
        messages=[
            {
                "role": "system",
                "content": f"{instructions}\n\nEvidence:\n{context_block}",
            },
            {"role": "user", "content": question},
        ],
        temperature=settings.temperature,
    )
    return response.choices[0].message.content


def query_rag(question: str) -> dict:
    """Run the full RAG pipeline: retrieve then generate.

    Returns a dict with 'question', 'answer', 'evidence' keys.
    """
    openai_client = _get_openai_client()
    search_client = _get_search_client()
    settings = get_settings()

    evidence = _retrieve(
        openai_client, search_client, question, top_k=settings.search_top_k
    )
    answer = _generate(openai_client, question, evidence)

    return {"question": question, "answer": answer, "evidence": evidence}


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def demo() -> None:
    """Run 5 demo queries through the RAG pipeline."""
    settings = get_settings()

    console.print("[bold]Initializing RAG pipeline...[/bold]")

    project_client = _get_project_client()
    _ensure_agent_definition(project_client, settings)

    openai_client = _get_openai_client()
    search_client = _get_search_client()

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
            evidence = _retrieve(
                openai_client, search_client, question,
                top_k=settings.search_top_k,
            )
            console.print(f"  [dim]Retrieved {len(evidence)} documents[/dim]")

            answer = _generate(openai_client, question, evidence)
            console.print(f"[bold green]Response:[/bold green]\n{answer}")
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

    console.print("[bold]Initializing RAG pipeline...[/bold]")

    project_client = _get_project_client()
    _ensure_agent_definition(project_client, settings)

    question = typer.prompt("Enter your medical question")
    console.print(f"\n[bold cyan]Question:[/bold cyan] {question}")

    result = query_rag(question)
    console.print(f"  [dim]Retrieved {len(result['evidence'])} documents[/dim]")
    console.print(f"[bold green]Response:[/bold green]\n{result['answer']}")

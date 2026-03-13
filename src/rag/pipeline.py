"""End-to-end RAG pipeline: Azure AI Search retrieval + local SLM generation.

Architecture:
- **Retrieval**: AzureOpenAI (text-embedding-3-large) embeds the query, then
  Azure AI Search runs vector + semantic hybrid search.
- **Generation**: Phi-4-mini running locally via Ollama produces the answer,
  grounded in the retrieved evidence.
- **Agent definition**: Stored in Azure AI Foundry for portal visibility and
  versioning (the Foundry SDK manages definitions but does not yet expose a
  runtime invoke API).

This split mirrors the RLAIF pipeline: the SLM (phi-4-mini) is the model being
improved, while Azure services provide the retrieval infrastructure and the AI
judge (gpt-5.4) is only used for evaluation/preference generation.
"""

from __future__ import annotations

import typer
from rich.console import Console

from src.config import get_settings

console = Console()


# ---------------------------------------------------------------------------
# Foundry agent definition (CRUD — visible in the portal)
# ---------------------------------------------------------------------------

def _get_project_client():
    """Create an AIProjectClient using managed identity."""
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    settings = get_settings()
    endpoint = settings.azure_ai_project.project_endpoint
    if not endpoint:
        console.print(
            "[red]AZURE_AI_PROJECT_ENDPOINT not set[/red]"
        )
        raise typer.Exit(1)

    return AIProjectClient(
        endpoint=endpoint,
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
        if conn.type in (
            ConnectionType.AZURE_AI_SEARCH, "AzureAISearch"
        ):
            return conn.id

    raise RuntimeError(
        "No Azure AI Search connection found. "
        "Ensure Bicep created 'ai-search-connection'."
    )


def _ensure_agent_definition(project_client, settings) -> str:
    """Ensure the agent definition exists in Foundry.

    Returns the agent version id (e.g. 'pubmedqa-rag:1').
    The definition records which model and tools the agent
    uses; it's visible and testable in the Foundry portal.
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
        details = project_client.agents.get(agent_name)
        aid = details.versions.latest.id
        console.print(
            f"  [green]Found agent '{agent_name}' "
            f"(id: {aid})[/green]"
        )
        return aid
    except (ResourceNotFoundError, Exception):
        pass

    console.print(
        f"  [yellow]Agent '{agent_name}' not found "
        f"— creating...[/yellow]"
    )

    conn_id = _get_search_connection_id(project_client)

    qt_str = settings.yaml_config.get("search", {}).get(
        "query_type", "vector_semantic_hybrid"
    )
    qt_map = {
        "simple": AzureAISearchQueryType.SIMPLE,
        "semantic": AzureAISearchQueryType.SEMANTIC,
        "vector": AzureAISearchQueryType.VECTOR,
        "vector_simple_hybrid": (
            AzureAISearchQueryType.VECTOR_SIMPLE_HYBRID
        ),
        "vector_semantic_hybrid": (
            AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID
        ),
    }
    qt = qt_map.get(
        qt_str,
        AzureAISearchQueryType.VECTOR_SEMANTIC_HYBRID,
    )

    search_tool = AzureAISearchTool(
        azure_ai_search=AzureAISearchToolResource(
            indexes=[
                AISearchIndexResource(
                    project_connection_id=conn_id,
                    index_name=settings.azure_search.index,
                    query_type=qt,
                    top_k=settings.search_top_k,
                )
            ]
        )
    )

    instructions = settings.yaml_config.get(
        "agent", {}
    ).get(
        "instructions",
        "You are a medical QA assistant.",
    )

    defn = PromptAgentDefinition(
        model=settings.agent_model,
        instructions=instructions,
        tools=[search_tool],
        temperature=settings.temperature,
    )

    ver = project_client.agents.create_version(
        agent_name=agent_name,
        definition=defn,
        description="PubMedQA RAG — phi-4-mini SLM",
    )

    console.print(
        f"  [green]Created agent '{agent_name}' "
        f"(id: {ver.id})[/green]"
    )
    return ver.id


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def _get_azure_openai_client():
    """AzureOpenAI client for embeddings (account-level)."""
    from azure.identity import (
        DefaultAzureCredential,
        get_bearer_token_provider,
    )
    from openai import AzureOpenAI

    settings = get_settings()
    endpoint = settings.azure_ai_project.project_endpoint
    base = endpoint.split("/api/projects/")[0]

    tp = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        azure_endpoint=base,
        azure_ad_token_provider=tp,
        api_version="2025-03-01-preview",
    )


def _get_slm_client():
    """OpenAI-compatible client pointing at local Ollama."""
    from openai import OpenAI

    settings = get_settings()
    return OpenAI(
        base_url=settings.slm_ollama_base_url,
        api_key="ollama",
    )


def _get_search_client():
    """Azure AI Search client for the pubmedqa index."""
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient

    settings = get_settings()
    return SearchClient(
        endpoint=settings.azure_search.endpoint,
        index_name=settings.azure_search.index,
        credential=DefaultAzureCredential(),
    )


# ---------------------------------------------------------------------------
# Retrieve (Azure) + Generate (local SLM)
# ---------------------------------------------------------------------------

def _retrieve(
    azure_client, search_client, question: str, top_k: int = 5
) -> list[dict]:
    """Embed the question and run hybrid search."""
    from azure.search.documents.models import VectorizedQuery

    settings = get_settings()

    emb = azure_client.embeddings.create(
        model=settings.embedding_deployment,
        input=[question],
    )
    q_vec = emb.data[0].embedding

    results = search_client.search(
        search_text=question,
        vector_queries=[
            VectorizedQuery(
                vector=q_vec,
                k_nearest_neighbors=top_k,
                fields="content_vector",
            )
        ],
        query_type="semantic",
        semantic_configuration_name=(
            settings.search_semantic_config
        ),
        top=top_k,
        select=[
            "question", "context",
            "long_answer", "final_decision",
        ],
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


def _generate(
    slm_client, question: str, evidence: list[dict]
) -> str:
    """Generate an answer using the local SLM (phi-4-mini)."""
    settings = get_settings()

    parts = []
    for i, doc in enumerate(evidence, 1):
        parts.append(
            f"[{i}] Question: {doc['question']}\n"
            f"    Context: {doc['context'][:500]}\n"
            f"    Answer: {doc.get('long_answer', 'N/A')}\n"
            f"    Decision: "
            f"{doc.get('final_decision', 'N/A')}"
        )
    ctx = "\n\n".join(parts)

    instructions = settings.yaml_config.get(
        "agent", {}
    ).get(
        "instructions",
        "You are a medical QA assistant. "
        "Use ONLY the provided evidence.",
    )

    response = slm_client.chat.completions.create(
        model=settings.slm_ollama_model,
        messages=[
            {
                "role": "system",
                "content": (
                    f"{instructions}\n\nEvidence:\n{ctx}"
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=settings.temperature,
    )
    return response.choices[0].message.content


def query_rag(question: str) -> dict:
    """Full RAG pipeline: retrieve (Azure) then generate (local SLM).

    Returns dict with 'question', 'answer', 'evidence'.
    """
    azure_client = _get_azure_openai_client()
    slm_client = _get_slm_client()
    search_client = _get_search_client()
    settings = get_settings()

    evidence = _retrieve(
        azure_client, search_client, question,
        top_k=settings.search_top_k,
    )
    answer = _generate(slm_client, question, evidence)

    return {
        "question": question,
        "answer": answer,
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def demo() -> None:
    """Run 5 demo queries through the RAG pipeline."""
    settings = get_settings()

    console.print("[bold]Initializing RAG pipeline...[/bold]")
    console.print(
        f"  SLM: [cyan]{settings.slm_ollama_model}[/cyan] "
        f"via Ollama ({settings.slm_ollama_base_url})"
    )
    console.print(
        "  Embeddings: [cyan]"
        f"{settings.embedding_deployment}[/cyan] via Azure"
    )
    console.print(
        "  Search: [cyan]"
        f"{settings.azure_search.index}[/cyan] "
        "(vector + semantic hybrid)"
    )

    project_client = _get_project_client()
    _ensure_agent_definition(project_client, settings)

    azure_client = _get_azure_openai_client()
    slm_client = _get_slm_client()
    search_client = _get_search_client()

    questions = [
        "Does aspirin reduce the risk of cardiovascular "
        "events?",
        "Is metformin effective for type 2 diabetes "
        "management?",
        "Can regular exercise reduce the risk of "
        "Alzheimer's disease?",
        "Does vitamin D supplementation prevent "
        "fractures in elderly patients?",
        "Is there evidence that statins reduce mortality "
        "in heart disease patients?",
    ]

    for i, q in enumerate(questions, 1):
        console.print(f"\n[bold cyan]Question {i}:[/bold cyan] {q}")
        try:
            evidence = _retrieve(
                azure_client, search_client, q,
                top_k=settings.search_top_k,
            )
            console.print(
                f"  [dim]Retrieved {len(evidence)} docs[/dim]"
            )
            answer = _generate(slm_client, q, evidence)
            console.print(
                f"[bold green]Response:[/bold green]\n{answer}"
            )
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
        console.print("─" * 72)

    console.print(
        f"\n[bold green]Demo complete — agent "
        f"'{settings.agent_name}' persists in "
        f"Foundry.[/bold green]"
    )


def query() -> None:
    """Interactive single query through the RAG pipeline."""
    settings = get_settings()

    console.print("[bold]Initializing RAG pipeline...[/bold]")
    console.print(
        f"  SLM: [cyan]{settings.slm_ollama_model}[/cyan] "
        f"via Ollama"
    )

    project_client = _get_project_client()
    _ensure_agent_definition(project_client, settings)

    question = typer.prompt("Enter your medical question")
    console.print(
        f"\n[bold cyan]Question:[/bold cyan] {question}"
    )

    result = query_rag(question)
    console.print(
        f"  [dim]Retrieved {len(result['evidence'])} "
        f"docs[/dim]"
    )
    console.print(
        f"[bold green]Response:[/bold green]\n"
        f"{result['answer']}"
    )

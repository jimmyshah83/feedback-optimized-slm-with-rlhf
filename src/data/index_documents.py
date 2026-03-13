"""Embed preprocessed documents and index into Azure AI Search.

Uses the AIProjectClient to get an OpenAI-compatible client for embeddings,
and the Azure Search SDK with DefaultAzureCredential for index management.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.progress import track

from src.config import get_settings

app = typer.Typer()
console = Console()

UPLOAD_BATCH_SIZE = 100
EMBED_BATCH_SIZE = 16


def _create_index(
    index_client,
    index_name: str,
    dimensions: int,
    semantic_config_name: str,
) -> None:
    """Create or update the Azure AI Search index with vector + semantic config."""
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        SearchableField,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SemanticConfiguration,
        SemanticField,
        SemanticPrioritizedFields,
        SemanticSearch,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="question", type=SearchFieldDataType.String),
        SearchableField(name="context", type=SearchFieldDataType.String),
        SearchableField(name="long_answer", type=SearchFieldDataType.String),
        SimpleField(
            name="final_decision",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=dimensions,
            vector_search_profile_name="vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-algo")],
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-algo",
            )
        ],
    )

    semantic_config = SemanticConfiguration(
        name=semantic_config_name,
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="context")],
            title_field=SemanticField(field_name="question"),
        ),
    )

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=SemanticSearch(configurations=[semantic_config]),
    )

    index_client.create_or_update_index(index)


def _get_embeddings(openai_client, texts: list[str], model: str) -> list[list[float]]:
    """Get embeddings for a batch of texts via the project's OpenAI client."""
    response = openai_client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]


def _load_documents(processed_path) -> list[dict]:
    """Load preprocessed documents from JSONL."""
    docs = []
    with open(processed_path) as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


@app.command()
def run(
    rebuild: bool = typer.Option(False, "--rebuild", help="Delete and recreate the index"),
) -> None:
    """Embed and index documents into Azure AI Search."""
    from azure.identity import DefaultAzureCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient

    settings = get_settings()
    credential = DefaultAzureCredential()

    # --- Load documents ---
    processed_path = settings.processed_dir / settings.processed_file
    console.print(f"[bold]Loading documents from {processed_path}...[/bold]")
    documents = _load_documents(processed_path)
    console.print(f"  Loaded {len(documents)} documents")

    # --- Create/update search index ---
    search_endpoint = settings.azure_search.endpoint
    index_name = settings.azure_search.index

    if not search_endpoint:
        console.print("[red]AZURE_SEARCH_ENDPOINT not set in .env[/red]")
        raise typer.Exit(1)

    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)

    if rebuild:
        try:
            index_client.delete_index(index_name)
            console.print(f"  [yellow]Deleted existing index '{index_name}'[/yellow]")
        except Exception:
            pass

    console.print(f"[bold]Creating/updating index '{index_name}'...[/bold]")
    _create_index(
        index_client,
        index_name,
        dimensions=settings.embedding_dimensions,
        semantic_config_name=settings.search_semantic_config,
    )
    console.print(f"  [green]Index '{index_name}' ready[/green]")

    # --- Get AzureOpenAI client for embeddings (managed identity) ---
    # Embeddings use the Foundry account endpoint, not the project-scoped path
    project_endpoint = settings.azure_ai_project.project_endpoint
    if not project_endpoint:
        console.print("[red]AZURE_AI_PROJECT_ENDPOINT not set in .env[/red]")
        raise typer.Exit(1)

    from azure.identity import get_bearer_token_provider
    from openai import AzureOpenAI

    base_endpoint = project_endpoint.split("/api/projects/")[0]
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    openai_client = AzureOpenAI(
        azure_endpoint=base_endpoint,
        azure_ad_token_provider=token_provider,
        api_version="2024-10-21",
    )
    embedding_model = settings.embedding_deployment

    console.print(
        f"[bold]Embedding documents with '{embedding_model}' "
        f"({settings.embedding_dimensions}d)...[/bold]"
    )

    # --- Embed and upload in batches ---
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=credential,
    )

    uploaded = 0
    for batch_start in track(
        range(0, len(documents), UPLOAD_BATCH_SIZE),
        description="Indexing",
    ):
        batch = documents[batch_start : batch_start + UPLOAD_BATCH_SIZE]

        texts = [doc["context"] for doc in batch]
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            sub = texts[i : i + EMBED_BATCH_SIZE]
            embeddings.extend(_get_embeddings(openai_client, sub, embedding_model))

        search_docs = []
        for doc, embedding in zip(batch, embeddings):
            search_docs.append(
                {
                    "id": doc["pubid"],
                    "question": doc["question"],
                    "context": doc["context"],
                    "long_answer": doc["long_answer"],
                    "final_decision": doc["final_decision"],
                    "content_vector": embedding,
                }
            )

        result = search_client.upload_documents(search_docs)
        uploaded += sum(1 for r in result if r.succeeded)

    console.print(
        f"[green]Indexed {uploaded}/{len(documents)} documents "
        f"into '{index_name}'[/green]"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

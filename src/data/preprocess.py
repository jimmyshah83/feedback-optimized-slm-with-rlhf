"""Preprocess PubMedQA into retrieval-ready documents."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.progress import track

from src.config import get_settings

app = typer.Typer()
console = Console()


def _build_document(record: dict) -> dict:
    """Transform a raw PubMedQA record into a flat document for indexing and evaluation."""
    context = record.get("context", {})

    if isinstance(context, dict):
        abstracts = context.get("contexts", [])
        labels = context.get("labels", [])
        meshes = context.get("meshes", [])
    else:
        abstracts = [str(context)]
        labels = []
        meshes = []

    combined_context = "\n\n".join(
        f"[{label}] {text}" if label else text
        for label, text in zip(
            labels + [""] * max(0, len(abstracts) - len(labels)),
            abstracts,
        )
    )

    return {
        "pubid": str(record.get("pubid", "")),
        "question": record["question"],
        "context": combined_context,
        "long_answer": record.get("long_answer", ""),
        "final_decision": record.get("final_decision", ""),
        "meshes": meshes,
    }


def preprocess_dataset(output_path: str | None = None) -> list[dict]:
    """Load raw PubMedQA, transform, and write to JSONL."""
    from datasets import load_dataset

    settings = get_settings()
    out_dir = settings.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_path or str(out_dir / settings.processed_file)

    console.print(f"[bold]Loading raw dataset from {settings.raw_dir}...[/bold]")

    ds = load_dataset(
        settings.dataset_name,
        settings.dataset_split,
        cache_dir=str(settings.raw_dir),
    )
    train_split = ds["train"]

    documents: list[dict] = []
    for record in track(train_split, description="Preprocessing"):
        documents.append(_build_document(record))

    with open(out_file, "w") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")

    console.print(f"[green]Wrote {len(documents)} documents to {out_file}[/green]")

    non_empty = sum(1 for d in documents if d["context"].strip())
    console.print(f"  Documents with context: {non_empty}/{len(documents)}")
    decisions = {}
    for d in documents:
        dec = d["final_decision"] or "unknown"
        decisions[dec] = decisions.get(dec, 0) + 1
    console.print(f"  Decision distribution: {decisions}")

    return documents


@app.command()
def run(
    output: str = typer.Option(None, help="Override output JSONL file path"),
) -> None:
    """Preprocess PubMedQA into retrieval-ready documents."""
    preprocess_dataset(output)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

"""Download the PubMedQA dataset from Hugging Face and cache it locally."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from src.config import get_settings

app = typer.Typer()
console = Console()


def _download(cache_dir: str | None = None) -> None:
    from datasets import load_dataset

    settings = get_settings()
    raw_dir = cache_dir or str(settings.raw_dir)

    console.print(
        f"[bold]Downloading {settings.dataset_name} "
        f"(split: {settings.dataset_split})...[/bold]"
    )

    ds = load_dataset(
        settings.dataset_name,
        settings.dataset_split,
        cache_dir=raw_dir,
    )

    train_split = ds["train"]
    console.print(f"[green]Downloaded {len(train_split)} records to {raw_dir}[/green]")

    table = Table(title="Sample Record")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="white", max_width=80)

    sample = train_split[0]
    for key in ("question", "final_decision", "long_answer"):
        value = str(sample.get(key, "N/A"))
        table.add_row(key, value[:200] + ("..." if len(value) > 200 else ""))

    context_val = sample.get("context", {})
    if isinstance(context_val, dict):
        abstracts = context_val.get("contexts", [])
        table.add_row("context (abstracts)", f"{len(abstracts)} abstract(s)")
    else:
        table.add_row("context", str(context_val)[:200])

    console.print(table)


@app.command()
def run(
    cache_dir: str = typer.Option(None, help="Override raw data cache directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print config and exit"),
) -> None:
    """Download PubMedQA dataset from Hugging Face."""
    settings = get_settings()

    if dry_run:
        console.print("[bold]Dry run — would download:[/bold]")
        console.print(f"  Dataset:  {settings.dataset_name}")
        console.print(f"  Split:    {settings.dataset_split}")
        console.print(f"  Cache to: {cache_dir or settings.raw_dir}")
        raise typer.Exit()

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    _download(cache_dir)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

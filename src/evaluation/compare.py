"""Phase 6: Cross-iteration comparison report for the RL loop.

Reads the loop history and produces a formatted comparison table
showing how metrics improved (or not) across iterations.
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.table import Table

from src.config import PROJECT_ROOT

console = Console()

LOOP_DIR = PROJECT_ROOT / "data" / "rl_loop"


app = typer.Typer()


@app.command()
def run() -> None:
    """Generate comparison report across RL iterations."""
    history_path = LOOP_DIR / "history.json"

    if not history_path.exists():
        console.print(
            "[red]No loop history found.[/red]"
        )
        console.print("  Run: uv run rl-loop")
        raise typer.Exit(1)

    with open(history_path) as f:
        history = json.load(f)

    if not history:
        console.print("[yellow]History is empty.[/yellow]")
        raise typer.Exit(0)

    console.print(
        "[bold]RLAIF Cross-Iteration Comparison[/bold]\n"
    )

    table = Table(title="Metrics by Iteration")
    table.add_column("Iteration", justify="right")
    table.add_column("PubMedQA Acc", justify="right")
    table.add_column("Groundedness", justify="right")
    table.add_column("Relevance", justify="right")
    table.add_column("Completeness", justify="right")
    table.add_column("Timestamp")

    for h in history:
        table.add_row(
            str(h.get("iteration", "?")),
            f"{h.get('pubmedqa_accuracy', 0):.1%}",
            f"{h.get('avg_groundedness', 0):.2f}",
            f"{h.get('avg_relevance', 0):.2f}",
            f"{h.get('avg_completeness', 0):.2f}",
            h.get("timestamp", "")[:19],
        )
    console.print(table)

    if len(history) >= 2:
        console.print("\n[bold]Deltas (last vs first):[/bold]")
        first = history[0]
        last = history[-1]

        metrics = [
            ("PubMedQA Accuracy",
             "pubmedqa_accuracy", True),
            ("Groundedness",
             "avg_groundedness", False),
            ("Relevance",
             "avg_relevance", False),
            ("Completeness",
             "avg_completeness", False),
        ]

        delta_table = Table()
        delta_table.add_column("Metric")
        delta_table.add_column("First", justify="right")
        delta_table.add_column("Last", justify="right")
        delta_table.add_column("Delta", justify="right")

        def _fmt_pct(v: float) -> str:
            return f"{v:.1%}"

        def _fmt_dec(v: float) -> str:
            return f"{v:.2f}"

        for label, key, is_pct in metrics:
            v0 = first.get(key, 0)
            v1 = last.get(key, 0)
            delta = v1 - v0

            fmt = _fmt_pct if is_pct else _fmt_dec
            d_str = (
                f"{delta:+.1%}" if is_pct
                else f"{delta:+.2f}"
            )

            color = "green" if delta > 0 else (
                "red" if delta < 0 else "white"
            )
            delta_table.add_row(
                label, fmt(v0), fmt(v1),
                f"[{color}]{d_str}[/{color}]",
            )

        console.print(delta_table)

    best_idx = max(
        range(len(history)),
        key=lambda i: history[i].get(
            "avg_relevance", 0
        ),
    )
    console.print(
        f"\n  Best iteration: "
        f"[bold cyan]{best_idx}[/bold cyan] "
        f"(relevance = "
        f"{history[best_idx].get('avg_relevance', 0):.2f})"
    )


def main() -> None:
    app()

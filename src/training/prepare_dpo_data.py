"""Phase 5: Convert AI judge preference pairs into DPO training format.

Reads `data/judge/preferences_iter{N}.jsonl` and produces a
HuggingFace Dataset with columns (prompt, chosen, rejected)
ready for TRL's DPOTrainer.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from src.config import PROJECT_ROOT, get_settings

console = Console()

JUDGE_DIR = PROJECT_ROOT / "data" / "judge"
TRAINING_DIR = PROJECT_ROOT / "data" / "training"


def load_preferences(path: Path) -> list[dict]:
    """Load preference pairs from judge JSONL."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def format_for_dpo(
    pairs: list[dict],
) -> list[dict]:
    """Convert judge pairs to DPOTrainer format.

    DPOTrainer expects: prompt (str), chosen (str), rejected (str).
    We include the RAG context in the prompt so the model learns
    to generate grounded answers.
    """
    settings = get_settings()
    instructions = settings.yaml_config.get(
        "agent", {}
    ).get(
        "instructions",
        "You are a medical QA assistant.",
    )

    formatted = []
    for p in pairs:
        context = p.get("context", "")
        question = p["question"]

        prompt = (
            f"{instructions}\n\n"
            f"Evidence:\n{context}\n\n"
            f"Question: {question}"
        )

        formatted.append({
            "prompt": prompt,
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        })

    return formatted


app = typer.Typer()


@app.command()
def run(
    iteration: int = typer.Option(
        0, help="RL loop iteration number"
    ),
) -> None:
    """Prepare DPO training data from AI judge results."""
    console.print("[bold]Phase 5a: Prepare DPO Data[/bold]")

    src_path = JUDGE_DIR / f"preferences_iter{iteration}.jsonl"
    if not src_path.exists():
        console.print(
            f"[red]Judge output not found: {src_path}[/red]"
        )
        console.print(
            "  Run: uv run judge "
            f"--iteration {iteration}"
        )
        raise typer.Exit(1)

    pairs = load_preferences(src_path)
    console.print(
        f"  Loaded {len(pairs)} preference pairs "
        f"from iteration {iteration}"
    )

    formatted = format_for_dpo(pairs)

    out_dir = TRAINING_DIR / f"iter{iteration}" / "dpo_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.jsonl"

    with open(out_path, "w") as f:
        for row in formatted:
            f.write(json.dumps(row) + "\n")

    console.print(
        f"  Saved {len(formatted)} DPO examples "
        f"to [dim]{out_path}[/dim]"
    )

    stats = {
        "total": len(formatted),
        "avg_prompt_len": (
            sum(len(r["prompt"]) for r in formatted)
            // max(len(formatted), 1)
        ),
        "avg_chosen_len": (
            sum(len(r["chosen"]) for r in formatted)
            // max(len(formatted), 1)
        ),
        "avg_rejected_len": (
            sum(len(r["rejected"]) for r in formatted)
            // max(len(formatted), 1)
        ),
    }
    console.print(f"  Stats: {stats}")


def main() -> None:
    app()

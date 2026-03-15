"""Phase 6: Iterative RLAIF loop orchestrator.

Runs the full cycle: judge → prepare-dpo → train → merge → benchmark
for N iterations, tracking convergence and producing a comparison
report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.config import PROJECT_ROOT, get_settings

console = Console()

LOOP_DIR = PROJECT_ROOT / "data" / "rl_loop"


app = typer.Typer()


@app.command()
def run(
    iterations: int = typer.Option(
        0, help="Number of iterations (0 = use config)"
    ),
    questions: int = typer.Option(
        0, help="Questions per iteration (0 = use config)"
    ),
    eval_samples: int = typer.Option(
        0, help="Eval samples per benchmark (0 = use config)"
    ),
    skip_training: bool = typer.Option(
        False,
        help="Skip DPO training (judge + benchmark only)",
    ),
) -> None:
    """Run the iterative RLAIF pipeline."""
    settings = get_settings()

    if iterations <= 0:
        iterations = settings.rl_iterations
    if questions <= 0:
        questions = settings.rl_questions_per_iteration
    if eval_samples <= 0:
        eval_samples = settings.eval_count

    console.print("[bold]Phase 6: RLAIF Loop[/bold]")
    console.print(
        f"  Iterations: [cyan]{iterations}[/cyan]"
    )
    console.print(
        f"  Questions/iter: [cyan]{questions}[/cyan]"
    )
    console.print(
        f"  Eval samples: [cyan]{eval_samples}[/cyan]"
    )
    console.print(
        f"  Skip training: [cyan]{skip_training}[/cyan]"
    )

    LOOP_DIR.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    prev_score: float | None = None

    for i in range(iterations):
        console.print(
            f"\n{'=' * 60}"
            f"\n[bold cyan]  Iteration {i} / {iterations}"
            f"[/bold cyan]\n{'=' * 60}"
        )

        iter_result = _run_iteration(
            iteration=i,
            questions=questions,
            eval_samples=eval_samples,
            skip_training=skip_training,
        )

        history.append(iter_result)

        _save_history(history)

        current_score = iter_result.get(
            "avg_relevance", 0.0
        )

        if prev_score is not None:
            delta = current_score - prev_score
            console.print(
                f"\n  Score delta: "
                f"[cyan]{delta:+.3f}[/cyan]"
            )
            threshold = settings.rl_convergence_threshold
            if abs(delta) < threshold:
                console.print(
                    f"  [yellow]Converged "
                    f"(delta < {threshold})[/yellow]"
                )
                break

        prev_score = current_score

    console.print(
        "\n[bold green]RLAIF loop complete![/bold green]"
    )
    _print_summary_table(history)


def _run_iteration(
    iteration: int,
    questions: int,
    eval_samples: int,
    skip_training: bool,
) -> dict:
    """Execute one full iteration of the RLAIF loop."""
    result: dict = {
        "iteration": iteration,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    console.print(
        "\n[bold]Step 1: AI Judge[/bold]"
    )
    _run_judge(iteration, questions)
    result["judge_questions"] = questions

    if not skip_training:
        console.print(
            "\n[bold]Step 2: Prepare DPO data[/bold]"
        )
        _run_prepare_dpo(iteration)

        console.print(
            "\n[bold]Step 3: DPO Training[/bold]"
        )
        _run_train_dpo(iteration)

        console.print(
            "\n[bold]Step 4: Merge adapter[/bold]"
        )
        _run_merge_adapter(iteration)

    console.print(
        "\n[bold]Step 5: Benchmark[/bold]"
    )
    metrics = _run_benchmark(iteration, eval_samples)
    result.update(metrics)

    return result


def _run_judge(iteration: int, questions: int) -> None:
    from src.judge.ai_judge import run as judge_fn

    judge_fn(iteration=iteration, samples=questions)


def _run_prepare_dpo(iteration: int) -> None:
    from src.training.prepare_dpo_data import (
        run as prep_fn,
    )

    prep_fn(iteration=iteration)


def _run_train_dpo(iteration: int) -> None:
    from src.training.dpo_trainer import run as train_fn

    train_fn(iteration=iteration, model_path="")


def _run_merge_adapter(iteration: int) -> None:
    from src.training.merge_adapter import (
        run as merge_fn,
    )

    merge_fn(iteration=iteration, register_ollama=False)


def _run_benchmark(
    iteration: int, eval_samples: int
) -> dict:
    """Run benchmark and return key metrics."""
    from src.evaluation.benchmarks import (
        _compute_pubmedqa_accuracy,
        _generate_responses,
        _load_eval_questions,
        _run_ai_evaluation,
    )

    settings = get_settings()
    processed = (
        settings.processed_dir / settings.processed_file
    )
    eval_questions = _load_eval_questions(
        processed, settings.eval_count
    )

    results = _generate_responses(
        eval_questions, eval_samples
    )
    accuracy = _compute_pubmedqa_accuracy(results)

    eval_dir = PROJECT_ROOT / "data" / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    eval_name = f"rl_iter{iteration}_{ts}"

    data_path = eval_dir / f"{eval_name}.jsonl"
    with open(data_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    try:
        eval_result = _run_ai_evaluation(
            str(data_path), eval_name
        )
        ai_metrics = eval_result.get("metrics", {})
    except Exception as exc:
        console.print(
            f"  [yellow]AI Evaluation error: {exc}"
            f"[/yellow]"
        )
        ai_metrics = {}

    return {
        "pubmedqa_accuracy": accuracy.get(
            "accuracy", 0.0
        ),
        "avg_groundedness": ai_metrics.get(
            "groundedness.groundedness", 0.0
        ),
        "avg_relevance": ai_metrics.get(
            "relevance.relevance", 0.0
        ),
        "avg_completeness": ai_metrics.get(
            "completeness.response_completeness", 0.0
        ),
        "eval_name": eval_name,
    }


def _save_history(history: list[dict]) -> None:
    out = LOOP_DIR / "history.json"
    with open(out, "w") as f:
        json.dump(history, f, indent=2)


def _print_summary_table(history: list[dict]) -> None:
    table = Table(title="RLAIF Iteration Summary")
    table.add_column("Iter", justify="right")
    table.add_column("PubMedQA Acc", justify="right")
    table.add_column("Groundedness", justify="right")
    table.add_column("Relevance", justify="right")
    table.add_column("Completeness", justify="right")

    for h in history:
        table.add_row(
            str(h.get("iteration", "?")),
            f"{h.get('pubmedqa_accuracy', 0):.1%}",
            f"{h.get('avg_groundedness', 0):.2f}",
            f"{h.get('avg_relevance', 0):.2f}",
            f"{h.get('avg_completeness', 0):.2f}",
        )
    console.print(table)


def main() -> None:
    app()

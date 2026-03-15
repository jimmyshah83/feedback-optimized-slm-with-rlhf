"""Phase 3: Benchmark the RAG pipeline using Azure AI Evaluation SDK.

Runs phi-4-mini RAG on the held-out evaluation set, scores with
Groundedness / Relevance / ResponseCompleteness via gpt-5.4,
computes PubMedQA classification accuracy, and logs everything
to Azure AI Foundry for portal visibility.
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

RESULTS_DIR = PROJECT_ROOT / "data" / "evaluations"


def _load_eval_questions(
    path: Path, eval_count: int
) -> list[dict]:
    """Load the last `eval_count` docs from processed JSONL."""
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    return docs[-eval_count:]


def _generate_responses(
    questions: list[dict], samples: int
) -> list[dict]:
    """Run RAG pipeline on each question, return results."""
    from src.rag.pipeline import (
        _get_azure_openai_client,
        _get_search_client,
        _get_slm_client,
        _retrieve,
        _generate,
    )

    settings = get_settings()
    azure_client = _get_azure_openai_client()
    slm_client = _get_slm_client()
    search_client = _get_search_client()

    subset = questions[:samples]
    results = []

    from rich.progress import track

    for doc in track(
        subset, description="Generating responses"
    ):
        question = doc["question"]
        try:
            evidence = _retrieve(
                azure_client,
                search_client,
                question,
                top_k=settings.search_top_k,
            )
            context = "\n\n".join(
                e["context"][:400] for e in evidence
            )
            answer = _generate(slm_client, question, evidence)
        except Exception as exc:
            context = ""
            answer = f"[ERROR] {exc}"

        results.append({
            "query": question,
            "response": answer,
            "context": context,
            "ground_truth": doc.get("long_answer", ""),
            "expected_decision": doc.get(
                "final_decision", ""
            ),
        })

    return results


_PATCHED = False


def _patch_max_tokens():
    """Patch OpenAI SDK so max_tokens→max_completion_tokens.

    The azure-ai-evaluation SDK's prompty templates pass
    max_tokens, which gpt-5.4 rejects. This interceptor
    transparently renames the parameter at the lowest level.
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    from openai.resources.chat.completions import (
        completions as _cc,
    )

    _orig = _cc.AsyncCompletions.create

    async def _patched(self, *args, **kwargs):
        if "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop(
                "max_tokens"
            )
        return await _orig(self, *args, **kwargs)

    _cc.AsyncCompletions.create = _patched


def _run_ai_evaluation(
    data_path: str,
    evaluation_name: str,
) -> dict:
    """Run Azure AI Evaluation SDK evaluators and log to Foundry."""
    _patch_max_tokens()

    from azure.ai.evaluation import (
        EvaluatorConfig,
        GroundednessEvaluator,
        RelevanceEvaluator,
        ResponseCompletenessEvaluator,
        evaluate,
    )
    from azure.identity import DefaultAzureCredential

    settings = get_settings()
    credential = DefaultAzureCredential()

    endpoint = settings.azure_ai_project.project_endpoint
    base_endpoint = endpoint.split("/api/projects/")[0]

    token = credential.get_token(
        "https://cognitiveservices.azure.com/.default"
    ).token

    model_config = {
        "type": "azure_openai",
        "azure_deployment": settings.judge_chat_deployment,
        "azure_endpoint": base_endpoint,
        "api_key": token,
        "api_version": "2025-03-01-preview",
    }

    evaluators = {
        "groundedness": GroundednessEvaluator(
            model_config=model_config
        ),
        "relevance": RelevanceEvaluator(
            model_config=model_config
        ),
        "completeness": ResponseCompletenessEvaluator(
            model_config=model_config
        ),
    }

    evaluator_config = {
        "groundedness": EvaluatorConfig(
            column_mapping={
                "query": "${data.query}",
                "response": "${data.response}",
                "context": "${data.context}",
            }
        ),
        "relevance": EvaluatorConfig(
            column_mapping={
                "query": "${data.query}",
                "response": "${data.response}",
                "context": "${data.context}",
            }
        ),
        "completeness": EvaluatorConfig(
            column_mapping={
                "query": "${data.query}",
                "response": "${data.response}",
                "ground_truth": "${data.ground_truth}",
            }
        ),
    }

    project_endpoint = (
        settings.azure_ai_project.project_endpoint
    )

    output_path = str(
        RESULTS_DIR / f"{evaluation_name}_details"
    )

    result = evaluate(
        data=data_path,
        evaluators=evaluators,
        evaluator_config=evaluator_config,
        evaluation_name=evaluation_name,
        azure_ai_project=project_endpoint,
        output_path=output_path,
    )

    return result


def _compute_pubmedqa_accuracy(results: list[dict]) -> dict:
    """Compute PubMedQA classification accuracy."""
    from src.evaluation.metrics import pubmedqa_accuracy

    responses = [r["response"] for r in results]
    ground_truths = [r["expected_decision"] for r in results]
    return pubmedqa_accuracy(responses, ground_truths)


app = typer.Typer()


@app.command()
def run(
    tag: str = typer.Option(
        "baseline", help="Tag for this evaluation run"
    ),
    samples: int = typer.Option(
        0,
        help="Number of eval samples (0 = use config)",
    ),
) -> None:
    """Run benchmark evaluation on the held-out set."""
    settings = get_settings()

    if samples <= 0:
        samples = settings.eval_count

    console.print("[bold]Phase 3: Benchmark Evaluation[/bold]")
    console.print(
        f"  SLM: [cyan]{settings.slm_ollama_model}[/cyan]"
    )
    console.print(
        f"  Judge: [cyan]{settings.judge_chat_deployment}"
        f"[/cyan] (Azure)"
    )
    console.print(f"  Samples: [cyan]{samples}[/cyan]")
    console.print(f"  Tag: [cyan]{tag}[/cyan]")

    processed = (
        settings.processed_dir / settings.processed_file
    )
    eval_questions = _load_eval_questions(
        processed, settings.eval_count
    )
    console.print(
        f"  Loaded {len(eval_questions)} eval questions"
    )

    console.print(
        "\n[bold]Step 1: Generate RAG responses "
        f"({samples} questions)...[/bold]"
    )
    results = _generate_responses(eval_questions, samples)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    eval_name = f"benchmark_{tag}_{ts}"

    data_path = RESULTS_DIR / f"{eval_name}.jsonl"
    with open(data_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    console.print(
        f"  Saved responses to [dim]{data_path}[/dim]"
    )

    console.print(
        "\n[bold]Step 2: PubMedQA classification "
        "accuracy...[/bold]"
    )
    accuracy = _compute_pubmedqa_accuracy(results)

    acc_table = Table(title="PubMedQA Accuracy")
    acc_table.add_column("Metric")
    acc_table.add_column("Value", justify="right")
    acc_table.add_row(
        "Accuracy", f"{accuracy['accuracy']:.1%}"
    )
    acc_table.add_row(
        "Correct / Parsed",
        f"{accuracy['correct']} / {accuracy['parsed']}",
    )
    acc_table.add_row(
        "Unparsed", str(accuracy["unparsed"])
    )
    for cls in ("yes", "no", "maybe"):
        c = accuracy["per_class"][cls]
        if c["total"] > 0:
            pct = c["correct"] / c["total"]
            acc_table.add_row(
                f"  {cls}",
                f"{c['correct']}/{c['total']} "
                f"({pct:.0%})",
            )
    console.print(acc_table)

    console.print(
        "\n[bold]Step 3: Azure AI Evaluation "
        "(Groundedness, Relevance, Completeness)...[/bold]"
    )
    console.print(
        "  Scoring with gpt-5.4 + logging to Foundry..."
    )

    eval_result = _run_ai_evaluation(
        data_path=str(data_path),
        evaluation_name=eval_name,
    )

    metrics = eval_result.get("metrics", {})
    studio_url = eval_result.get("studio_url", None)

    eval_table = Table(title="Azure AI Evaluation Metrics")
    eval_table.add_column("Metric")
    eval_table.add_column("Score", justify="right")
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            eval_table.add_row(k, f"{v:.3f}")
        else:
            eval_table.add_row(k, str(v))
    console.print(eval_table)

    summary = {
        "evaluation_name": eval_name,
        "tag": tag,
        "timestamp": ts,
        "slm_model": settings.slm_ollama_model,
        "samples": samples,
        "pubmedqa_accuracy": accuracy,
        "ai_evaluation_metrics": metrics,
        "studio_url": studio_url,
    }
    summary_path = RESULTS_DIR / f"{eval_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    console.print(
        "\n[bold green]Benchmark complete![/bold green]"
    )
    console.print(
        f"  Summary: [dim]{summary_path}[/dim]"
    )
    if studio_url:
        console.print(
            f"  Foundry: [dim]{studio_url}[/dim]"
        )
    else:
        console.print(
            "  [yellow]Foundry upload skipped — check"
            " azure_ai_project config[/yellow]"
        )


def main() -> None:
    app()

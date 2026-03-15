"""Phase 4: AI Judge — gpt-5.4 generates DPO preference pairs.

For each training question the SLM (phi-4-mini) generates two
candidate responses at different temperatures.  gpt-5.4 then
evaluates both against the PubMedQA gold answer on four rubric
dimensions (medical accuracy, faithfulness, completeness, clarity)
and picks a winner, producing (chosen, rejected) pairs for DPO.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from src.config import PROJECT_ROOT, get_settings

console = Console()

JUDGE_DIR = PROJECT_ROOT / "data" / "judge"

JUDGE_SYSTEM_PROMPT = """\
You are an expert medical QA evaluator.  You will receive:
- A QUESTION
- The GOLD reference answer (expert-written)
- Two CANDIDATE responses (A and B)

Evaluate each candidate on four dimensions (1–5 scale):
1. **medical_accuracy** – factual correctness relative to the gold answer
2. **faithfulness** – how well grounded the response is in the provided evidence
3. **completeness** – does it cover the key points from the gold answer?
4. **clarity** – is the response well-structured and easy to understand?

Return STRICTLY the following JSON (no markdown, no extra text):
{
  "winner": "A" or "B",
  "dimension_scores_a": [
    {"dimension": "<dim>", "score": <1-5>, "reason": "<brief>"},
    ...
  ],
  "dimension_scores_b": [
    {"dimension": "<dim>", "score": <1-5>, "reason": "<brief>"},
    ...
  ],
  "explanation": "<one sentence why the winner is better>"
}
"""


def _load_training_questions(
    path: Path, train_count: int
) -> list[dict]:
    """Load the first `train_count` docs from processed JSONL."""
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
                if len(docs) >= train_count:
                    break
    return docs


def _get_judge_client():
    """AzureOpenAI client for gpt-5.4 judge calls."""
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


def _generate_candidates(
    slm_client,
    azure_client,
    search_client,
    question: str,
    temps: tuple[float, float] = (0.3, 0.9),
) -> tuple[str, str]:
    """Generate two candidate SLM responses at different temperatures."""
    from src.rag.pipeline import _retrieve

    settings = get_settings()
    evidence = _retrieve(
        azure_client, search_client, question,
        top_k=settings.search_top_k,
    )

    parts = []
    for i, doc in enumerate(evidence, 1):
        parts.append(
            f"[{i}] Question: {doc['question']}\n"
            f"    Context: {doc['context'][:500]}\n"
            f"    Answer: {doc.get('long_answer', 'N/A')}\n"
            f"    Decision: {doc.get('final_decision', 'N/A')}"
        )
    ctx = "\n\n".join(parts)

    instructions = settings.yaml_config.get(
        "agent", {}
    ).get(
        "instructions",
        "You are a medical QA assistant. "
        "Use ONLY the provided evidence.",
    )

    responses = []
    for temp in temps:
        resp = slm_client.chat.completions.create(
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
            temperature=temp,
        )
        responses.append(resp.choices[0].message.content)

    context_summary = "\n\n".join(
        e["context"][:300] for e in evidence
    )
    return responses[0], responses[1], context_summary


def _call_judge(
    judge_client,
    question: str,
    gold_answer: str,
    response_a: str,
    response_b: str,
    max_retries: int = 5,
) -> dict:
    """Call gpt-5.4 to judge two candidates. Returns parsed verdict."""
    settings = get_settings()
    judge_cfg = settings.judge_config

    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"GOLD ANSWER:\n{gold_answer}\n\n"
        f"CANDIDATE A:\n{response_a}\n\n"
        f"CANDIDATE B:\n{response_b}"
    )

    for attempt in range(max_retries):
        try:
            resp = judge_client.chat.completions.create(
                model=settings.judge_chat_deployment,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=judge_cfg.get("temperature", 0.0),
                max_completion_tokens=judge_cfg.get(
                    "max_tokens", 2048
                ),
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(raw)
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {
                "winner": "A",
                "dimension_scores_a": [],
                "dimension_scores_b": [],
                "explanation": f"JSON parse failed: {raw[:200]}",
            }
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "RateLimit" in err_str:
                wait = min(60 * (attempt + 1), 120)
                console.print(
                    f"  [yellow]Rate limited, waiting "
                    f"{wait}s...[/yellow]"
                )
                time.sleep(wait)
                continue
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            raise


def _avg_score(dim_scores: list[dict]) -> float:
    """Compute average score across all rubric dimensions."""
    if not dim_scores:
        return 0.0
    return sum(d.get("score", 0) for d in dim_scores) / len(
        dim_scores
    )


app = typer.Typer()


@app.command()
def run(
    iteration: int = typer.Option(
        0, help="RL loop iteration number"
    ),
    samples: int = typer.Option(
        0,
        help="Number of training questions (0 = use config)",
    ),
) -> None:
    """Run AI judge to generate DPO preference pairs."""
    from src.rag.pipeline import (
        _get_azure_openai_client,
        _get_search_client,
        _get_slm_client,
    )

    settings = get_settings()

    if samples <= 0:
        samples = settings.rl_questions_per_iteration

    console.print("[bold]Phase 4: AI Judge[/bold]")
    console.print(
        f"  Judge: [cyan]{settings.judge_chat_deployment}"
        f"[/cyan] (Azure OpenAI)"
    )
    console.print(
        f"  SLM: [cyan]{settings.slm_ollama_model}[/cyan]"
        f" (Ollama)"
    )
    console.print(f"  Iteration: [cyan]{iteration}[/cyan]")
    console.print(f"  Questions: [cyan]{samples}[/cyan]")

    processed = (
        settings.processed_dir / settings.processed_file
    )
    train_questions = _load_training_questions(
        processed, settings.train_count
    )
    subset = train_questions[:samples]
    console.print(
        f"  Loaded {len(subset)} training questions"
    )

    azure_client = _get_azure_openai_client()
    slm_client = _get_slm_client()
    search_client = _get_search_client()
    judge_client = _get_judge_client()

    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = JUDGE_DIR / f"preferences_iter{iteration}.jsonl"

    pairs = []
    errors = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            "Judging", total=len(subset)
        )

        for doc in subset:
            question = doc["question"]
            gold = doc.get("long_answer", "")
            gold_decision = doc.get("final_decision", "")

            try:
                resp_a, resp_b, ctx = _generate_candidates(
                    slm_client,
                    azure_client,
                    search_client,
                    question,
                )

                verdict = _call_judge(
                    judge_client,
                    question,
                    gold,
                    resp_a,
                    resp_b,
                )

                winner = verdict.get("winner", "A")
                scores_a = verdict.get(
                    "dimension_scores_a", []
                )
                scores_b = verdict.get(
                    "dimension_scores_b", []
                )
                avg_a = _avg_score(scores_a)
                avg_b = _avg_score(scores_b)

                if winner == "A":
                    chosen, rejected = resp_a, resp_b
                    sc, sr = avg_a, avg_b
                else:
                    chosen, rejected = resp_b, resp_a
                    sc, sr = avg_b, avg_a

                pair = {
                    "iteration": iteration,
                    "question": question,
                    "context": ctx,
                    "gold_answer": gold,
                    "gold_decision": gold_decision,
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_label": winner,
                    "score_chosen": sc,
                    "score_rejected": sr,
                    "verdict": verdict,
                }
                pairs.append(pair)

            except Exception as exc:
                errors += 1
                console.print(
                    f"  [red]Error: {exc}[/red]"
                )

            progress.advance(task)

    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    console.print(
        "\n[bold green]Judge complete![/bold green]"
    )
    console.print(
        f"  Preference pairs: [cyan]{len(pairs)}[/cyan]"
    )
    console.print(f"  Errors: [cyan]{errors}[/cyan]")
    console.print(f"  Output: [dim]{out_path}[/dim]")

    if pairs:
        chosen_avg = sum(
            p["score_chosen"] for p in pairs
        ) / len(pairs)
        rejected_avg = sum(
            p["score_rejected"] for p in pairs
        ) / len(pairs)
        console.print(
            f"  Avg chosen score: "
            f"[cyan]{chosen_avg:.2f}[/cyan]"
        )
        console.print(
            f"  Avg rejected score: "
            f"[cyan]{rejected_avg:.2f}[/cyan]"
        )


def main() -> None:
    app()

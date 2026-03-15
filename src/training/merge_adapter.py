"""Phase 5: Merge LoRA adapter into base Phi-4-mini model.

After DPO training, the adapter weights live in
`data/training/iter{N}/adapter/`. This script merges them
back into the full model, saves to `data/training/iter{N}/merged/`,
and optionally registers the model with Ollama for local inference.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from src.config import PROJECT_ROOT, get_settings

console = Console()

TRAINING_DIR = PROJECT_ROOT / "data" / "training"


app = typer.Typer()


@app.command()
def run(
    iteration: int = typer.Option(
        0, help="RL loop iteration number"
    ),
    register_ollama: bool = typer.Option(
        False,
        help="Register merged model with Ollama",
    ),
) -> None:
    """Merge LoRA adapter weights into base model."""
    import torch
    from peft import PeftModel
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    settings = get_settings()

    adapter_dir = (
        TRAINING_DIR / f"iter{iteration}" / "adapter"
    )
    merged_dir = (
        TRAINING_DIR / f"iter{iteration}" / "merged"
    )

    console.print(
        "[bold]Phase 5c: Merge LoRA Adapter[/bold]"
    )
    console.print(
        f"  Iteration: [cyan]{iteration}[/cyan]"
    )

    if not adapter_dir.exists():
        console.print(
            f"[red]Adapter not found: {adapter_dir}[/red]"
        )
        console.print(
            "  Run: uv run train-dpo "
            f"--iteration {iteration}"
        )
        raise typer.Exit(1)

    if iteration > 0:
        prev_merged = (
            TRAINING_DIR
            / f"iter{iteration - 1}"
            / "merged"
        )
        base_path = (
            str(prev_merged)
            if prev_merged.exists()
            else settings.model_name
        )
    else:
        base_path = settings.model_name

    console.print(f"  Base model: [cyan]{base_path}[/cyan]")
    console.print(
        f"  Adapter: [dim]{adapter_dir}[/dim]"
    )

    console.print("  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path)

    console.print("  Loading adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_dir))

    console.print("  Merging...")
    model = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    console.print(
        "\n[bold green]Merge complete![/bold green]"
    )
    console.print(
        f"  Merged model: [dim]{merged_dir}[/dim]"
    )

    if register_ollama:
        _register_with_ollama(merged_dir, iteration)


def _register_with_ollama(
    merged_dir: Path, iteration: int
) -> None:
    """Create an Ollama Modelfile and register the merged model."""
    import subprocess

    settings = get_settings()
    model_tag = f"phi4-mini-dpo-iter{iteration}"

    modelfile = merged_dir / "Modelfile"
    modelfile.write_text(
        f'FROM {merged_dir}\n'
        f'PARAMETER temperature {settings.temperature}\n'
        f'PARAMETER num_predict {settings.max_new_tokens}\n'
        f'SYSTEM """{settings.yaml_config.get("agent", {}).get("instructions", "You are a medical QA assistant.")}"""\n'
    )

    console.print(
        f"  Registering with Ollama as "
        f"'{model_tag}'..."
    )

    try:
        result = subprocess.run(
            ["ollama", "create", model_tag, "-f",
             str(modelfile)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            console.print(
                f"  [green]Registered: {model_tag}[/green]"
            )
        else:
            console.print(
                f"  [yellow]Ollama registration failed: "
                f"{result.stderr[:200]}[/yellow]"
            )
            console.print(
                "  You can manually convert to GGUF "
                "and register later."
            )
    except FileNotFoundError:
        console.print(
            "  [yellow]Ollama not found — skipping "
            "registration[/yellow]"
        )


def main() -> None:
    app()

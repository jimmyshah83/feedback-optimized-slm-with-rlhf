"""Phase 5: TRL DPOTrainer with LoRA/QLoRA on Phi-4-mini.

Uses QLoRA (4-bit) when CUDA is available, plain LoRA with FP16
on MPS (Apple Silicon), and LoRA with FP32 on CPU as a fallback.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from src.config import PROJECT_ROOT, get_settings

console = Console()

TRAINING_DIR = PROJECT_ROOT / "data" / "training"


def _select_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_dataset(data_path: Path):
    from datasets import Dataset

    rows = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return Dataset.from_list(rows)


def _build_model_and_tokenizer(
    model_name: str, device: str, training_cfg: dict
):
    """Load model with optional quantisation and LoRA config."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    torch_dtype = torch.float32

    if device == "cuda":
        bits = training_cfg.get("quantization_bits", 4)
        if bits == 4:
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            torch_dtype = torch.bfloat16
    elif device == "mps":
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=False,
    )

    if device == "mps":
        model = model.to("mps")

    lora_cfg = LoraConfig(
        r=training_cfg.get("lora_rank", 16),
        lora_alpha=training_cfg.get("lora_alpha", 32),
        target_modules=training_cfg.get(
            "lora_target_modules", ["q_proj", "v_proj"]
        ),
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_cfg)
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in model.parameters())
    console.print(
        f"  Trainable: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )

    return model, tokenizer


app = typer.Typer()


@app.command()
def run(
    iteration: int = typer.Option(
        0, help="RL loop iteration number"
    ),
    model_path: str = typer.Option(
        "",
        help="Override model path (default: base or "
        "previous iteration merged)",
    ),
) -> None:
    """Run DPO fine-tuning with LoRA/QLoRA."""
    from trl import DPOConfig, DPOTrainer

    settings = get_settings()
    training_cfg = settings.training_config
    device = _select_device()

    console.print("[bold]Phase 5b: DPO Training[/bold]")
    console.print(f"  Device: [cyan]{device}[/cyan]")
    console.print(
        f"  Iteration: [cyan]{iteration}[/cyan]"
    )

    if not model_path:
        if iteration > 0:
            prev = (
                TRAINING_DIR
                / f"iter{iteration - 1}"
                / "merged"
            )
            if prev.exists():
                model_path = str(prev)
            else:
                model_path = settings.model_name
        else:
            model_path = settings.model_name
    console.print(f"  Model: [cyan]{model_path}[/cyan]")

    data_path = (
        TRAINING_DIR
        / f"iter{iteration}"
        / "dpo_data"
        / "train.jsonl"
    )
    if not data_path.exists():
        console.print(
            f"[red]DPO data not found: {data_path}[/red]"
        )
        console.print(
            "  Run: uv run prepare-dpo "
            f"--iteration {iteration}"
        )
        raise typer.Exit(1)

    dataset = _load_dataset(data_path)
    console.print(f"  Dataset: {len(dataset)} examples")

    model, tokenizer = _build_model_and_tokenizer(
        model_path, device, training_cfg
    )

    output_dir = (
        TRAINING_DIR / f"iter{iteration}" / "adapter"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = training_cfg.get(
        "per_device_batch_size", 4
    )
    if device == "mps":
        batch_size = min(batch_size, 2)

    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        beta=training_cfg.get("dpo_beta", 0.1),
        learning_rate=training_cfg.get(
            "learning_rate", 5e-5
        ),
        num_train_epochs=training_cfg.get(
            "num_epochs", 3
        ),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(
            1, 4 // batch_size
        ),
        logging_steps=1,
        save_strategy="epoch",
        remove_unused_columns=False,
        bf16=(device == "cuda"),
        fp16=(device == "mps"),
        max_length=1024,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    console.print("\n  [bold]Training...[/bold]")
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    console.print(
        "\n[bold green]DPO training complete![/bold green]"
    )
    console.print(
        f"  Adapter saved to [dim]{output_dir}[/dim]"
    )


def main() -> None:
    app()

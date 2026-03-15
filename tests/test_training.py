"""Tests for Phase 5: DPO training pipeline."""

from __future__ import annotations

import json
from pathlib import Path


class TestLoadPreferences:
    """Test the preference pair JSONL loader."""

    def test_loads_all_pairs(self, tmp_path: Path) -> None:
        from src.training.prepare_dpo_data import (
            load_preferences,
        )

        path = tmp_path / "prefs.jsonl"
        pairs = [
            {
                "question": f"Q{i}",
                "chosen": "good",
                "rejected": "bad",
                "context": "ctx",
            }
            for i in range(5)
        ]
        with open(path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")

        result = load_preferences(path)
        assert len(result) == 5

    def test_skips_blank_lines(
        self, tmp_path: Path
    ) -> None:
        from src.training.prepare_dpo_data import (
            load_preferences,
        )

        path = tmp_path / "prefs.jsonl"
        with open(path, "w") as f:
            f.write('{"question":"Q1","chosen":"a","rejected":"b"}\n')
            f.write("\n")
            f.write('{"question":"Q2","chosen":"c","rejected":"d"}\n')

        result = load_preferences(path)
        assert len(result) == 2


class TestFormatForDpo:
    """Test the DPO format conversion."""

    def test_prompt_includes_context(self) -> None:
        from src.training.prepare_dpo_data import (
            format_for_dpo,
        )

        pairs = [
            {
                "question": "Is X effective?",
                "chosen": "Yes, X is...",
                "rejected": "No...",
                "context": "Study shows X works",
            }
        ]
        formatted = format_for_dpo(pairs)
        assert len(formatted) == 1
        assert "Is X effective?" in formatted[0]["prompt"]
        assert "Study shows X works" in formatted[0]["prompt"]
        assert formatted[0]["chosen"] == "Yes, X is..."
        assert formatted[0]["rejected"] == "No..."

    def test_empty_input(self) -> None:
        from src.training.prepare_dpo_data import (
            format_for_dpo,
        )

        assert format_for_dpo([]) == []


class TestDeviceSelection:
    """Test device selection helper."""

    def test_returns_valid_device(self) -> None:
        from src.training.dpo_trainer import (
            _select_device,
        )

        device = _select_device()
        assert device in ("cuda", "mps", "cpu")


class TestLoadDataset:
    """Test dataset loading from JSONL."""

    def test_loads_jsonl(self, tmp_path: Path) -> None:
        from src.training.dpo_trainer import (
            _load_dataset,
        )

        path = tmp_path / "train.jsonl"
        rows = [
            {
                "prompt": "Q1?",
                "chosen": "good",
                "rejected": "bad",
            },
            {
                "prompt": "Q2?",
                "chosen": "great",
                "rejected": "poor",
            },
        ]
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        ds = _load_dataset(path)
        assert len(ds) == 2
        assert ds[0]["prompt"] == "Q1?"
        assert ds[1]["chosen"] == "great"


class TestTrainingConfig:
    """Verify training config properties."""

    def test_lora_target_modules(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        s = get_settings()
        cfg = s.training_config
        modules = cfg.get("lora_target_modules", [])
        assert "qkv_proj" in modules
        assert "o_proj" in modules

    def test_lora_rank(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        s = get_settings()
        cfg = s.training_config
        assert cfg.get("lora_rank") == 16

    def test_dpo_beta(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        s = get_settings()
        cfg = s.training_config
        assert cfg.get("dpo_beta") == 0.1

    def test_learning_rate(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        s = get_settings()
        cfg = s.training_config
        assert cfg.get("learning_rate") == 5e-5

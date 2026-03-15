"""Tests for Phase 6: RL loop orchestrator and comparison report."""

from __future__ import annotations

import json
from pathlib import Path


class TestHistorySerialization:
    """Test loop history save/load."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        history = [
            {
                "iteration": 0,
                "pubmedqa_accuracy": 0.65,
                "avg_relevance": 3.2,
            },
            {
                "iteration": 1,
                "pubmedqa_accuracy": 0.72,
                "avg_relevance": 3.8,
            },
        ]

        path = tmp_path / "history.json"
        with open(path, "w") as f:
            json.dump(history, f)

        with open(path) as f:
            loaded = json.load(f)

        assert len(loaded) == 2
        assert loaded[1]["pubmedqa_accuracy"] == 0.72

    def test_empty_history(self, tmp_path: Path) -> None:
        path = tmp_path / "history.json"
        with open(path, "w") as f:
            json.dump([], f)

        with open(path) as f:
            loaded = json.load(f)

        assert loaded == []


class TestCompare:
    """Test comparison report logic."""

    def test_prints_without_error(
        self, tmp_path: Path
    ) -> None:
        from src.evaluation.compare import LOOP_DIR

        history = [
            {
                "iteration": 0,
                "pubmedqa_accuracy": 0.5,
                "avg_groundedness": 3.0,
                "avg_relevance": 3.0,
                "avg_completeness": 3.0,
                "timestamp": "2026-01-01T00:00:00",
            },
            {
                "iteration": 1,
                "pubmedqa_accuracy": 0.7,
                "avg_groundedness": 4.0,
                "avg_relevance": 4.0,
                "avg_completeness": 4.0,
                "timestamp": "2026-01-02T00:00:00",
            },
        ]

        LOOP_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOOP_DIR / "history.json", "w") as f:
            json.dump(history, f)

        from src.evaluation.compare import run

        run()


class TestOrchestratorConfig:
    """Verify orchestrator config properties."""

    def test_rl_iterations(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        s = get_settings()
        assert s.rl_iterations == 3

    def test_rl_questions_per_iteration(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        s = get_settings()
        assert s.rl_questions_per_iteration == 800

    def test_convergence_threshold(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        s = get_settings()
        assert s.rl_convergence_threshold == 0.01

    def test_eval_count(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        s = get_settings()
        assert s.eval_count == 200

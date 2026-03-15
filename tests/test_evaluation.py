"""Tests for Phase 3: Evaluation (metrics + benchmark helpers)."""

from __future__ import annotations

import json
from pathlib import Path

class TestExtractDecision:
    """PubMedQA yes/no/maybe extraction from model output."""

    def test_explicit_decision_yes(self) -> None:
        from src.evaluation.metrics import extract_decision

        assert extract_decision("Decision: yes") == "yes"

    def test_explicit_decision_no(self) -> None:
        from src.evaluation.metrics import extract_decision

        assert extract_decision("Decision: no") == "no"

    def test_explicit_decision_maybe(self) -> None:
        from src.evaluation.metrics import extract_decision

        assert (
            extract_decision("**Decision:** maybe") == "maybe"
        )

    def test_trailing_yes(self) -> None:
        from src.evaluation.metrics import extract_decision

        text = "Based on the evidence, yes"
        assert extract_decision(text) == "yes"

    def test_trailing_no(self) -> None:
        from src.evaluation.metrics import extract_decision

        text = "The answer is no"
        assert extract_decision(text) == "no"

    def test_no_decision(self) -> None:
        from src.evaluation.metrics import extract_decision

        assert extract_decision("Some random text") is None

    def test_empty_string(self) -> None:
        from src.evaluation.metrics import extract_decision

        assert extract_decision("") is None


class TestPubmedqaAccuracy:
    """Aggregate accuracy computation."""

    def test_perfect_accuracy(self) -> None:
        from src.evaluation.metrics import pubmedqa_accuracy

        r = pubmedqa_accuracy(
            ["Decision: yes", "Decision: no", "Decision: maybe"],
            ["yes", "no", "maybe"],
        )
        assert r["accuracy"] == 1.0
        assert r["correct"] == 3
        assert r["parsed"] == 3
        assert r["unparsed"] == 0

    def test_partial_accuracy(self) -> None:
        from src.evaluation.metrics import pubmedqa_accuracy

        r = pubmedqa_accuracy(
            ["Decision: yes", "Decision: no"],
            ["yes", "yes"],
        )
        assert r["accuracy"] == 0.5
        assert r["correct"] == 1

    def test_all_unparsed(self) -> None:
        from src.evaluation.metrics import pubmedqa_accuracy

        r = pubmedqa_accuracy(
            ["gibberish", "nonsense"],
            ["yes", "no"],
        )
        assert r["accuracy"] == 0.0
        assert r["unparsed"] == 2

    def test_per_class_counts(self) -> None:
        from src.evaluation.metrics import pubmedqa_accuracy

        r = pubmedqa_accuracy(
            [
                "Decision: yes",
                "Decision: yes",
                "Decision: no",
            ],
            ["yes", "no", "no"],
        )
        assert r["per_class"]["yes"]["correct"] == 1
        assert r["per_class"]["yes"]["total"] == 1
        assert r["per_class"]["no"]["correct"] == 1
        assert r["per_class"]["no"]["total"] == 2


class TestLoadEvalQuestions:
    """Test the JSONL loading helper."""

    def test_loads_last_n(self, tmp_path: Path) -> None:
        from src.evaluation.benchmarks import (
            _load_eval_questions,
        )

        path = tmp_path / "data.jsonl"
        docs = [{"id": str(i), "question": f"Q{i}?"} for i in range(10)]
        with open(path, "w") as f:
            for d in docs:
                f.write(json.dumps(d) + "\n")

        result = _load_eval_questions(path, 3)
        assert len(result) == 3
        assert result[0]["id"] == "7"
        assert result[2]["id"] == "9"

    def test_fewer_than_requested(
        self, tmp_path: Path
    ) -> None:
        from src.evaluation.benchmarks import (
            _load_eval_questions,
        )

        path = tmp_path / "data.jsonl"
        with open(path, "w") as f:
            f.write('{"id": "1", "question": "Q?"}\n')

        result = _load_eval_questions(path, 200)
        assert len(result) == 1


class TestEvalConfig:
    """Verify evaluation-related config settings."""

    def test_eval_count(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        s = get_settings()
        assert s.eval_count == 200

    def test_judge_deployment(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        s = get_settings()
        assert s.judge_chat_deployment == "gpt-54"

    def test_evaluation_metrics(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        s = get_settings()
        metrics = s.yaml_config.get("evaluation", {}).get(
            "metrics", []
        )
        assert "groundedness" in metrics
        assert "relevance" in metrics

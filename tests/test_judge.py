"""Tests for Phase 4: AI Judge (preference pair generation)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock


class TestLoadTrainingQuestions:
    """Test the JSONL loading helper for training data."""

    def test_loads_first_n(self, tmp_path: Path) -> None:
        from src.judge.ai_judge import _load_training_questions

        path = tmp_path / "data.jsonl"
        docs = [
            {"id": str(i), "question": f"Q{i}?"}
            for i in range(10)
        ]
        with open(path, "w") as f:
            for d in docs:
                f.write(json.dumps(d) + "\n")

        result = _load_training_questions(path, 3)
        assert len(result) == 3
        assert result[0]["id"] == "0"
        assert result[2]["id"] == "2"

    def test_fewer_than_requested(
        self, tmp_path: Path
    ) -> None:
        from src.judge.ai_judge import _load_training_questions

        path = tmp_path / "data.jsonl"
        with open(path, "w") as f:
            f.write('{"id": "1", "question": "Q?"}\n')

        result = _load_training_questions(path, 800)
        assert len(result) == 1


class TestAvgScore:
    """Test score averaging helper."""

    def test_average_of_scores(self) -> None:
        from src.judge.ai_judge import _avg_score

        scores = [
            {"dimension": "a", "score": 4, "reason": ""},
            {"dimension": "b", "score": 2, "reason": ""},
        ]
        assert _avg_score(scores) == 3.0

    def test_empty_scores(self) -> None:
        from src.judge.ai_judge import _avg_score

        assert _avg_score([]) == 0.0

    def test_single_score(self) -> None:
        from src.judge.ai_judge import _avg_score

        assert _avg_score(
            [{"dimension": "a", "score": 5, "reason": ""}]
        ) == 5.0


class TestCallJudge:
    """Test the gpt-5.4 judge call logic."""

    def test_parses_valid_json(self) -> None:
        from src.judge.ai_judge import _call_judge

        verdict = {
            "winner": "B",
            "dimension_scores_a": [
                {
                    "dimension": "accuracy",
                    "score": 3,
                    "reason": "ok",
                }
            ],
            "dimension_scores_b": [
                {
                    "dimension": "accuracy",
                    "score": 5,
                    "reason": "great",
                }
            ],
            "explanation": "B is better",
        }

        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(verdict)
                )
            )
        ]
        mock_client.chat.completions.create.return_value = (
            mock_resp
        )

        result = _call_judge(
            mock_client, "Q?", "gold", "resp_a", "resp_b"
        )
        assert result["winner"] == "B"
        assert len(result["dimension_scores_b"]) == 1

    def test_handles_markdown_wrapped_json(self) -> None:
        from src.judge.ai_judge import _call_judge

        verdict = {
            "winner": "A",
            "dimension_scores_a": [],
            "dimension_scores_b": [],
            "explanation": "A wins",
        }

        mock_client = MagicMock()
        raw = f"```json\n{json.dumps(verdict)}\n```"
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(message=MagicMock(content=raw))
        ]
        mock_client.chat.completions.create.return_value = (
            mock_resp
        )

        result = _call_judge(
            mock_client, "Q?", "gold", "a", "b"
        )
        assert result["winner"] == "A"


class TestGenerateCandidates:
    """Test two-temperature candidate generation."""

    def test_returns_two_responses_and_context(
        self,
    ) -> None:
        from src.judge.ai_judge import _generate_candidates

        mock_slm = MagicMock()
        call_count = [0]

        def _make_response(*args, **kwargs):
            call_count[0] += 1
            return MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content=f"Response {call_count[0]}"
                        )
                    )
                ]
            )

        mock_slm.chat.completions.create.side_effect = (
            _make_response
        )

        mock_azure = MagicMock()
        mock_azure.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 3072)
        ]

        doc = {
            "question": "Q?",
            "context": "C",
            "long_answer": "A",
            "final_decision": "yes",
        }
        mock_result = MagicMock()
        mock_result.__getitem__ = lambda s, k: doc[k]
        mock_result.get = lambda k, d="": doc.get(k, d)

        mock_search = MagicMock()
        mock_search.search.return_value = [mock_result]

        a, b, ctx = _generate_candidates(
            mock_slm, mock_azure, mock_search, "Q?"
        )

        assert a == "Response 1"
        assert b == "Response 2"
        assert isinstance(ctx, str)
        assert mock_slm.chat.completions.create.call_count == 2


class TestModels:
    """Test Pydantic data models."""

    def test_preference_pair_creation(self) -> None:
        from src.judge.models import PreferencePair

        p = PreferencePair(
            iteration=0,
            question="Q?",
            context="C",
            gold_answer="gold",
            gold_decision="yes",
            chosen="good",
            rejected="bad",
            chosen_label="A",
            score_chosen=4.5,
            score_rejected=2.0,
        )
        assert p.chosen == "good"
        assert p.score_chosen > p.score_rejected

    def test_dimension_score_validation(self) -> None:
        from src.judge.models import DimensionScore

        s = DimensionScore(
            dimension="accuracy", score=5, reason="perfect"
        )
        assert s.score == 5

    def test_judge_verdict(self) -> None:
        from src.judge.models import (
            DimensionScore,
            JudgeVerdict,
        )

        v = JudgeVerdict(
            winner="A",
            dimension_scores_a=[
                DimensionScore(
                    dimension="acc", score=5, reason="ok"
                )
            ],
            dimension_scores_b=[
                DimensionScore(
                    dimension="acc", score=3, reason="ok"
                )
            ],
            explanation="A is better",
        )
        assert v.winner == "A"


class TestJudgeConfig:
    """Verify judge config properties."""

    def test_judge_deployment(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        s = get_settings()
        assert s.judge_chat_deployment == "gpt-54"

    def test_judge_temperature(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        s = get_settings()
        assert s.judge_temperature == 0.0

    def test_judge_max_tokens(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        s = get_settings()
        assert s.judge_max_tokens == 2048

    def test_judge_rubric_dimensions(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        s = get_settings()
        dims = s.judge_rubric_dimensions
        assert "medical_accuracy" in dims
        assert "faithfulness" in dims
        assert len(dims) == 4

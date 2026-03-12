"""Tests for Phase 1: data download and preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.preprocess import _build_document


class TestBuildDocument:
    """Unit tests for the document transformation function."""

    def test_produces_required_fields(self, sample_raw_record: dict) -> None:
        doc = _build_document(sample_raw_record)
        required = {"pubid", "question", "context", "long_answer", "final_decision", "meshes"}
        assert required.issubset(doc.keys())

    def test_pubid_is_string(self, sample_raw_record: dict) -> None:
        doc = _build_document(sample_raw_record)
        assert isinstance(doc["pubid"], str)

    def test_context_combines_abstracts_with_labels(self, sample_raw_record: dict) -> None:
        doc = _build_document(sample_raw_record)
        assert "[BACKGROUND]" in doc["context"]
        assert "[RESULTS]" in doc["context"]
        assert "cardioprotective" in doc["context"]

    def test_preserves_question(self, sample_raw_record: dict) -> None:
        doc = _build_document(sample_raw_record)
        assert doc["question"] == sample_raw_record["question"]

    def test_preserves_final_decision(self, sample_raw_record: dict) -> None:
        doc = _build_document(sample_raw_record)
        assert doc["final_decision"] == "yes"

    def test_preserves_meshes(self, sample_raw_record: dict) -> None:
        doc = _build_document(sample_raw_record)
        assert "Aspirin" in doc["meshes"]

    def test_handles_missing_labels_gracefully(self) -> None:
        record = {
            "pubid": 1,
            "question": "Test?",
            "context": {
                "contexts": ["Abstract text here."],
                "labels": [],
                "meshes": [],
            },
            "long_answer": "Answer.",
            "final_decision": "no",
        }
        doc = _build_document(record)
        assert "Abstract text here." in doc["context"]

    def test_handles_plain_string_context(self) -> None:
        record = {
            "pubid": 2,
            "question": "Test?",
            "context": "Just a plain string context.",
            "long_answer": "Answer.",
            "final_decision": "maybe",
        }
        doc = _build_document(record)
        assert "plain string context" in doc["context"]


class TestProcessedFileFormat:
    """Validate the JSONL output structure."""

    def test_jsonl_is_readable(self, tmp_processed_file: Path) -> None:
        with open(tmp_processed_file) as f:
            lines = f.readlines()
        assert len(lines) == 5

    def test_each_line_is_valid_json(self, tmp_processed_file: Path) -> None:
        with open(tmp_processed_file) as f:
            for line in f:
                doc = json.loads(line)
                assert "question" in doc
                assert "context" in doc
                assert "final_decision" in doc

    def test_no_empty_questions(self, tmp_processed_file: Path) -> None:
        with open(tmp_processed_file) as f:
            for line in f:
                doc = json.loads(line)
                assert doc["question"].strip()


class TestConfigLoading:
    """Verify settings load without errors."""

    def test_settings_load(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()
        assert settings.dataset_name == "qiaojin/PubMedQA"
        assert settings.dataset_split == "pqa_labeled"

    def test_processed_dir_is_path(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings.processed_dir, Path)

    def test_model_name_default(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()
        assert "Phi-4" in settings.model_name or "phi-4" in settings.model_name.lower()

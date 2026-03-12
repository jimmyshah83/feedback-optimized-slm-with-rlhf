"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def sample_raw_record() -> dict:
    """A single raw PubMedQA record matching the HuggingFace dataset schema."""
    return {
        "pubid": 12345678,
        "question": "Does aspirin reduce the risk of cardiovascular events?",
        "context": {
            "contexts": [
                "Aspirin has been widely studied for its cardioprotective effects.",
                "A meta-analysis of randomized trials showed a significant reduction in major cardiovascular events.",
            ],
            "labels": ["BACKGROUND", "RESULTS"],
            "meshes": ["Aspirin", "Cardiovascular Diseases"],
        },
        "long_answer": "Yes, aspirin has been shown to reduce the risk of major cardiovascular events in multiple randomized controlled trials.",
        "final_decision": "yes",
    }


@pytest.fixture()
def sample_processed_doc() -> dict:
    """A preprocessed document as produced by preprocess._build_document."""
    return {
        "pubid": "12345678",
        "question": "Does aspirin reduce the risk of cardiovascular events?",
        "context": (
            "[BACKGROUND] Aspirin has been widely studied for its cardioprotective effects.\n\n"
            "[RESULTS] A meta-analysis of randomized trials showed a significant reduction in major cardiovascular events."
        ),
        "long_answer": "Yes, aspirin has been shown to reduce the risk of major cardiovascular events in multiple randomized controlled trials.",
        "final_decision": "yes",
        "meshes": ["Aspirin", "Cardiovascular Diseases"],
    }


@pytest.fixture()
def tmp_processed_file(tmp_path: Path, sample_processed_doc: dict) -> Path:
    """Write a small JSONL file and return its path."""
    out = tmp_path / "test_processed.jsonl"
    with open(out, "w") as f:
        for _ in range(5):
            f.write(json.dumps(sample_processed_doc) + "\n")
    return out

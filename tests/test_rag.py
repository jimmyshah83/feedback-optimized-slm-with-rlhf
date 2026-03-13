"""Tests for Phase 2: RAG pipeline (index + retrieve + generate via SLM)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestIndexDocuments:
    """Unit tests for document loading and embedding helpers."""

    def test_load_documents(self, tmp_path: Path) -> None:
        from src.data.index_documents import _load_documents

        out = tmp_path / "test.jsonl"
        docs = [
            {"pubid": "1", "question": "Q1?", "context": "C1"},
            {"pubid": "2", "question": "Q2?", "context": "C2"},
        ]
        with open(out, "w") as f:
            for doc in docs:
                f.write(json.dumps(doc) + "\n")

        loaded = _load_documents(out)
        assert len(loaded) == 2
        assert loaded[0]["pubid"] == "1"
        assert loaded[1]["question"] == "Q2?"

    def test_load_documents_skips_blank_lines(
        self, tmp_path: Path
    ) -> None:
        from src.data.index_documents import _load_documents

        out = tmp_path / "test.jsonl"
        with open(out, "w") as f:
            f.write('{"pubid": "1", "question": "Q?"}\n')
            f.write("\n")
            f.write('{"pubid": "2", "question": "Q2?"}\n')

        loaded = _load_documents(out)
        assert len(loaded) == 2

    def test_get_embeddings_returns_list(self) -> None:
        from src.data.index_documents import _get_embeddings

        mock_client = MagicMock()
        mock_item = MagicMock()
        mock_item.embedding = [0.1, 0.2, 0.3]
        mock_client.embeddings.create.return_value.data = [
            mock_item, mock_item
        ]

        result = _get_embeddings(
            mock_client, ["text1", "text2"], "model"
        )
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]


class TestRagPipeline:
    """Unit tests for RAG retrieve + generate helpers."""

    def test_retrieve_calls_search(self) -> None:
        """_retrieve embeds the question and runs hybrid search."""
        from src.rag.pipeline import _retrieve

        mock_azure = MagicMock()
        mock_azure.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 3072)
        ]

        row = {
            "question": "Q1?",
            "context": "C1",
            "long_answer": "A1",
            "final_decision": "yes",
        }
        mock_result = MagicMock()
        mock_result.__getitem__ = lambda s, k: row[k]
        mock_result.get = lambda k, d="": row.get(k, d)

        mock_search = MagicMock()
        mock_search.search.return_value = [mock_result]

        docs = _retrieve(mock_azure, mock_search, "test?", 3)

        assert len(docs) == 1
        assert docs[0]["question"] == "Q1?"
        assert docs[0]["context"] == "C1"
        mock_azure.embeddings.create.assert_called_once()
        mock_search.search.assert_called_once()

    def test_generate_uses_slm_client(self) -> None:
        """_generate sends evidence to the local SLM."""
        from src.rag.pipeline import _generate

        mock_slm = MagicMock()
        mock_slm.chat.completions.create.return_value = (
            MagicMock(choices=[
                MagicMock(
                    message=MagicMock(content="Yes.")
                )
            ])
        )

        evidence = [{
            "question": "Q1?",
            "context": "C1",
            "long_answer": "A1",
            "final_decision": "yes",
        }]
        result = _generate(mock_slm, "Does it work?", evidence)
        assert result == "Yes."

        call = mock_slm.chat.completions.create.call_args
        assert call.kwargs["model"] == "phi4-mini"

    def test_generate_includes_all_evidence(self) -> None:
        """All evidence docs appear in the system prompt."""
        from src.rag.pipeline import _generate

        mock_slm = MagicMock()
        mock_slm.chat.completions.create.return_value = (
            MagicMock(choices=[
                MagicMock(
                    message=MagicMock(content="Answer")
                )
            ])
        )

        evidence = [
            {
                "question": f"Q{i}?",
                "context": f"C{i}",
                "long_answer": f"A{i}",
                "final_decision": "yes",
            }
            for i in range(3)
        ]
        _generate(mock_slm, "Test?", evidence)

        call = mock_slm.chat.completions.create.call_args
        sys_msg = call.kwargs["messages"][0]["content"]
        for i in range(3):
            assert f"Q{i}?" in sys_msg
            assert f"C{i}" in sys_msg

    def test_ensure_agent_returns_existing(self) -> None:
        """When agent exists, return its id."""
        from src.rag.pipeline import _ensure_agent_definition

        mock_client = MagicMock()
        mock_agent = MagicMock()
        mock_agent.versions.latest.id = "pubmedqa-rag:2"
        mock_client.agents.get.return_value = mock_agent

        settings = MagicMock()
        settings.agent_name = "pubmedqa-rag"

        result = _ensure_agent_definition(
            mock_client, settings
        )
        assert result == "pubmedqa-rag:2"
        mock_client.agents.create_version.assert_not_called()


class TestConfigPhase2:
    """Verify Phase 2 config properties load correctly."""

    def test_agent_config(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()
        assert settings.agent_name == "pubmedqa-rag"
        assert settings.agent_model == "phi4-mini"

    def test_slm_config(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()
        assert settings.slm_ollama_model == "phi4-mini"
        assert "11434" in settings.slm_ollama_base_url

    def test_embedding_config(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()
        assert settings.embedding_deployment == (
            "text-embedding-3-large"
        )
        assert settings.embedding_dimensions == 3072

    def test_search_config(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()
        assert settings.search_top_k == 5
        assert settings.search_semantic_config == (
            "pubmedqa-semantic"
        )

    def test_judge_uses_gpt54(self) -> None:
        """gpt-5.4 is the judge, not the RAG model."""
        from src.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()
        assert settings.judge_chat_deployment == "gpt-54"
        assert settings.agent_model != "gpt-54"

    def test_no_openai_settings(self) -> None:
        """AzureOpenAISettings was removed."""
        from src import config

        assert not hasattr(config, "AzureOpenAISettings")

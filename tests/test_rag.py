"""Tests for Phase 2: RAG pipeline (index + agent)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestIndexDocuments:
    """Unit tests for document loading and index creation helpers."""

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

    def test_load_documents_skips_blank_lines(self, tmp_path: Path) -> None:
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
        mock_client.embeddings.create.return_value.data = [mock_item, mock_item]

        result = _get_embeddings(mock_client, ["text1", "text2"], "model")
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]


class TestRagPipeline:
    """Unit tests for RAG pipeline helpers."""

    def test_build_search_tool(self) -> None:
        from src.config import get_settings
        from src.rag.pipeline import _build_search_tool

        get_settings.cache_clear()
        settings = get_settings()
        tool = _build_search_tool("test-connection-id", settings)

        assert tool.azure_ai_search is not None
        indexes = tool.azure_ai_search.indexes
        assert len(indexes) == 1
        assert indexes[0].project_connection_id == "test-connection-id"
        assert indexes[0].index_name == settings.azure_search.index
        assert indexes[0].top_k == settings.search_top_k

    def test_query_agent_extracts_text(self) -> None:
        from src.rag.pipeline import _query_agent

        mock_client = MagicMock()
        mock_thread = MagicMock()
        mock_thread.id = "thread-123"
        mock_client.beta.threads.create.return_value = mock_thread

        mock_run = MagicMock()
        mock_run.status = "completed"
        mock_client.beta.threads.runs.create_and_poll.return_value = mock_run

        mock_text_block = MagicMock()
        mock_text_block.text.value = "The answer is yes."
        mock_msg = MagicMock()
        mock_msg.role = "assistant"
        mock_msg.content = [mock_text_block]
        mock_client.beta.threads.messages.list.return_value.data = [mock_msg]

        result = _query_agent(mock_client, "agent-id", "Does it work?")
        assert result == "The answer is yes."

    def test_query_agent_raises_on_failure(self) -> None:
        from src.rag.pipeline import _query_agent

        mock_client = MagicMock()
        mock_thread = MagicMock()
        mock_thread.id = "thread-456"
        mock_client.beta.threads.create.return_value = mock_thread

        mock_run = MagicMock()
        mock_run.status = "failed"
        mock_run.last_error = "Rate limit exceeded"
        mock_client.beta.threads.runs.create_and_poll.return_value = mock_run

        with pytest.raises(RuntimeError, match="Agent run failed"):
            _query_agent(mock_client, "agent-id", "Will this fail?")


class TestConfigPhase2:
    """Verify Phase 2 config properties load correctly."""

    def test_agent_config(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()
        assert settings.agent_name == "pubmedqa-rag"
        assert settings.agent_model == "gpt-54"

    def test_embedding_config(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()
        assert settings.embedding_deployment == "text-embedding-3-large"
        assert settings.embedding_dimensions == 3072

    def test_search_config(self) -> None:
        from src.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()
        assert settings.search_top_k == 5
        assert settings.search_semantic_config == "pubmedqa-semantic"

    def test_no_openai_settings(self) -> None:
        """AzureOpenAISettings was removed — verify it doesn't exist."""
        from src import config

        assert not hasattr(config, "AzureOpenAISettings")

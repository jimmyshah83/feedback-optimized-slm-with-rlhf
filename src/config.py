from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


class AzureAIProjectSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AZURE_AI_")

    project_endpoint: str = ""


class AzureSearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AZURE_SEARCH_")

    endpoint: str = ""
    index: str = "pubmedqa-index"


class AzureCosmosSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AZURE_COSMOS_")

    endpoint: str = ""
    key: str = ""
    database: str = "rag-feedback"
    container: str = "judge-results"


class AzureMLSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AZURE_ML_")

    subscription_id: str = ""
    resource_group: str = ""
    workspace_name: str = ""


class AzureStorageSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AZURE_STORAGE_")

    connection_string: str = ""
    container: str = "model-artifacts"


class Settings(BaseSettings):
    """Central configuration loaded from config/settings.yaml + environment variables."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    yaml_config: dict[str, Any] = Field(default_factory=dict)

    azure_ai_project: AzureAIProjectSettings = Field(
        default_factory=AzureAIProjectSettings
    )
    azure_search: AzureSearchSettings = Field(default_factory=AzureSearchSettings)
    azure_cosmos: AzureCosmosSettings = Field(default_factory=AzureCosmosSettings)
    azure_ml: AzureMLSettings = Field(default_factory=AzureMLSettings)
    azure_storage: AzureStorageSettings = Field(default_factory=AzureStorageSettings)

    # --- Model ---

    @property
    def model_name(self) -> str:
        return self.yaml_config.get("model", {}).get("name", "microsoft/Phi-4-mini-instruct")

    @property
    def max_new_tokens(self) -> int:
        return self.yaml_config.get("model", {}).get("max_new_tokens", 512)

    @property
    def temperature(self) -> float:
        return self.yaml_config.get("model", {}).get("temperature", 0.3)

    # --- Embedding ---

    @property
    def embedding_deployment(self) -> str:
        return self.yaml_config.get("embedding", {}).get("deployment", "text-embedding-3-large")

    @property
    def embedding_dimensions(self) -> int:
        return self.yaml_config.get("embedding", {}).get("dimensions", 3072)

    # --- Search ---

    @property
    def search_top_k(self) -> int:
        return self.yaml_config.get("search", {}).get("top_k", 5)

    @property
    def search_semantic_config(self) -> str:
        return self.yaml_config.get("search", {}).get("semantic_config", "pubmedqa-semantic")

    # --- Data ---

    @property
    def dataset_name(self) -> str:
        return self.yaml_config.get("data", {}).get("dataset_name", "qiaojin/PubMedQA")

    @property
    def dataset_split(self) -> str:
        return self.yaml_config.get("data", {}).get("dataset_split", "pqa_labeled")

    @property
    def raw_dir(self) -> Path:
        return PROJECT_ROOT / self.yaml_config.get("data", {}).get("raw_dir", "data/raw")

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / self.yaml_config.get("data", {}).get("processed_dir", "data/processed")

    @property
    def processed_file(self) -> str:
        return self.yaml_config.get("data", {}).get("processed_file", "pubmedqa_processed.jsonl")

    @property
    def train_count(self) -> int:
        return self.yaml_config.get("data", {}).get("train_count", 800)

    @property
    def eval_count(self) -> int:
        return self.yaml_config.get("data", {}).get("eval_count", 200)

    # --- Evaluation ---

    @property
    def eval_samples(self) -> int:
        return self.yaml_config.get("evaluation", {}).get("eval_samples", 200)

    @property
    def eval_metrics(self) -> list[str]:
        return self.yaml_config.get("evaluation", {}).get(
            "metrics", ["groundedness", "relevance", "response_completeness"]
        )

    # --- SLM (local Ollama) ---

    @property
    def slm_ollama_base_url(self) -> str:
        return self.yaml_config.get("slm", {}).get(
            "ollama_base_url", "http://localhost:11434/v1"
        )

    @property
    def slm_ollama_model(self) -> str:
        return self.yaml_config.get("slm", {}).get(
            "ollama_model", "phi4-mini"
        )

    # --- Agent ---

    @property
    def agent_name(self) -> str:
        return self.yaml_config.get("agent", {}).get("name", "pubmedqa-rag")

    @property
    def agent_model(self) -> str:
        return self.yaml_config.get("agent", {}).get("model", "phi4-mini")

    # --- Judge ---

    @property
    def judge_config(self) -> dict[str, Any]:
        return self.yaml_config.get("judge", {})

    @property
    def judge_chat_deployment(self) -> str:
        return self.judge_config.get("chat_deployment", "gpt-54")

    @property
    def judge_rubric_dimensions(self) -> list[str]:
        return self.judge_config.get(
            "rubric_dimensions",
            ["medical_accuracy", "faithfulness", "completeness", "clarity"],
        )

    @property
    def judge_temperature(self) -> float:
        return self.judge_config.get("temperature", 0.0)

    @property
    def judge_max_tokens(self) -> int:
        return self.judge_config.get("max_tokens", 2048)

    # --- RL Loop ---

    @property
    def rl_iterations(self) -> int:
        return self.yaml_config.get("rl_loop", {}).get("iterations", 3)

    @property
    def rl_questions_per_iteration(self) -> int:
        return self.yaml_config.get("rl_loop", {}).get("questions_per_iteration", 800)

    @property
    def rl_convergence_threshold(self) -> float:
        return self.yaml_config.get("rl_loop", {}).get("convergence_threshold", 0.01)

    # --- Training ---

    @property
    def training_config(self) -> dict[str, Any]:
        return self.yaml_config.get("training", {})


@lru_cache
def get_settings() -> Settings:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    yaml_path = CONFIG_DIR / "settings.yaml"
    yaml_data = _load_yaml(yaml_path) if yaml_path.exists() else {}
    return Settings(yaml_config=yaml_data)

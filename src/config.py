from __future__ import annotations

import os
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


class AzureSearchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AZURE_SEARCH_")

    endpoint: str = ""
    key: str = ""
    index: str = "pubmedqa-index"


class AzureOpenAISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AZURE_OPENAI_")

    endpoint: str = ""
    api_key: str = ""
    embedding_deployment: str = "text-embedding-3-small"
    api_version: str = "2024-10-21"


class AzureCosmosSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AZURE_COSMOS_")

    endpoint: str = ""
    key: str = ""
    database: str = "rag-feedback"
    container: str = "feedback"


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

    azure_search: AzureSearchSettings = Field(default_factory=AzureSearchSettings)
    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)
    azure_cosmos: AzureCosmosSettings = Field(default_factory=AzureCosmosSettings)
    azure_ml: AzureMLSettings = Field(default_factory=AzureMLSettings)
    azure_storage: AzureStorageSettings = Field(default_factory=AzureStorageSettings)

    @property
    def model_name(self) -> str:
        return self.yaml_config.get("model", {}).get("name", "microsoft/Phi-4-mini-instruct")

    @property
    def max_new_tokens(self) -> int:
        return self.yaml_config.get("model", {}).get("max_new_tokens", 512)

    @property
    def temperature(self) -> float:
        return self.yaml_config.get("model", {}).get("temperature", 0.3)

    @property
    def embedding_dimensions(self) -> int:
        return self.yaml_config.get("embedding", {}).get("dimensions", 1536)

    @property
    def search_top_k(self) -> int:
        return self.yaml_config.get("search", {}).get("top_k", 5)

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
    def eval_samples(self) -> int:
        return self.yaml_config.get("evaluation", {}).get("samples", 50)

    @property
    def training_config(self) -> dict[str, Any]:
        return self.yaml_config.get("training", {})


@lru_cache
def get_settings() -> Settings:
    yaml_path = CONFIG_DIR / "settings.yaml"
    yaml_data = _load_yaml(yaml_path) if yaml_path.exists() else {}
    return Settings(yaml_config=yaml_data)

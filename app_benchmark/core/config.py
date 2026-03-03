"""
Configuration and model registry for AI Benchmark app.
"""
import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class ModelConfig:
    name: str
    deployment: str
    provider: str
    description: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    default_selected: bool = True


COMPETITOR_MODELS = [
    ModelConfig(
        name="GPT-4.1-mini",
        deployment="gpt-4.1-mini",
        provider="OpenAI",
        description="Fast and efficient baseline model",
        input_cost_per_1m=0.40,
        output_cost_per_1m=1.60,
        default_selected=True,
    ),
    ModelConfig(
        name="o3-mini",
        deployment="o3-mini",
        provider="OpenAI",
        description="Reasoning-focused model",
        input_cost_per_1m=1.10,
        output_cost_per_1m=4.40,
        default_selected=True,
    ),
    ModelConfig(
        name="Phi-4",
        deployment="Phi-4",
        provider="Microsoft",
        description="Lightweight but capable small model",
        input_cost_per_1m=0.07,
        output_cost_per_1m=0.28,
        default_selected=True,
    ),
]

EVALUATOR_MODEL = "gpt-4o"


@dataclass
class Settings:
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str
    azure_ai_search_endpoint: str
    azure_search_api_key: str
    azure_search_index_name: str
    vector_field_name: str
    semantic_configuration_name: str
    search_top_k: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        azure_openai_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "https://aisa-rf7yqag4sc6qk.openai.azure.com/"),
        azure_openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        azure_openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_ai_search_endpoint=os.environ.get("AZURE_AI_SEARCH_ENDPOINT", "https://srch-rf7yqag4sc6qk.search.windows.net"),
        azure_search_api_key=os.environ.get("AZURE_SEARCH_API_KEY", ""),
        azure_search_index_name=os.environ.get("AZURE_SEARCH_INDEX_NAME", "documents"),
        vector_field_name=os.environ.get("VECTOR_FIELD_NAME", "embedding"),
        semantic_configuration_name=os.environ.get("SEMANTIC_CONFIGURATION_NAME", "default-semantic"),
        search_top_k=int(os.environ.get("SEARCH_TOP_K", "5")),
    )

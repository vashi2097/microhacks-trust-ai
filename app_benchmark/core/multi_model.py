"""
Runs multiple LLM models in parallel on the same question.
Captures latency and token usage per model.
o3-mini uses max_completion_tokens instead of max_tokens and no temperature.
"""
import time
import logging
import concurrent.futures
from dataclasses import dataclass
from openai import AzureOpenAI
from .config import Settings, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    model_name: str
    answer: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    content_filtered: bool = False
    error: str = ""

    def cost(self, model_config: ModelConfig) -> float:
        input_cost = (self.input_tokens / 1_000_000) * model_config.input_cost_per_1m
        output_cost = (self.output_tokens / 1_000_000) * model_config.output_cost_per_1m
        return round(input_cost + output_cost, 6)


# Models that use reasoning API format (no temperature, max_completion_tokens)
REASONING_MODELS = {"o3-mini", "o1-mini", "o1"}


class MultiModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )

    def _call_model(self, model_config: ModelConfig, system_prompt: str, question: str) -> ModelResponse:
        start = time.time()
        try:
            is_reasoning = model_config.deployment in REASONING_MODELS
            kwargs = {
                "model": model_config.deployment,
                "messages": [
                    {"role": "user", "content": f"{system_prompt}\n\nQuestion: {question}"},
                ],
            }
            if is_reasoning:
                kwargs["max_completion_tokens"] = 1024
            else:
                kwargs["messages"] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ]
                kwargs["max_tokens"] = 1024
                kwargs["temperature"] = 0.3

            response = self.client.chat.completions.create(**kwargs)
            latency = round(time.time() - start, 2)
            return ModelResponse(
                model_name=model_config.name,
                answer=response.choices[0].message.content or "",
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                latency_seconds=latency,
            )
        except Exception as e:
            latency = round(time.time() - start, 2)
            error_str = str(e)
            if "content_filter" in error_str or "content management policy" in error_str:
                return ModelResponse(model_name=model_config.name, answer="I cannot answer that question.", content_filtered=True, latency_seconds=latency)
            logger.error(f"Error with {model_config.name}: {e}")
            return ModelResponse(model_name=model_config.name, error=error_str, latency_seconds=latency)

    def run_parallel(self, selected_models: list, system_prompt: str, question: str) -> list:
        responses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_models)) as executor:
            futures = {
                executor.submit(self._call_model, model, system_prompt, question): model
                for model in selected_models
            }
            for future in concurrent.futures.as_completed(futures):
                try:
                    responses.append(future.result())
                except Exception as e:
                    model = futures[future]
                    responses.append(ModelResponse(model_name=model.name, error=str(e)))

        order = {m.name: i for i, m in enumerate(selected_models)}
        responses.sort(key=lambda x: order.get(x.model_name, 99))
        return responses

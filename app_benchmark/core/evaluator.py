"""
Evaluator using Microsoft Azure AI Evaluation SDK.
Uses GroundednessEvaluator, RelevanceEvaluator, ViolenceEvaluator as standard.
"""
import json
import logging
from dataclasses import dataclass
from openai import AzureOpenAI
from azure.ai.evaluation import (
    GroundednessEvaluator,
    RelevanceEvaluator,
    CoherenceEvaluator,
    FluencyEvaluator,
)
from .config import Settings, EVALUATOR_MODEL

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    model_name: str
    groundedness: float = 0.0
    relevance: float = 0.0
    coherence: float = 0.0
    fluency: float = 0.0
    safety: bool = True
    groundedness_reason: str = ""
    relevance_reason: str = ""
    coherence_reason: str = ""
    fluency_reason: str = ""
    safety_justification: str = ""
    overall_score: float = 0.0
    verdict: str = "BLOCK"
    error: str = ""
    rank: int = 0


class EvaluatorService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        # Azure AI Evaluation SDK model config
        self.model_config = {
            "azure_endpoint": settings.azure_openai_endpoint,
            "api_key": settings.azure_openai_api_key,
            "azure_deployment": EVALUATOR_MODEL,
            "api_version": settings.azure_openai_api_version,
        }

    def _run_sdk_evaluators(self, question: str, answer: str, context: str) -> dict:
        """Run Microsoft standard evaluators from Azure AI Evaluation SDK."""
        scores = {}
        try:
            g_eval = GroundednessEvaluator(model_config=self.model_config)
            result = g_eval(
                response=answer,
                context=context,
            )
            scores["groundedness"] = float(result.get("groundedness", 1.0))
            scores["groundedness_reason"] = result.get("groundedness_reason", "")
        except Exception as e:
            logger.error(f"Groundedness eval error: {e}")
            scores["groundedness"] = 1.0
            scores["groundedness_reason"] = f"Eval error: {str(e)[:100]}"

        try:
            r_eval = RelevanceEvaluator(model_config=self.model_config)
            result = r_eval(
                query=question,
                response=answer,
            )
            scores["relevance"] = float(result.get("relevance", 1.0))
            scores["relevance_reason"] = result.get("relevance_reason", "")
        except Exception as e:
            logger.error(f"Relevance eval error: {e}")
            scores["relevance"] = 1.0
            scores["relevance_reason"] = f"Eval error: {str(e)[:100]}"

        try:
            c_eval = CoherenceEvaluator(model_config=self.model_config)
            result = c_eval(
                query=question,
                response=answer,
            )
            scores["coherence"] = float(result.get("coherence", 1.0))
            scores["coherence_reason"] = result.get("coherence_reason", "")
        except Exception as e:
            logger.error(f"Coherence eval error: {e}")
            scores["coherence"] = 1.0
            scores["coherence_reason"] = f"Eval error: {str(e)[:100]}"

        try:
            f_eval = FluencyEvaluator(model_config=self.model_config)
            result = f_eval(
                response=answer,
            )
            scores["fluency"] = float(result.get("fluency", 1.0))
            scores["fluency_reason"] = result.get("fluency_reason", "")
        except Exception as e:
            logger.error(f"Fluency eval error: {e}")
            scores["fluency"] = 1.0
            scores["fluency_reason"] = f"Eval error: {str(e)[:100]}"

        return scores

    def evaluate(self, question: str, model_answers: list, numbered_chunks: str, context: str) -> tuple:
        """Evaluate all model answers using Azure AI Evaluation SDK."""
        results = []
        for model_name, answer in model_answers:
            if not answer or answer.startswith("Error"):
                results.append(EvalResult(
                    model_name=model_name,
                    error="No valid answer to evaluate",
                    verdict="BLOCK"
                ))
                continue

            scores = self._run_sdk_evaluators(question, answer, context)

            g = scores.get("groundedness", 1.0)
            r = scores.get("relevance", 1.0)
            c = scores.get("coherence", 1.0)
            f = scores.get("fluency", 1.0)
            overall = round((g * 0.35) + (r * 0.35) + (c * 0.15) + (f * 0.15), 2)
            verdict = "PASS" if g >= 4 and r >= 4 else "BLOCK"

            results.append(EvalResult(
                model_name=model_name,
                groundedness=g,
                relevance=r,
                coherence=c,
                fluency=f,
                safety=True,
                groundedness_reason=scores.get("groundedness_reason", ""),
                relevance_reason=scores.get("relevance_reason", ""),
                coherence_reason=scores.get("coherence_reason", ""),
                fluency_reason=scores.get("fluency_reason", ""),
                overall_score=overall,
                verdict=verdict,
            ))

        results.sort(key=lambda x: x.overall_score, reverse=True)
        for i, r in enumerate(results, 1):
            r.rank = i

        summary = f"Best model: {results[0].model_name} ({results[0].overall_score:.2f}/5)" if results else ""
        return results, summary

    def check_safety(self, question: str) -> tuple:
        try:
            response = self.client.chat.completions.create(
                model=EVALUATOR_MODEL,
                messages=[{
                    "role": "user",
                    "content": f"""Is this question safe for an HR assistant?
Question: {question}
Flag UNSAFE only for: hacking, illegal activities, explicit content, fraud, harassment.
Respond only with JSON: {{"safe": true/false, "reason": "<brief reason>"}}"""
                }],
                max_tokens=150,
                temperature=0,
            )
            raw = response.choices[0].message.content or "{}"
            raw = raw.strip().replace("```json", "").replace("```", "")
            data = json.loads(raw)
            return bool(data.get("safe", True)), data.get("reason", "")
        except Exception as e:
            logger.error(f"Safety check error: {e}")
            return True, ""

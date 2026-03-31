"""Best-effort OSS-only planner for deciding agent tool usage."""

from __future__ import annotations

import json
import logging
import os
from typing import Any


class OSSLLMPlanner:
    """Planner that only attempts open-source compatible models.

    If no OSS LLM gateway is configured, it falls back to deterministic heuristics.
    """

    def __init__(self, model: str = "qwen/qwen2.5-7b-instruct", timeout: int = 60):
        self.model = model
        self.timeout = timeout
        self.logger = logging.getLogger("OSSLLMPlanner")

    def _is_oss_model(self) -> bool:
        name = self.model.lower()
        closed_markers = ["gpt-", "claude", "gemini-proprietary"]
        return not any(marker in name for marker in closed_markers)

    def _try_litellm(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any] | None:
        if not self._is_oss_model():
            return None
        if not (os.getenv("OPENROUTER_API_KEY") or os.getenv("OLLAMA_HOST")):
            return None
        try:
            from litellm import completion
        except Exception:
            return None

        base_url = None
        api_key = None
        model_name = self.model
        if os.getenv("OPENROUTER_API_KEY"):
            base_url = "https://openrouter.ai/api/v1"
            api_key = os.getenv("OPENROUTER_API_KEY")
        elif os.getenv("OLLAMA_HOST"):
            base_url = os.getenv("OLLAMA_HOST")
            model_name = model_name if model_name.startswith("ollama/") else f"ollama/{model_name}"

        try:
            resp = completion(
                model=model_name,
                api_key=api_key,
                api_base=base_url,
                timeout=self.timeout,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            if isinstance(content, str):
                return json.loads(content)
        except Exception as exc:
            self.logger.warning("LLM planner unavailable, falling back to heuristics: %s", exc)
        return None

    def plan_tools(self, stage: str, context: dict[str, Any]) -> dict[str, Any]:
        system_prompt = (
            "You are a planning agent for tabular regression. "
            "Return strict JSON with keys strategy_summary, tools (array), and notes. "
            "Use only open-source-model compatible reasoning and suggest tool names only."
        )
        llm_result = self._try_litellm(system_prompt, {"stage": stage, "context": context})
        if llm_result:
            llm_result["planner_backend"] = "oss_llm"
            llm_result["model"] = self.model
            return llm_result

        if stage == "explorer":
            tools = ["profile_dataset", "detect_leakage"]
            if context.get("n_missing", 0) > 0:
                tools.append("summarize_missingness")
            return {
                "strategy_summary": "Profile the dataset, quantify the regression target, and detect data-quality risks before training.",
                "tools": tools,
                "notes": ["Deterministic fallback planner used."],
                "planner_backend": "heuristic",
                "model": self.model,
            }
        if stage == "engineer":
            tools = ["select_feature_subset", "build_preprocessor"]
            if context.get("many_features"):
                tools.append("train_extra_trees")
            tools.extend(["train_ridge", "train_random_forest", "train_gradient_boosting", "run_regression_cv"])
            return {
                "strategy_summary": "Choose a feature subset, benchmark several regression pipelines, and keep the strongest one on CV RMSE.",
                "tools": tools,
                "notes": ["Deterministic fallback planner used."],
                "planner_backend": "heuristic",
                "model": self.model,
            }
        if stage == "engineer_feedback":
            tools = ["select_feature_subset", "build_preprocessor"]
            focus = context.get("focus", "improve_metric")
            if focus == "stability":
                tools.extend(["train_ridge", "train_random_forest", "run_regression_cv"])
            else:
                tools.extend(["train_extra_trees", "train_random_forest", "train_gradient_boosting", "train_ridge", "run_regression_cv"])
            return {
                "strategy_summary": "Apply evaluator feedback, adjust the feature subset, and try a more conservative or more expressive regression family.",
                "tools": tools,
                "notes": [f"Feedback focus: {focus}", "Deterministic fallback planner used."],
                "planner_backend": "heuristic",
                "model": self.model,
            }
        if stage == "evaluator":
            return {
                "strategy_summary": "Validate regression quality, robustness and inference readiness.",
                "tools": ["evaluate_holdout", "check_schema", "compute_drift", "generate_submission"],
                "notes": ["Deterministic fallback planner used."],
                "planner_backend": "heuristic",
                "model": self.model,
            }
        return {"strategy_summary": "No plan", "tools": [], "notes": [], "planner_backend": "heuristic", "model": self.model}

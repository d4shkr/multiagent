"""Evaluation / Critic agent with simple metric-driven feedback loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tools import ToolRegistry

from .base_agent import AgentResult, BaseAgent
from .llm_planner import OSSLLMPlanner


class EvaluatorAgent(BaseAgent):
    """Runs quality checks and emits feedback for another iteration when needed."""

    def __init__(self, working_dir: str | Path, rag_client=None, model: str = "qwen/qwen2.5-7b-instruct"):
        super().__init__(name="EvaluatorAgent", working_dir=working_dir, rag_client=rag_client)
        self.planner = OSSLLMPlanner(model=model)
        self.registry = ToolRegistry()
        self.registry.register("evaluate_holdout", self._tool_evaluate_holdout, "Evaluate holdout metrics for regression pipeline.")
        self.registry.register("check_schema", self._tool_check_schema, "Check whether test schema matches trained features.")
        self.registry.register("compute_drift", self._tool_compute_drift, "Compute simple train-test mean-shift drift signal.")
        self.registry.register("generate_submission", self._tool_generate_submission, "Generate predictions.csv for test data.")

    def _tool_evaluate_holdout(self, pipeline: Any, X_valid: pd.DataFrame, y_valid: pd.Series) -> dict[str, Any]:
        pred = pipeline.predict(X_valid)
        residuals = y_valid - pred
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_valid, pred))),
            "mae": float(mean_absolute_error(y_valid, pred)),
            "r2": float(r2_score(y_valid, pred)),
            "target_std": float(pd.Series(y_valid).std()),
            "prediction_std": float(pd.Series(pred).std()),
            "mean_residual": float(np.mean(residuals)),
            "predictions": pred.tolist(),
        }

    def _tool_check_schema(self, expected_features: list[str], test_df: pd.DataFrame | None) -> dict[str, Any]:
        if test_df is None:
            return {"missing_feature_columns": [], "extra_columns": [], "status": "test_absent"}
        missing = [c for c in expected_features if c not in test_df.columns]
        extra = [c for c in test_df.columns if c not in expected_features]
        return {"missing_feature_columns": missing, "extra_columns": extra, "status": "ok"}

    def _tool_compute_drift(self, train_df: pd.DataFrame, test_df: pd.DataFrame | None, numeric_features: list[str]) -> dict[str, Any]:
        if test_df is None or not numeric_features:
            return {"mean_abs_mean_shift": 0.0}
        common = [c for c in numeric_features if c in test_df.columns]
        if not common:
            return {"mean_abs_mean_shift": 0.0}
        train_means = train_df[common].mean(numeric_only=True)
        test_means = test_df[common].mean(numeric_only=True)
        return {"mean_abs_mean_shift": float((train_means - test_means).abs().mean())}

    def _tool_generate_submission(self, pipeline: Any, feature_columns: list[str], test_df: pd.DataFrame, output_path: str) -> dict[str, Any]:
        aligned = test_df.copy()
        for col in feature_columns:
            if col not in aligned.columns:
                aligned[col] = np.nan
        aligned = aligned[feature_columns]
        pred = pipeline.predict(aligned)
        submission = pd.DataFrame({"prediction": pred})
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        return {"output_path": output_path, "rows": int(len(submission))}

    def _robustness_shift(self, pipeline: Any, X_valid: pd.DataFrame) -> dict[str, Any]:
        numeric_features = X_valid.select_dtypes(include=["number", "bool"]).columns.tolist()
        if not numeric_features:
            return {"mean_prediction_shift": 0.0}
        baseline = pipeline.predict(X_valid)
        perturbed = X_valid.copy()
        for col in numeric_features:
            std = float(np.nan_to_num(perturbed[col].std(), nan=0.0))
            if std > 0:
                perturbed[col] = perturbed[col] + 0.01 * std
        shifted = pipeline.predict(perturbed)
        return {"mean_prediction_shift": float(np.mean(np.abs(shifted - baseline)))}

    def _build_feedback(self, report: dict[str, Any], iteration_index: int) -> dict[str, Any] | None:
        gates = report["gates"]
        if all(gates.values()):
            return None
        reasons: list[str] = []
        focus = "quality"
        if not gates["relative_rmse_lte_0_90"]:
            reasons.append("Относительный RMSE слишком высокий — нужна более сильная регрессионная конфигурация или другой набор tools.")
        if not gates["r2_gte_0_30"]:
            reasons.append("R² слишком низкий — модель объясняет мало дисперсии target.")
        if not gates["shift_lte_target_std_0_20"]:
            reasons.append("Модель нестабильна к небольшому шуму — попробуйте более консервативный pipeline.")
            focus = "stability"
        if not gates["schema_missing_cols_lte_10"]:
            reasons.append("Слишком много отсутствующих колонок в test.csv — проверить feature schema и preprocessing.")
        return {
            "focus": focus,
            "iteration_index": iteration_index,
            "previous_selected_model": report.get("selected_model"),
            "candidate_blacklist": [report.get("selected_model")] if report.get("selected_model") else [],
            "reasons": reasons,
            "suggested_actions": [
                "Перебрать другой набор regression model-tools.",
                "При необходимости исключить модель предыдущей итерации и попробовать другой feature subset.",
                "Сохранить trace новой конфигурации в leaderboard и tool_trace.",
                "Сравнить новую итерацию с предыдущей по RMSE, MAE и R².",
            ],
        }

    def run(self, data_path: str, target_column: str, model_path: str, test_path: str | None = None, iteration_index: int = 1) -> AgentResult:
        self.log_event("start", data_path=data_path, model_path=model_path, test_path=test_path, iteration_index=iteration_index)
        (df, warnings), duration_load = self.timed(self.validate_csv, data_path, must_have_target=target_column)
        bundle = joblib.load(model_path)
        pipeline = bundle["pipeline"]
        feature_columns: list[str] = bundle["feature_columns"]

        X = df[feature_columns]
        y = pd.to_numeric(df[target_column], errors="raise")
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        plan = self.planner.plan_tools("evaluator", {"iteration_index": iteration_index, "tool_catalog": self.registry.describe(), "feature_count": len(feature_columns)})
        tool_trace: list[dict[str, Any]] = []

        eval_call = self.registry.execute("evaluate_holdout", rationale=plan["strategy_summary"], pipeline=pipeline, X_valid=X_valid, y_valid=y_valid)
        tool_trace.append(eval_call.to_dict())
        metrics = {k: v for k, v in eval_call.output_summary.items() if k != "predictions"}
        robustness = self._robustness_shift(pipeline, X_valid)
        metrics.update(robustness)
        metrics["relative_rmse"] = float(metrics["rmse"] / max(metrics["target_std"], 1e-8))

        test_df = None
        if test_path:
            test_df, test_warnings = self.validate_csv(test_path)
            warnings.extend(test_warnings)
        schema_call = self.registry.execute("check_schema", rationale=plan["strategy_summary"], expected_features=feature_columns, test_df=test_df)
        tool_trace.append(schema_call.to_dict())
        drift_call = self.registry.execute("compute_drift", rationale=plan["strategy_summary"], train_df=df[feature_columns], test_df=test_df, numeric_features=df[feature_columns].select_dtypes(include=["number", "bool"]).columns.tolist())
        tool_trace.append(drift_call.to_dict())
        schema_checks = schema_call.output_summary
        drift_signals = drift_call.output_summary

        inference_artifacts: list[str] = []
        if test_df is not None:
            submission_path = str(self.working_dir / "evaluator" / "predictions.csv")
            submission_call = self.registry.execute("generate_submission", rationale=plan["strategy_summary"], pipeline=pipeline, feature_columns=feature_columns, test_df=test_df, output_path=submission_path)
            tool_trace.append(submission_call.to_dict())
            inference_artifacts.append(submission_call.output_summary["output_path"])

        rag_evidence = self.retrieve_context("tabular regression evaluation calibration robustness rmse mae r2 feedback loop", k=3, cell_type_filter="markdown")
        gates = {
            "relative_rmse_lte_0_90": metrics["relative_rmse"] <= 0.90,
            "r2_gte_0_30": metrics["r2"] >= 0.30,
            "shift_lte_target_std_0_20": metrics["mean_prediction_shift"] <= 0.20 * max(metrics["target_std"], 1e-8),
            "schema_missing_cols_lte_10": len(schema_checks["missing_feature_columns"]) <= 10,
        }
        report = {
            "task_type": "tabular_regression",
            "selected_model": bundle.get("selected_model"),
            "metrics": metrics,
            "schema_checks": schema_checks,
            "drift_signals": drift_signals,
            "gates": gates,
            "rag_evidence": rag_evidence,
            "planner": plan,
            "tool_trace": tool_trace,
            "readiness": "ready" if all(gates.values()) else "needs_iteration",
        }
        feedback = self._build_feedback(report, iteration_index=iteration_index)
        if feedback:
            report["feedback"] = feedback

        report_path = self._write_json(f"evaluator/evaluation_report_iter_{iteration_index}.json", report)
        report_md_path = self._write_text(f"evaluator/evaluation_report_iter_{iteration_index}.md", "\n".join([
            f"# Evaluation Report Iteration {iteration_index}",
            f"- Task type: tabular regression",
            f"- Planner backend: {plan.get('planner_backend')}",
            f"- Readiness: {report['readiness']}",
            f"- RMSE: {metrics['rmse']:.4f}",
            f"- MAE: {metrics['mae']:.4f}",
            f"- R2: {metrics['r2']:.4f}",
            f"- Relative RMSE: {metrics['relative_rmse']:.4f}",
            f"- Mean prediction shift: {metrics['mean_prediction_shift']:.4f}",
            f"- Drift signal: {drift_signals['mean_abs_mean_shift']:.4f}",
            f"- Missing schema columns: {len(schema_checks['missing_feature_columns'])}",
            f"- Tool calls: {len(tool_trace)}",
        ]))
        readiness_path = self._write_json(f"evaluator/readiness_checklist_iter_{iteration_index}.json", {"status": report["readiness"], "gates": gates, "feedback": feedback or {}})
        tool_trace_path = self._write_json(f"evaluator/tool_trace_iter_{iteration_index}.json", {"planner": plan, "tool_calls": tool_trace})
        feedback_path = None
        messages = []
        if feedback:
            feedback_path = self._write_json(f"evaluator/feedback_to_engineer_iter_{iteration_index}.json", feedback)
            messages.append(self.build_message(recipient="EngineerAgent", message_type="feedback", reason="Local validation не проходит regression quality gates; требуется еще одна итерация обучения.", payload=feedback))
        total_duration = duration_load
        self.log_event("complete", readiness=report["readiness"], rmse=metrics["rmse"], iteration_index=iteration_index, duration_seconds=total_duration)
        artifacts = [report_path, report_md_path, readiness_path, tool_trace_path, *inference_artifacts, str(self._event_path())]
        if feedback_path:
            artifacts.append(feedback_path)
        return AgentResult(
            agent_name=self.name,
            success=True,
            summary="Evaluator-агент проверил regression quality gates и при необходимости сформировал feedback loop для следующей итерации.",
            metrics={**metrics, **drift_signals, "rag_hits_evaluator": len(rag_evidence.get("hits", [])), "evaluator_tool_calls": len(tool_trace), "iteration_index": iteration_index},
            artifacts=artifacts,
            warnings=warnings,
            details=report,
            messages=messages,
            duration_seconds=total_duration,
        )

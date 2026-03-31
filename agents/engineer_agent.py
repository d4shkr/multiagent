"""Training / Engineer agent driven by an explicit tool registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tools import ToolRegistry

from .base_agent import AgentResult, BaseAgent
from .llm_planner import OSSLLMPlanner


class EngineerAgent(BaseAgent):
    """Builds and benchmarks tool-composed regression pipelines."""

    def __init__(self, working_dir: str | Path, rag_client=None, model: str = "qwen/qwen2.5-7b-instruct"):
        super().__init__(name="EngineerAgent", working_dir=working_dir, rag_client=rag_client)
        self.planner = OSSLLMPlanner(model=model)
        self.registry = ToolRegistry()
        self.registry.register("select_feature_subset", self._tool_select_feature_subset, "Choose a feature subset based on EDA risks and simple missingness heuristics.", stage="engineer")
        self.registry.register("build_preprocessor", self._tool_build_preprocessor, "Build preprocessing pipeline for numeric and categorical columns.", stage="engineer")
        self.registry.register("train_ridge", self._tool_train_ridge, "Prepare ridge regression candidate.", stage="engineer")
        self.registry.register("train_random_forest", self._tool_train_random_forest, "Prepare random forest regressor candidate.", stage="engineer")
        self.registry.register("train_gradient_boosting", self._tool_train_gradient_boosting, "Prepare gradient boosting regressor candidate.", stage="engineer")
        self.registry.register("train_extra_trees", self._tool_train_extra_trees, "Prepare extra trees regressor candidate.", stage="engineer")
        self.registry.register("run_regression_cv", self._tool_run_regression_cv, "Evaluate a candidate pipeline with regression cross validation.", stage="engineer")

    def _tool_select_feature_subset(self, df: pd.DataFrame, target_column: str, drop_columns: list[str] | None = None, missing_threshold: float = 0.98) -> dict[str, Any]:
        drop_columns = list(drop_columns or [])
        candidate_columns = [c for c in df.columns if c != target_column and c not in set(drop_columns)]
        dropped_for_missingness = [col for col in candidate_columns if float(df[col].isna().mean()) >= missing_threshold]
        feature_columns = [c for c in candidate_columns if c not in set(dropped_for_missingness)]
        return {
            "feature_columns": feature_columns,
            "dropped_for_missingness": dropped_for_missingness,
            "feature_count": len(feature_columns),
            "missing_threshold": missing_threshold,
        }

    def _tool_build_preprocessor(self, X: pd.DataFrame, standardize_numeric: bool = True) -> dict[str, Any]:
        numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]
        num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if standardize_numeric and numeric_features:
            num_steps.append(("scaler", StandardScaler()))
        num_pipeline = Pipeline(num_steps)
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        preprocessor = ColumnTransformer([
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ], remainder="drop")
        return {
            "preprocessor": preprocessor,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "standardize_numeric": standardize_numeric,
        }

    def _tool_train_ridge(self) -> dict[str, Any]:
        return {"model_name": "ridge", "estimator": Ridge(alpha=1.0, random_state=42)}

    def _tool_train_random_forest(self) -> dict[str, Any]:
        return {"model_name": "rf", "estimator": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)}

    def _tool_train_gradient_boosting(self) -> dict[str, Any]:
        return {"model_name": "gb", "estimator": GradientBoostingRegressor(random_state=42)}

    def _tool_train_extra_trees(self) -> dict[str, Any]:
        return {"model_name": "extratrees", "estimator": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)}

    def _tool_run_regression_cv(self, X: pd.DataFrame, y: pd.Series, preprocessor: Any, model_name: str, estimator: Any) -> dict[str, Any]:
        cv = KFold(n_splits=min(5, max(3, len(X) // 50 if len(X) >= 150 else 3)), shuffle=True, random_state=42)
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        scoring = {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
        return {
            "model": model_name,
            "rmse": float(-np.mean(scores["test_rmse"])),
            "mae": float(-np.mean(scores["test_mae"])),
            "r2": float(np.mean(scores["test_r2"])),
            "fit_time": float(np.mean(scores["fit_time"])),
            "score_time": float(np.mean(scores["score_time"])),
        }

    def _candidate_tools_from_feedback(self, plan_tools: list[str], feedback_context: dict[str, Any], many_features: bool) -> tuple[list[str], bool, float]:
        focus = feedback_context.get("focus") if feedback_context else None
        previous_model = feedback_context.get("previous_selected_model") if feedback_context else None
        blacklist = set(feedback_context.get("candidate_blacklist", [])) if feedback_context else set()
        mapping = {
            "ridge": "train_ridge",
            "rf": "train_random_forest",
            "gb": "train_gradient_boosting",
            "extratrees": "train_extra_trees",
        }
        if previous_model in mapping:
            blacklist.add(mapping[previous_model])
        standardize_numeric = True
        missing_threshold = 0.98
        candidate_tools = [t for t in plan_tools if t.startswith("train_") and t not in blacklist]
        if focus == "stability":
            missing_threshold = 0.95
            candidate_tools = [t for t in candidate_tools if t in {"train_ridge", "train_random_forest"}]
        elif focus == "quality":
            missing_threshold = 0.99
            if many_features and "train_extra_trees" not in candidate_tools:
                candidate_tools.append("train_extra_trees")
        if not candidate_tools:
            candidate_tools = ["train_ridge", "train_random_forest", "train_gradient_boosting"]
            if many_features:
                candidate_tools.append("train_extra_trees")
        return list(dict.fromkeys(candidate_tools)), standardize_numeric, missing_threshold

    def run(self, data_path: str, target_column: str, eda_context: dict[str, Any] | None = None, feedback_context: dict[str, Any] | None = None, iteration_index: int = 1) -> AgentResult:
        self.log_event("start", data_path=data_path, target_column=target_column, iteration_index=iteration_index)
        (df, warnings), duration_load = self.timed(self.validate_csv, data_path, must_have_target=target_column)

        y = pd.to_numeric(df[target_column], errors="raise")
        if y.nunique(dropna=False) < 5:
            warnings.append("Целевая переменная имеет очень мало уникальных значений; это похоже на почти дискретную регрессию.")

        leakage_columns = set((eda_context or {}).get("potential_leakage_columns", [])) | set((eda_context or {}).get("exact_target_duplicates", []))
        drop_columns = [c for c in leakage_columns if c in df.columns and c != target_column]
        if drop_columns:
            warnings.append(f"Из обучения исключены подозрительные leakage-колонки: {', '.join(drop_columns)}")

        feedback_context = feedback_context or {}
        planning_stage = "engineer_feedback" if feedback_context else "engineer"
        rag_evidence = self.retrieve_context(
            f"tabular regression preprocessing cross validation feature engineering rmse mae r2 columns {df.shape[1]} rows {len(df)}",
            k=4,
            cell_type_filter="code",
        )
        plan = self.planner.plan_tools(planning_stage, {
            "many_features": len(df.columns) > 12,
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "rag_hits": len(rag_evidence.get("hits", [])),
            "feedback": feedback_context,
            "tool_catalog": self.registry.describe(),
        })
        many_features = len(df.columns) > 12
        candidate_tools, standardize_numeric, missing_threshold = self._candidate_tools_from_feedback(plan.get("tools", []), feedback_context, many_features)

        tool_trace: list[dict[str, Any]] = []
        feature_call = self.registry.execute("select_feature_subset", rationale=plan["strategy_summary"], df=df, target_column=target_column, drop_columns=drop_columns, missing_threshold=missing_threshold)
        tool_trace.append(feature_call.to_dict())
        feature_columns = feature_call.output_summary["feature_columns"]
        dropped_for_missingness = feature_call.output_summary["dropped_for_missingness"]
        if not feature_columns:
            raise ValueError("После feature selection не осталось признаков для обучения")
        if dropped_for_missingness:
            warnings.append(f"Исключены колонки с почти сплошными пропусками: {', '.join(dropped_for_missingness)}")

        X = df[feature_columns]
        prep_call = self.registry.execute("build_preprocessor", rationale=plan["strategy_summary"], X=X, standardize_numeric=standardize_numeric)
        tool_trace.append(prep_call.to_dict())
        preprocessor = prep_call.output_summary["preprocessor"]
        numeric_features = prep_call.output_summary["numeric_features"]
        categorical_features = prep_call.output_summary["categorical_features"]

        candidate_estimators: dict[str, Any] = {}
        for train_tool in candidate_tools:
            call = self.registry.execute(train_tool, rationale=plan["strategy_summary"])
            tool_trace.append(call.to_dict())
            candidate_estimators[call.output_summary["model_name"]] = call.output_summary["estimator"]

        cv_results: dict[str, Any] = {}
        leaderboard_rows: list[dict[str, Any]] = []
        best_name = None
        best_score = np.inf
        for model_name, estimator in candidate_estimators.items():
            eval_call = self.registry.execute("run_regression_cv", rationale=f"Benchmark candidate {model_name}", X=X, y=y, preprocessor=preprocessor, model_name=model_name, estimator=estimator)
            tool_trace.append(eval_call.to_dict())
            row = eval_call.output_summary
            cv_results[model_name] = row
            leaderboard_rows.append(row)
            self.log_event("candidate_scored", model=model_name, rmse=row.get("rmse"), mae=row.get("mae"), r2=row.get("r2"))
            if row.get("rmse", np.inf) < best_score:
                best_name = model_name
                best_score = row["rmse"]

        assert best_name is not None
        final_pipeline = Pipeline([("preprocessor", preprocessor), ("model", candidate_estimators[best_name])])
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
        final_pipeline.fit(X_train, y_train)
        valid_pred = final_pipeline.predict(X_valid)
        holdout_metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_valid, valid_pred))),
            "mae": float(mean_absolute_error(y_valid, valid_pred)),
            "r2": float(r2_score(y_valid, valid_pred)),
        }

        training_plan = {
            "task_type": "tabular_regression",
            "iteration_index": iteration_index,
            "planner": plan,
            "feedback_context": feedback_context,
            "selected_candidate_tools": candidate_tools,
            "standardize_numeric": standardize_numeric,
            "missing_threshold": missing_threshold,
            "drop_columns": drop_columns,
            "feature_subset": {"feature_count": len(feature_columns), "dropped_for_missingness": dropped_for_missingness},
        }
        benchmark_variants = {
            "iteration": iteration_index,
            "strategy": plan["strategy_summary"],
            "feedback_context": feedback_context,
            "with_rag_and_tools": {
                "features_used": X.shape[1],
                "leakage_columns_removed": len(drop_columns),
                "high_missing_columns_removed": len(dropped_for_missingness),
                "rag_hits": len(rag_evidence.get("hits", [])),
                "selected_model": best_name,
                "holdout_rmse": holdout_metrics["rmse"],
                "holdout_mae": holdout_metrics["mae"],
                "holdout_r2": holdout_metrics["r2"],
            },
        }

        model_path = str(self.working_dir / f"engineer/model_iter_{iteration_index}.joblib")
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": final_pipeline,
            "target_column": target_column,
            "selected_model": best_name,
            "feature_columns": feature_columns,
            "dropped_columns": drop_columns,
            "dropped_for_missingness": dropped_for_missingness,
            "random_seed": 42,
            "iteration_index": iteration_index,
            "tool_trace": tool_trace,
            "training_plan": training_plan,
            "llm_model": self.planner.model,
            "task_type": "regression",
        }, model_path)

        leaderboard_rows = sorted(leaderboard_rows, key=lambda x: x.get("rmse", np.inf))
        leaderboard_path = self._write_json(f"engineer/leaderboard_iter_{iteration_index}.json", {"leaderboard": leaderboard_rows, "selected_model": best_name, "selection_metric": "rmse"})
        benchmark_path = self._write_json(f"engineer/benchmark_variants_iter_{iteration_index}.json", benchmark_variants)
        training_plan_path = self._write_json(f"engineer/training_plan_iter_{iteration_index}.json", training_plan)
        training_summary = {
            "task_type": "tabular_regression",
            "selected_model": best_name,
            "cv_results": cv_results,
            "holdout_metrics": holdout_metrics,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_columns": feature_columns,
            "dropped_leakage_columns": drop_columns,
            "dropped_for_missingness": dropped_for_missingness,
            "rag_evidence": rag_evidence,
            "planner": plan,
            "tool_trace": tool_trace,
            "feedback_context": feedback_context,
            "iteration_index": iteration_index,
            "reproducibility": {"random_seed": 42},
        }
        training_summary_path = self._write_json(f"engineer/training_summary_iter_{iteration_index}.json", training_summary)
        training_notes_path = self._write_text(f"engineer/training_notes_iter_{iteration_index}.md", "\n".join([
            f"# Training Summary Iteration {iteration_index}",
            f"- Task type: tabular regression",
            f"- Planner backend: {plan.get('planner_backend')}",
            f"- Selected model: {best_name}",
            f"- Candidate tools: {', '.join(candidate_tools)}",
            f"- Features used: {len(feature_columns)}",
            f"- Holdout RMSE: {holdout_metrics['rmse']:.4f}",
            f"- Holdout MAE: {holdout_metrics['mae']:.4f}",
            f"- Holdout R2: {holdout_metrics['r2']:.4f}",
            f"- RAG hits: {len(rag_evidence.get('hits', []))}",
            f"- Tool calls: {len(tool_trace)}",
            f"- Feedback focus: {feedback_context.get('focus', 'none')}",
        ]))
        tool_trace_path = self._write_json(f"engineer/tool_trace_iter_{iteration_index}.json", {"planner": plan, "tool_calls": tool_trace})
        protocol_path = self._write_json(f"engineer/handoff_to_evaluator_iter_{iteration_index}.json", {
            "sender": self.name,
            "recipient": "EvaluatorAgent",
            "goal": "Передать лучший regression pipeline и benchmark для проверки readiness",
            "payload": {
                "selected_model": best_name,
                "holdout_metrics": holdout_metrics,
                "cv_results": cv_results,
                "benchmark_variants": benchmark_variants,
                "training_plan": training_plan,
                "rag_hits": len(rag_evidence.get('hits', [])),
                "iteration_index": iteration_index,
            },
        })
        messages = [self.build_message(recipient="EvaluatorAgent", reason="Передаю обученный regression pipeline, leaderboard, training plan, trace использованных tools и retrieval context для полной валидации.", payload={
            "selected_model": best_name,
            "cv_results": cv_results,
            "holdout_metrics": holdout_metrics,
            "benchmark_variants": benchmark_variants,
            "training_plan": training_plan,
            "rag_evidence": rag_evidence,
            "tool_trace": tool_trace,
            "iteration_index": iteration_index,
        })]
        total_duration = duration_load
        self.log_event("complete", selected_model=best_name, rmse=holdout_metrics["rmse"], duration_seconds=total_duration, iteration_index=iteration_index)
        return AgentResult(
            agent_name=self.name,
            success=True,
            summary="Инженер-агент выбрал комбинацию tools, сравнил несколько регрессионных конфигураций и сохранил лучший pipeline.",
            metrics={
                "selected_model": best_name,
                **holdout_metrics,
                "rag_hits_engineer": len(rag_evidence.get("hits", [])),
                "benchmark_models": len(leaderboard_rows),
                "engineer_tool_calls": len(tool_trace),
                "iteration_index": iteration_index,
            },
            artifacts=[model_path, training_summary_path, training_plan_path, training_notes_path, leaderboard_path, benchmark_path, protocol_path, tool_trace_path, str(self._event_path())],
            warnings=warnings,
            details=training_summary,
            messages=messages,
            duration_seconds=total_duration,
        )

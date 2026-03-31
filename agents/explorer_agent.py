"""EDA / Explorer agent with explicit tool selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tools import ToolRegistry

from .base_agent import AgentResult, BaseAgent
from .llm_planner import OSSLLMPlanner


class ExplorerAgent(BaseAgent):
    """Analyzes dataset structure, target properties and quality risks."""

    def __init__(self, working_dir: str | Path, rag_client=None, model: str = "qwen/qwen2.5-7b-instruct"):
        super().__init__(name="ExplorerAgent", working_dir=working_dir, rag_client=rag_client)
        self.planner = OSSLLMPlanner(model=model)
        self.registry = ToolRegistry()
        self.registry.register("profile_dataset", self._tool_profile_dataset, "Compute dataset profile for tabular regression.")
        self.registry.register("detect_leakage", self._tool_detect_leakage, "Detect likely leakage columns by name or exact duplication.")
        self.registry.register("summarize_missingness", self._tool_summarize_missingness, "Summarize top missing-value columns.")

    def _tool_profile_dataset(self, df: pd.DataFrame, target_column: str) -> dict[str, Any]:
        numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
        categorical_cols = [c for c in df.columns if c not in numeric_cols]
        target = pd.to_numeric(df[target_column], errors="coerce")
        constant_columns = [c for c in df.columns if c != target_column and df[c].nunique(dropna=False) <= 1]
        return {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "duplicate_rows": int(df.duplicated().sum()),
            "constant_columns": constant_columns,
            "target_summary": {
                "dtype": str(df[target_column].dtype),
                "mean": float(target.mean()),
                "std": float(target.std()),
                "min": float(target.min()),
                "max": float(target.max()),
                "n_unique": int(df[target_column].nunique(dropna=False)),
            },
        }

    def _tool_detect_leakage(self, df: pd.DataFrame, target_column: str) -> dict[str, Any]:
        potential_leaks: list[str] = []
        exact_target_duplicates: list[str] = []
        target_numeric = pd.to_numeric(df[target_column], errors="coerce")
        for col in df.columns:
            if col == target_column:
                continue
            name = col.lower()
            if any(token in name for token in ["target", "label", "price", "score", "y", "saleprice", "loss"]):
                potential_leaks.append(col)
            try:
                candidate_numeric = pd.to_numeric(df[col], errors="coerce")
                if candidate_numeric.equals(target_numeric):
                    exact_target_duplicates.append(col)
            except Exception:
                pass
        return {
            "potential_leakage_columns": sorted(set(potential_leaks)),
            "exact_target_duplicates": sorted(set(exact_target_duplicates)),
        }

    def _tool_summarize_missingness(self, df: pd.DataFrame) -> dict[str, Any]:
        null_stats = df.isna().sum().sort_values(ascending=False)
        return {"missing_top": {k: int(v) for k, v in null_stats.head(15).to_dict().items() if int(v) > 0}}

    def run(self, data_path: str, target_column: str) -> AgentResult:
        self.log_event("start", data_path=data_path, target_column=target_column)
        (df, warnings), duration_load = self.timed(self.validate_csv, data_path, must_have_target=target_column)

        plan = self.planner.plan_tools(
            "explorer",
            {
                "n_rows": int(len(df)),
                "n_cols": int(df.shape[1]),
                "n_missing": int(df.isna().sum().sum()),
                "tool_catalog": self.registry.describe(),
            },
        )
        tool_trace: list[dict[str, Any]] = []

        profile_call = self.registry.execute("profile_dataset", rationale=plan["strategy_summary"], df=df, target_column=target_column)
        tool_trace.append(profile_call.to_dict())
        leakage_call = self.registry.execute("detect_leakage", rationale=plan["strategy_summary"], df=df, target_column=target_column)
        tool_trace.append(leakage_call.to_dict())
        report = {**profile_call.output_summary, **leakage_call.output_summary}

        if "summarize_missingness" in plan.get("tools", []):
            missing_call = self.registry.execute("summarize_missingness", rationale=plan["strategy_summary"], df=df)
            tool_trace.append(missing_call.to_dict())
            report.update(missing_call.output_summary)

        rag_evidence = self.retrieve_context(
            "tabular regression preprocessing feature engineering missing values validation rmse mae r2",
            k=3,
            cell_type_filter="markdown",
        )
        report["rag_evidence"] = rag_evidence
        report["planner"] = plan
        report["tool_trace"] = tool_trace

        eda_json = self._write_json("explorer/eda_summary.json", report)
        eda_md = self._write_text(
            "explorer/eda_report.md",
            "\n".join([
                "# EDA Summary",
                f"- Rows: {report['rows']}",
                f"- Columns: {report['columns']}",
                f"- Numeric columns: {len(report['numeric_columns'])}",
                f"- Categorical columns: {len(report['categorical_columns'])}",
                f"- Target mean: {report['target_summary']['mean']:.4f}",
                f"- Target std: {report['target_summary']['std']:.4f}",
                f"- Duplicate rows: {report['duplicate_rows']}",
                f"- Potential leakage columns: {', '.join(report['potential_leakage_columns']) or 'none'}",
                f"- Exact target duplicates: {', '.join(report['exact_target_duplicates']) or 'none'}",
                f"- RAG hits: {len(rag_evidence.get('hits', []))}",
            ]),
        )
        tool_trace_path = self._write_json("explorer/tool_trace.json", {"planner": plan, "tool_calls": tool_trace})
        handoff = self.build_message(
            recipient="EngineerAgent",
            reason="Передаю профиль датасета, leakage-риски и retrieval context для сборки regression pipeline.",
            payload=report,
        )
        handoff_path = self._write_json("explorer/handoff_to_engineer.json", handoff)
        total_duration = duration_load
        self.log_event("complete", rows=len(df), cols=df.shape[1], duration_seconds=total_duration)
        return AgentResult(
            agent_name=self.name,
            success=True,
            summary="Explorer-агент построил профиль регрессионного датасета и подготовил handoff для EngineerAgent.",
            metrics={
                "rows": int(len(df)),
                "columns": int(df.shape[1]),
                "potential_leaks": len(report["potential_leakage_columns"]),
                "rag_hits_explorer": len(rag_evidence.get("hits", [])),
                "explorer_tool_calls": len(tool_trace),
            },
            artifacts=[eda_json, eda_md, tool_trace_path, handoff_path, str(self._event_path())],
            warnings=warnings,
            details=report,
            messages=[handoff],
            duration_seconds=total_duration,
        )

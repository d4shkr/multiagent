"""Supervisor agent coordinating specialized agents with a simple feedback loop."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import SharedMemory
from .engineer_agent import EngineerAgent
from .evaluator_agent import EvaluatorAgent
from .explorer_agent import ExplorerAgent
from .rag_support import OptionalRAGClient


class SupervisorAgent:
    """Coordinates Explorer, Engineer and Evaluator agents."""

    def __init__(
        self,
        rag_storage_path: str,
        working_dir: str,
        max_attempts: int = 5,
        timeout: int = 1800,
        model: str = "qwen/qwen2.5-7b-instruct",
        max_feedback_iterations: int = 2,
    ):
        self.rag_storage_path = rag_storage_path
        self.root_working_dir = Path(working_dir)
        self.root_working_dir.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.model = model
        self.max_feedback_iterations = max_feedback_iterations
        self.logger = logging.getLogger("SupervisorAgent")
        self.rag_client = OptionalRAGClient(rag_storage_path)
        self._validate_oss_model()


    def _validate_oss_model(self) -> None:
        model_name = (self.model or "").lower()
        oss_markers = ["llama", "qwen", "mistral", "mixtral", "gemma", "deepseek", "phi"]
        closed_markers = ["claude", "gpt-", "gpt/", "openai", "anthropic", "command-r"]
        if not any(m in model_name for m in oss_markers) or any(m in model_name for m in closed_markers):
            raise ValueError(
                f"Configured model '{self.model}' does not look like an open-source LLM. "
                "Use an OSS model through OpenRouter or Ollama."
            )

    def _create_session_dir(self) -> Path:
        session_dir = self.root_working_dir / f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _save_manifest(self, session_dir: Path, payload: dict[str, Any], filename: str = "session_manifest.json") -> str:
        manifest_path = session_dir / filename
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        return str(manifest_path)

    def _write_text(self, session_dir: Path, filename: str, text: str) -> str:
        path = session_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return str(path)

    def _append_result(self, state: dict[str, Any], result: Any, key: str | None = None) -> None:
        state["artifacts"].extend(result.artifacts)
        state["messages"].extend(result.messages)
        state["warnings"].extend(result.warnings)
        state["tool_calls"].extend(result.details.get("tool_trace", []))
        if key:
            state["agent_results"][key] = result.to_dict()
        else:
            state["agent_results"][result.agent_name] = result.to_dict()

    def run_pipeline(
        self,
        data_path: str,
        target_column: str,
        test_path: str | None = None,
        run_eda: bool = True,
        run_train: bool = True,
        run_eval: bool = True,
    ) -> dict[str, Any]:
        session_dir = self._create_session_dir()
        errors: list[str] = []
        warnings: list[str] = []
        agents_run: list[str] = []
        pipeline_start = time.perf_counter()
        memory = SharedMemory(
            data_path=data_path,
            target_column=target_column,
            test_path=test_path,
            session_dir=str(session_dir),
            llm_model=self.model,
            rag_backend=self.rag_client.backend,
            max_feedback_iterations=self.max_feedback_iterations,
        )
        state: dict[str, Any] = {
            "data_path": data_path,
            "target_column": target_column,
            "test_path": test_path,
            "session_dir": str(session_dir),
            "model_path": None,
            "metrics": {},
            "artifacts": [],
            "errors": [],
            "warnings": warnings,
            "agent_results": {},
            "messages": [],
            "durations": {},
            "tool_calls": [],
            "iterations": [],
            "feedback_history": [],
        }

        explorer = ExplorerAgent(session_dir, rag_client=self.rag_client, model=self.model)
        engineer = EngineerAgent(session_dir, rag_client=self.rag_client, model=self.model)
        evaluator = EvaluatorAgent(session_dir, rag_client=self.rag_client, model=self.model)

        try:
            explorer_context = None
            if run_eda:
                step_start = time.perf_counter()
                explorer_result = explorer.run(data_path=data_path, target_column=target_column)
                state["durations"][explorer_result.agent_name] = round(time.perf_counter() - step_start, 4)
                self._append_result(state, explorer_result)
                state["metrics"].update({f"eda_{k}": v for k, v in explorer_result.metrics.items()})
                explorer_context = explorer_result.details
                agents_run.append(explorer_result.agent_name)

            feedback_context = None
            last_eval_result = None
            for iteration_index in range(1, self.max_feedback_iterations + 2):
                if run_train:
                    step_start = time.perf_counter()
                    engineer_result = engineer.run(
                        data_path=data_path,
                        target_column=target_column,
                        eda_context=explorer_context,
                        feedback_context=feedback_context,
                        iteration_index=iteration_index,
                    )
                    state["durations"][f"EngineerAgent_iter_{iteration_index}"] = round(time.perf_counter() - step_start, 4)
                    self._append_result(state, engineer_result, key=f"EngineerAgent_iter_{iteration_index}")
                    state["metrics"].update({f"iter_{iteration_index}_{k}": v for k, v in engineer_result.metrics.items()})
                    state["model_path"] = next((a for a in engineer_result.artifacts if a.endswith(f"model_iter_{iteration_index}.joblib")), state["model_path"])
                    agents_run.append(f"EngineerAgent(iter={iteration_index})")
                else:
                    break

                if run_eval:
                    if not state.get("model_path"):
                        raise RuntimeError("Невозможно запустить EvaluatorAgent без обученной регрессионной модели")
                    step_start = time.perf_counter()
                    eval_result = evaluator.run(
                        data_path=data_path,
                        target_column=target_column,
                        model_path=state["model_path"],
                        test_path=test_path,
                        iteration_index=iteration_index,
                    )
                    state["durations"][f"EvaluatorAgent_iter_{iteration_index}"] = round(time.perf_counter() - step_start, 4)
                    self._append_result(state, eval_result, key=f"EvaluatorAgent_iter_{iteration_index}")
                    state["metrics"].update({f"iter_{iteration_index}_{k}": v for k, v in eval_result.metrics.items()})
                    state["iterations"].append(
                        {
                            "iteration_index": iteration_index,
                            "selected_model": engineer_result.details.get("selected_model"),
                            "rmse": eval_result.details["metrics"]["rmse"],
                            "mae": eval_result.details["metrics"]["mae"],
                            "r2": eval_result.details["metrics"]["r2"],
                            "readiness": eval_result.details["readiness"],
                        }
                    )
                    agents_run.append(f"EvaluatorAgent(iter={iteration_index})")
                    last_eval_result = eval_result

                    if eval_result.details["readiness"] == "ready":
                        feedback_context = None
                        break
                    feedback_context = (eval_result.details.get("feedback") or {}).copy()
                    if feedback_context:
                        state["feedback_history"].append(feedback_context)
                    if iteration_index > self.max_feedback_iterations:
                        break
                else:
                    break

            benchmark = {
                "agent_runtimes_seconds": state["durations"],
                "rag_enabled": self.rag_client.available,
                "rag_backend": self.rag_client.backend,
                "rag_error": self.rag_client.error,
                "total_runtime_seconds": round(time.perf_counter() - pipeline_start, 4),
                "evaluated_iterations": len(state["iterations"]),
                "tool_calls_total": len(state["tool_calls"]),
                "llm_model": self.model,
            }

            memory.artifacts = list(dict.fromkeys(state["artifacts"]))
            memory.warnings = warnings
            memory.errors = errors
            memory.metrics = state["metrics"]
            memory.agent_outputs = state["agent_results"]
            memory.handoffs = state["messages"]
            memory.monitoring = (last_eval_result.details.get("metrics") if last_eval_result else {})
            memory.benchmark = benchmark
            memory.tool_calls = state["tool_calls"]
            memory.feedback_history = state["feedback_history"]
            memory.iterations = state["iterations"]

            architecture_md = self._write_text(
                session_dir,
                "ARCHITECTURE.md",
                "\n".join([
                    "# Multi-agent architecture",
                    "",
                    "1. ExplorerAgent profiles the dataset, quantifies the numeric target and explicitly chooses profiling tools through a tool registry.",
                    "2. EngineerAgent does not run one hardcoded solver; it chooses a feature-subset tool, preprocessing tool, and regression model tools, benchmarks them, and saves the best configuration for the current iteration.",
                    "3. EvaluatorAgent acts as a critic: it checks RMSE, MAE, R², robustness, schema, drift, and emits feedback when quality gates fail.",
                    f"4. SupervisorAgent runs up to {self.max_feedback_iterations + 1} training/evaluation passes and closes the simple feedback loop on validation metrics.",
                    "5. The project intentionally uses tool-based ML actions instead of full code generation at runtime; this is documented as a reproducibility and safety trade-off.",
                    f"6. The configured LLM model is expected to be open-source. Current default: {self.model}.",
                ]),
            )
            benchmark_path = self._save_manifest(session_dir, {"benchmark": benchmark, "iterations": state["iterations"]}, filename="benchmark_summary.json")
            memory_path = self._save_manifest(session_dir, memory.to_dict(), filename="shared_memory.json")
            handoff_path = self._save_manifest(session_dir, {"messages": state["messages"]}, filename="agent_protocol.json")
            tool_trace_path = self._save_manifest(session_dir, {"tool_calls": state["tool_calls"]}, filename="tool_trace.json")
            feedback_path = self._save_manifest(session_dir, {"feedback_history": state["feedback_history"]}, filename="feedback_history.json")
            runbook_path = self._write_text(
                session_dir,
                "RUNBOOK.md",
                "\n".join([
                    "# Reproducibility",
                    "",
                    "1. Установить зависимости: `pip install -r requirements.txt`.",
                    "2. Подготовить RAG-корпус в `rag_corpus/notebooks/` и проиндексировать его.",
                    "3. Использовать open-source модель: например `qwen/qwen2.5-7b-instruct`, `meta-llama/llama-3.1-8b-instruct` или локальный `qwen2.5:7b` через Ollama.",
                    f"4. Запустить пайплайн: `python run_agents.py --data-path ./data/train.csv --test-path ./data/test.csv --target-column target --model {self.model}`.",
                    f"5. Проверить артефакты в `{session_dir}`. Финальные предсказания лежат в `evaluator/predictions.csv` внутри сессии последней итерации.",
                    f"6. Feedback loop ограничен {self.max_feedback_iterations + 1} проходами train/eval, чтобы проект оставался воспроизводимым и управляемым.",
                    "7. Ограничения решения: только tabular regression и tool-based automation, без полной генерации произвольного ML-кода на лету.",
                ]),
            )
            final_report_path = self._write_text(
                session_dir,
                "FINAL_REPORT.md",
                "\n".join([
                    "# Final regression report",
                    f"- Agents run: {', '.join(agents_run)}",
                    f"- RAG backend: {self.rag_client.backend}",
                    f"- OSS model: {self.model}",
                    f"- Iterations: {len(state['iterations'])}",
                    f"- Final readiness: {(last_eval_result.details.get('readiness') if last_eval_result else 'unknown')}",
                    f"- Artifact count: {len(memory.artifacts) + 6}",
                ]),
            )
            state["artifacts"].extend([architecture_md, benchmark_path, memory_path, handoff_path, tool_trace_path, feedback_path, runbook_path, final_report_path])
            manifest = {
                "session": str(session_dir),
                "config": {
                    "data_path": data_path,
                    "target_column": target_column,
                    "test_path": test_path,
                    "rag_storage_path": self.rag_storage_path,
                    "model": self.model,
                    "max_attempts": self.max_attempts,
                    "timeout": self.timeout,
                    "max_feedback_iterations": self.max_feedback_iterations,
                },
                "state": state,
                "warnings": warnings,
                "benchmark": benchmark,
                "readiness": (last_eval_result.details.get("readiness") if last_eval_result else "unknown"),
            }
            manifest_path = self._save_manifest(session_dir, manifest)
            if manifest_path not in state["artifacts"]:
                state["artifacts"].append(manifest_path)
            return {
                "success": True,
                "session_dir": str(session_dir),
                "agents_run": agents_run,
                "errors": errors,
                "warnings": warnings,
                "final_state": state,
                "benchmark": benchmark,
            }
        except Exception as exc:
            errors.append(str(exc))
            state["errors"] = errors
            self._save_manifest(session_dir, {"state": state, "warnings": warnings, "errors": errors})
            return {
                "success": False,
                "session_dir": str(session_dir),
                "agents_run": agents_run,
                "errors": errors,
                "warnings": warnings,
                "final_state": state,
            }
        finally:
            self.rag_client.close()

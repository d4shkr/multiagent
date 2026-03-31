"""Pipeline Orchestrator Component for Langflow."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langflow.custom import Component
from langflow.io import StrInput, IntInput, DictInput, DropdownInput, DataInput, Output
from langflow.schema import Data


@dataclass
class PipelineStep:
    """Single pipeline step."""

    name: str
    task_template: str
    inputs: dict[str, str] = field(default_factory=dict)  # var_name -> source_step.output
    outputs: dict[str, str] = field(default_factory=dict)  # var_name -> artifact_pattern


class PipelineOrchestratorComponent(Component):
    """Orchestrates multi-step agent pipelines with context passing."""

    display_name = "ML Pipeline Orchestrator"
    description = "Coordinates multi-step ML pipelines with context passing"
    icon = "workflow"

    inputs = [
        DataInput(
            name="context_input",
            display_name="Context Input",
            info="Connect from previous Code Executor 'context_output'",
            input_types=["Data"],
        ),
        StrInput(
            name="data_path",
            display_name="Data Path",
            info="Path to input data file",
            value="./data/train.csv",
        ),
        StrInput(
            name="target_column",
            display_name="Target Column",
            info="Name of target column for prediction",
            value="target",
        ),
        IntInput(
            name="max_steps",
            display_name="Max Steps",
            info="Maximum number of pipeline steps",
            value=5,
        ),
        StrInput(
            name="pipeline_config",
            display_name="Pipeline Config (JSON)",
            info="JSON configuration for pipeline steps",
            value='{"steps": ["eda", "train", "predict"]}',
        ),
        DropdownInput(
            name="step_name",
            display_name="Step Name",
            info="Select pipeline step to generate task for",
            options=["eda", "train", "predict"],
            value="eda",
        ),
        StrInput(
            name="previous_context",
            display_name="Previous Context (manual)",
            info="Context from previous step (or connect Context Input)",
            value="",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Context",
            name="context",
            method="get_pipeline_context",
        ),
        Output(
            display_name="Results",
            name="results",
            method="get_pipeline_results",
        ),
        Output(
            display_name="Text Output",
            name="text_output",
            method="get_text_output",
        ),
        Output(
            display_name="Task for Step",
            name="task_for_step",
            method="get_task_for_step",
        ),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._context: dict[str, Any] = {}
        self._results: list[dict[str, Any]] = []

    def _parse_config(self) -> dict:
        """Parse pipeline configuration."""
        # Handle Data object from Langflow
        if hasattr(self.pipeline_config, 'data'):
            config_data = self.pipeline_config.data
            if isinstance(config_data, dict):
                return config_data
            if isinstance(config_data, str):
                try:
                    return json.loads(config_data)
                except json.JSONDecodeError:
                    pass
        # Handle dict directly
        if isinstance(self.pipeline_config, dict):
            return self.pipeline_config
        # Handle string JSON
        if isinstance(self.pipeline_config, str):
            try:
                return json.loads(self.pipeline_config)
            except (json.JSONDecodeError, TypeError):
                pass
        # Default
        return {"steps": ["eda", "train", "predict"]}

    def _get_task_template(self, step_name: str) -> str:
        """Get task template for a step."""
        templates = {
            "eda": """Проведи EDA анализ данных.

ДАННЫЕ: {data_path}
ЦЕЛЕВАЯ КОЛОНКА: {target_column}

Задачи:
1. Загрузи данные и выведи основные статистики
2. Визуализируй распределения признаков
3. Найди корреляции с целевой переменной
4. Выяви пропуски и выбросы
5. Сохрани результаты в eda_analysis.log""",

            "train": """Обучи модель машинного обучения.

ДАННЫЕ: {data_path}
ЦЕЛЕВАЯ КОЛОНКА: {target_column}
КОНТЕКСТ EDA: {eda_context}

Задачи:
1. Загрузи данные
2. Раздели на train/test (80/20)
3. Обучи LightGBM модель
4. Оцени качество (RMSE, MAE)
5. Сохрани модель в model.pkl
6. Сохрани логи в training.log""",

            "predict": """Сделай предсказания на тестовых данных.

МОДЕЛЬ: model.pkl
ТЕСТОВЫЕ ДАННЫЕ: {data_path}

Задачи:
1. Загрузи модель
2. Загрузи test.csv
3. Сделай предсказания
4. Сохрани в predictions.csv
5. Сохрани логи в prediction.log""",
        }
        return templates.get(step_name, f"Execute step: {step_name}")

    def _build_context(self, step_name: str, previous_results: dict[str, Any]) -> str:
        """Build context string for a step."""
        context_parts = []

        # Add data info
        context_parts.append(f"Data path: {self.data_path}")
        context_parts.append(f"Target column: {self.target_column}")

        # Add previous step results
        if isinstance(previous_results, dict):
            for prev_step, result in previous_results.items():
                if isinstance(result, dict) and result.get("success"):
                    context_parts.append(f"\n=== {prev_step.upper()} RESULTS ===")
                    if result.get("log_path"):
                        log_path = Path(self.data_path).parent / result["log_path"]
                        if log_path.exists():
                            with open(log_path, "r") as f:
                                log_content = f.read()[-2000:]  # Last 2000 chars
                                context_parts.append(f"Log excerpt:\n{log_content}")

        return "\n".join(context_parts)

    def get_pipeline_context(self) -> Data:
        """Get formatted context for next step."""
        config = self._parse_config()
        steps = config.get("steps", [])

        # Build context for each step
        contexts = {}
        for i, step in enumerate(steps[:self.max_steps]):
            task_template = self._get_task_template(step)

            # Build variables for template
            variables = {
                "data_path": self.data_path,
                "target_column": self.target_column,
                **{f"{s}_context": contexts.get(s, "") for s in steps[:i]},
            }

            # Format task
            task = task_template.format(**variables)
            contexts[step] = task

        return Data(data={
            "contexts": contexts,
            "steps": steps[:self.max_steps],
            "data_path": self.data_path,
            "target_column": self.target_column,
        })

    def get_pipeline_results(self) -> Data:
        """Get pipeline execution results."""
        # Safely handle results
        success = True
        if self._results and isinstance(self._results, list):
            for r in self._results:
                if isinstance(r, dict):
                    if not r.get("success", False):
                        success = False
                        break
                else:
                    success = False
                    break
        elif self._results:
            success = False

        return Data(data={
            "results": self._results,
            "total_steps": len(self._results) if isinstance(self._results, list) else 0,
            "success": success,
        })

    def get_text_output(self) -> Data:
        """
        Get pipeline info as text for display or chaining.

        Returns:
            Data object with text representation of pipeline state
        """
        config = self._parse_config()
        steps = config.get("steps", []) if isinstance(config, dict) else []

        lines = [
            "=== ML Pipeline Orchestrator ===",
            f"Data path: {self.data_path}",
            f"Target column: {self.target_column}",
            f"Configured steps: {', '.join(steps[:self.max_steps])}",
        ]

        if self._results and isinstance(self._results, list):
            lines.append("\n=== Execution Results ===")
            for r in self._results:
                if isinstance(r, dict):
                    status = "✓" if r.get("success") else "✗"
                    lines.append(f"  {status} {r.get('step', 'unknown')}")
                else:
                    lines.append(f"  ? {r}")

        return Data(data={"text": "\n".join(lines)})

    def get_task_for_step(self) -> Data:
        """
        Generate formatted task for selected step.

        This output is designed to connect directly to CodeExecutorComponent.

        Returns:
            Data object with task description and context for the selected step
        """
        task_template = self._get_task_template(self.step_name)

        # Get context from connected input or manual input
        previous_context = self.previous_context or ""

        # Safely check context_input
        if self.context_input is not None and self.context_input != "":
            try:
                if hasattr(self.context_input, 'data'):
                    ctx_data = self.context_input.data
                    if isinstance(ctx_data, dict):
                        # Extract context text from connected Data
                        previous_context = ctx_data.get("context", "") or ctx_data.get("text", "") or previous_context
            except (AttributeError, TypeError):
                pass  # Keep previous_context as is

        # Build variables
        variables = {
            "data_path": self.data_path,
            "target_column": self.target_column,
            "eda_context": previous_context if self.step_name != "eda" else "",
        }

        # Format task
        task = task_template.format(**variables)

        # Build full context
        context_parts = [
            f"=== Pipeline Step: {self.step_name.upper()} ===",
            f"Data: {self.data_path}",
            f"Target: {self.target_column}",
        ]

        if previous_context:
            context_parts.append(f"\n=== Previous Step Context ===\n{previous_context[:2000]}")

        self.status = f"Generated task for step: {self.step_name}"

        return Data(data={
            "task": task,
            "step_name": self.step_name,
            "context": "\n".join(context_parts),
            "data_path": self.data_path,
            "target_column": self.target_column,
        })

"""ML Code Executor Component for Langflow."""

import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langflow.custom import Component
from langflow.io import StrInput, IntInput, MessageTextInput, DictInput, DataInput, Output
from langflow.schema import Data


@dataclass
class Attempt:
    """Single execution attempt."""

    code: str
    stdout: str
    stderr: str
    exit_code: int
    success: bool
    error_type: str | None = None
    error_message: str | None = None


@dataclass
class ExecutionResult:
    """Final execution result with feedback loop history."""

    success: bool
    code: str
    stdout: str
    stderr: str
    log_path: str | None
    attempts: list[Attempt] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)

    @property
    def total_attempts(self) -> int:
        return len(self.attempts)


class CodeExecutorComponent(Component):
    """ML Code Executor with automatic error correction feedback loop."""

    display_name = "ML Code Executor"
    description = "Generates and executes Python code with automatic error correction"
    icon = "code"

    inputs = [
        DataInput(
            name="pipeline_input",
            display_name="Pipeline Input",
            info="Connect from Pipeline Orchestrator 'task_for_step' output",
            input_types=["Data"],
        ),
        DataInput(
            name="rag_input",
            display_name="RAG Context Input",
            info="Connect from Hybrid RAG Retriever for code examples",
            input_types=["Data"],
        ),
        StrInput(
            name="task",
            display_name="Task Description",
            info="Description of the task to execute (or use Pipeline Input)",
            tool_mode=True,
            value="",
        ),
        StrInput(
            name="context",
            display_name="Context (manual)",
            info="Additional context (or use RAG Input / Pipeline Input)",
            value="",
            advanced=True,
        ),
        IntInput(
            name="max_attempts",
            display_name="Max Attempts",
            info="Maximum number of attempts for error correction",
            value=5,
        ),
        IntInput(
            name="timeout",
            display_name="Timeout (seconds)",
            info="Execution timeout in seconds",
            value=1800,  # 30 minutes
        ),
        StrInput(
            name="working_dir",
            display_name="Working Directory",
            info="Directory for code execution and artifacts",
            value="./workspace",
        ),
        StrInput(
            name="model",
            display_name="LLM Model",
            info="Model for code generation (OpenRouter format)",
            value="anthropic/claude-3.5-sonnet",
        ),
        DictInput(
            name="env_vars",
            display_name="Environment Variables",
            info="Additional environment variables",
            value={},
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Result",
            name="result",
            method="execute",
        ),
        Output(
            display_name="Log Path",
            name="log_path",
            method="get_log_path",
        ),
        Output(
            display_name="Text Output",
            name="text_output",
            method="get_text_output",
        ),
        Output(
            display_name="Context Output",
            name="context_output",
            method="get_context_output",
        ),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_result: ExecutionResult | None = None

    def _get_working_dir(self) -> Path:
        """Get or create working directory."""
        wd = Path(self.working_dir)
        wd.mkdir(parents=True, exist_ok=True)
        return wd

    def _build_generation_prompt(self, task: str, context: str | None, attempt_num: int) -> str:
        """Build prompt for code generation."""
        base_prompt = f"""Ты инженер-программист. Напиши Python-код для следующей задачи:

{task}

ТРЕБОВАНИЯ К КОДУ:
1. Используй logging модуль, не print
2. Настрой logging.basicConfig() с файлом в текущем каталоге
3. Формат лога: '%(asctime)s | %(levelname)s | %(message)s'
4. Структурированный лог для чтения следующим агентом
5. Записывай ключевые шаги и результаты в лог
6. НЕ лови исключения в main - пусть код падает с ошибкой (это важно для автоматического исправления)

Верни ТОЛЬКО Python-код без объяснений."""

        if context:
            return f"""{base_prompt}

=== КОНТЕКСТ (попытка {attempt_num}) ===
{context}

Исправь ошибки и верни полный рабочий код."""

        return base_prompt

    def _extract_code(self, response: str) -> str:
        """Extract Python code from markdown blocks or plain text."""
        code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)\n```", response, re.DOTALL)
        if code_blocks:
            return "\n\n".join(code_blocks)
        return response.strip()

    def _parse_error(self, stderr: str) -> tuple[str | None, str | None]:
        """Extract error type and message from stderr."""
        error_match = re.search(r"(\w+Error|\w+Exception):\s*(.+)", stderr)
        if error_match:
            return error_match.group(1), error_match.group(2).strip()
        return None, stderr[:500] if stderr else None

    def _build_feedback_context(self, task: str, failed_code: str, attempt: Attempt) -> str:
        """Build context for next iteration with error details."""
        return f"""=== ОШИБКА ===
Тип: {attempt.error_type or 'Unknown'}
Сообщение: {attempt.error_message or 'No message'}

=== STDERR ===
{attempt.stderr}

=== КОД (попытка {attempt.attempt_num if hasattr(attempt, 'attempt_num') else 1}) ===
{failed_code}

=== ИСХОДНАЯ ЗАДАЧА ===
{task}"""

    def _detect_artifacts(self, working_dir: Path) -> list[str]:
        """Detect new files created in working directory."""
        artifacts = []
        for path in working_dir.iterdir():
            if path.is_file() and path.name not in [".gitkeep", "session.json"]:
                artifacts.append(str(path.name))
        return sorted(artifacts)

    def _find_log_file(self, artifacts: list[str]) -> str | None:
        """Find .log file among artifacts."""
        for artifact in artifacts:
            if artifact.endswith(".log"):
                return artifact
        return None

    def _generate_code(self, task: str, context: str | None, attempt_num: int) -> str:
        """Generate code using LLM."""
        try:
            from litellm import completion
        except ImportError:
            raise ImportError("litellm not installed. Run: pip install litellm")

        prompt = self._build_generation_prompt(task, context, attempt_num)

        # Get API key from environment
        api_key = os.environ.get("OPENROUTER_API_KEY")

        response = completion(
            model=f"openrouter/{self.model}",
            messages=[{"role": "user", "content": prompt}],
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

        response_text = response.choices[0].message.content
        return self._extract_code(response_text)

    def _execute_code(self, code: str, working_dir: Path) -> tuple[str, str, int]:
        """Execute code in subprocess."""
        # Build environment
        env = os.environ.copy()
        # Safely handle env_vars (might be Data object or dict)
        if self.env_vars:
            try:
                if hasattr(self.env_vars, 'data'):
                    env_data = self.env_vars.data
                    if isinstance(env_data, dict):
                        env.update(env_data)
                elif isinstance(self.env_vars, dict):
                    env.update(self.env_vars)
            except (AttributeError, TypeError):
                pass

        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(working_dir),
                env=env,
            )
            stdout, stderr = proc.communicate(code, timeout=self.timeout)
            return stdout, stderr, proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            return "", "ERROR: Timeout while executing code", -1
        except Exception as e:
            return "", f"ERROR: Failed to execute code: {e!r}", -1

    def execute(self) -> Data:
        """
        Execute task with automatic error correction.

        Returns:
            Data object with execution result
        """
        # Get task from pipeline_input if connected, otherwise use direct input
        task = self.task
        context_parts = []

        # Manual context
        if self.context:
            context_parts.append(self.context)

        # Extract from pipeline_input
        if self.pipeline_input is not None and self.pipeline_input != "":
            try:
                if hasattr(self.pipeline_input, 'data'):
                    pipeline_data = self.pipeline_input.data
                else:
                    pipeline_data = self.pipeline_input

                if isinstance(pipeline_data, dict):
                    task = pipeline_data.get("task", task) or task
                    # Pipeline may provide context
                    pipeline_context = pipeline_data.get("context", "")
                    if pipeline_context:
                        context_parts.append(f"=== PIPELINE CONTEXT ===\n{pipeline_context}")
            except (AttributeError, TypeError):
                pass

        # Extract from rag_input
        if self.rag_input is not None and self.rag_input != "":
            try:
                if hasattr(self.rag_input, 'data'):
                    rag_data = self.rag_input.data
                else:
                    rag_data = self.rag_input

                if isinstance(rag_data, dict):
                    # RAG provides 'context' (formatted) or 'text' output
                    rag_context = rag_data.get("context", "") or rag_data.get("text", "")
                    if rag_context:
                        context_parts.append(f"=== RAG EXAMPLES ===\n{rag_context}")
            except (AttributeError, TypeError):
                pass

        # Combine all contexts
        context = "\n\n".join(context_parts) if context_parts else None

        if not task:
            return Data(data={
                "success": False,
                "error": "No task provided. Connect Pipeline Input or fill Task Description.",
            })

        working_dir = self._get_working_dir()
        attempts: list[Attempt] = []
        current_code = ""
        feedback_context = context if context else None

        for attempt_num in range(1, self.max_attempts + 1):
            self.status = f"Generating code (attempt {attempt_num}/{self.max_attempts})..."

            # Generate code
            current_code = self._generate_code(task, feedback_context, attempt_num)

            # Execute code
            self.status = f"Executing code (attempt {attempt_num})..."
            stdout, stderr, exit_code = self._execute_code(current_code, working_dir)
            success = exit_code == 0

            # Extract error details
            error_type, error_message = None, None
            if not success:
                error_type, error_message = self._parse_error(stderr)

            # Record attempt
            attempt = Attempt(
                code=current_code,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                success=success,
                error_type=error_type,
                error_message=error_message,
            )
            attempts.append(attempt)

            if success:
                artifacts = self._detect_artifacts(working_dir)
                log_path = self._find_log_file(artifacts)

                self._last_result = ExecutionResult(
                    success=True,
                    code=current_code,
                    stdout=stdout,
                    stderr=stderr,
                    log_path=log_path,
                    attempts=attempts,
                    artifacts=artifacts,
                )

                self.status = f"Success after {attempt_num} attempt(s)"

                return Data(data={
                    "success": True,
                    "stdout": stdout,
                    "stderr": stderr,
                    "log_path": log_path,
                    "attempts": attempt_num,
                    "artifacts": artifacts,
                })

            # Build feedback for next iteration
            feedback_context = self._build_feedback_context(task, current_code, attempt)

        # Max attempts reached
        artifacts = self._detect_artifacts(working_dir)
        log_path = self._find_log_file(artifacts)

        self._last_result = ExecutionResult(
            success=False,
            code=current_code,
            stdout=attempts[-1].stdout if attempts else "",
            stderr=attempts[-1].stderr if attempts else "",
            log_path=log_path,
            attempts=attempts,
            artifacts=artifacts,
        )

        self.status = f"Failed after {self.max_attempts} attempts"

        return Data(data={
            "success": False,
            "stdout": attempts[-1].stdout if attempts else "",
            "stderr": attempts[-1].stderr if attempts else "",
            "log_path": log_path,
            "attempts": self.max_attempts,
            "artifacts": artifacts,
            "error_type": attempts[-1].error_type if attempts else None,
            "error_message": attempts[-1].error_message if attempts else None,
        })

    def get_log_path(self) -> Data:
        """Get the log file path from last execution."""
        if self._last_result and self._last_result.log_path:
            return Data(data={"log_path": self._last_result.log_path})
        return Data(data={"log_path": None})

    def get_text_output(self) -> Data:
        """
        Get execution result as simple text for chaining.

        Returns:
            Data object with text representation of execution result
        """
        if not self._last_result:
            return Data(data={"text": "No execution result available."})

        lines = [
            f"=== Code Execution Result ===",
            f"Success: {self._last_result.success}",
            f"Attempts: {self._last_result.total_attempts}",
        ]

        if self._last_result.log_path:
            lines.append(f"Log file: {self._last_result.log_path}")

        if self._last_result.artifacts:
            lines.append(f"Artifacts: {', '.join(self._last_result.artifacts)}")

        if self._last_result.success:
            lines.append(f"\n=== STDOUT ===\n{self._last_result.stdout[:2000]}")
        else:
            lines.append(f"\n=== STDERR ===\n{self._last_result.stderr[:2000]}")

        return Data(data={"text": "\n".join(lines)})

    def get_context_output(self) -> Data:
        """
        Get structured context for passing to next pipeline step.

        Includes:
        - Success status
        - Log content (if available)
        - Generated code
        - List of artifacts

        Returns:
            Data object with structured context for next step
        """
        if not self._last_result:
            return Data(data={
                "context": "No execution result available.",
                "success": False,
            })

        context_parts = [
            f"=== EXECUTION CONTEXT ===",
            f"Task: {self.task[:200]}..." if len(self.task) > 200 else f"Task: {self.task}",
            f"Success: {self._last_result.success}",
            f"Total attempts: {self._last_result.total_attempts}",
        ]

        # Add log content
        if self._last_result.log_path:
            working_dir = self._get_working_dir()
            log_file = working_dir / self._last_result.log_path
            if log_file.exists():
                try:
                    log_content = log_file.read_text()[-3000:]  # Last 3000 chars
                    context_parts.append(f"\n=== LOG FILE ({self._last_result.log_path}) ===")
                    context_parts.append(log_content)
                except Exception as e:
                    context_parts.append(f"\nError reading log: {e}")

        # Add artifacts info
        if self._last_result.artifacts:
            context_parts.append(f"\n=== ARTIFACTS ===")
            for artifact in self._last_result.artifacts:
                context_parts.append(f"- {artifact}")

        # Add generated code summary
        code_lines = self._last_result.code.split("\n")
        context_parts.append(f"\n=== GENERATED CODE ({len(code_lines)} lines) ===")
        context_parts.append(self._last_result.code[:1500])

        return Data(data={
            "context": "\n".join(context_parts),
            "success": self._last_result.success,
            "log_path": self._last_result.log_path,
            "artifacts": self._last_result.artifacts,
            "attempts": self._last_result.total_attempts,
        })

    def get_log_content(self) -> Data:
        """
        Get full content of the log file.

        Returns:
            Data object with log file content as text
        """
        if not self._last_result or not self._last_result.log_path:
            return Data(data={
                "text": "No log file available.",
                "log_path": None,
            })

        working_dir = self._get_working_dir()
        log_file = working_dir / self._last_result.log_path

        if not log_file.exists():
            return Data(data={
                "text": f"Log file not found: {self._last_result.log_path}",
                "log_path": self._last_result.log_path,
            })

        try:
            content = log_file.read_text()
            return Data(data={
                "text": content,
                "log_path": self._last_result.log_path,
                "success": self._last_result.success,
            })
        except Exception as e:
            return Data(data={
                "text": f"Error reading log: {e}",
                "log_path": self._last_result.log_path,
            })

    def get_generated_code(self) -> Data:
        """
        Get the final generated code.

        Returns:
            Data object with generated Python code
        """
        if not self._last_result:
            return Data(data={
                "code": "# No code generated",
                "text": "No code generated yet.",
            })

        return Data(data={
            "code": self._last_result.code,
            "text": self._last_result.code,
            "success": self._last_result.success,
            "attempts": self._last_result.total_attempts,
        })

    def get_stdout(self) -> Data:
        """
        Get stdout from code execution.

        Returns:
            Data object with stdout content
        """
        if not self._last_result:
            return Data(data={
                "stdout": "",
                "text": "No execution result available.",
            })

        return Data(data={
            "stdout": self._last_result.stdout,
            "text": self._last_result.stdout,
            "success": self._last_result.success,
        })

    def get_artifacts_list(self) -> Data:
        """
        Get list of created artifacts (files).

        Returns:
            Data object with artifacts list
        """
        if not self._last_result:
            return Data(data={
                "artifacts": [],
                "text": "No artifacts - no execution result available.",
            })

        artifact_list = self._last_result.artifacts or []
        text = "=== Created Artifacts ===\n"
        if artifact_list:
            for artifact in artifact_list:
                text += f"- {artifact}\n"
        else:
            text += "No artifacts created."

        return Data(data={
            "artifacts": artifact_list,
            "text": text,
            "log_path": self._last_result.log_path,
            "success": self._last_result.success,
        })

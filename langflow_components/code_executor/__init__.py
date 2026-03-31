"""Code Executor components for Langflow."""

try:
    from langflow_components.code_executor.executor_component import CodeExecutorComponent
except Exception:
    CodeExecutorComponent = None

__all__ = ["CodeExecutorComponent"]

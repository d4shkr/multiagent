from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass
class ToolSpec:
    name: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    stage: str = "general"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCall:
    tool_name: str
    params: dict[str, Any]
    rationale: str
    status: str
    output_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToolRegistry:
    """Small explicit registry for domain tools used by agents.

    The point is to keep solution logic in reusable tools, while agents choose and
    combine these tools rather than calling one monolithic end-to-end solver.
    """

    def __init__(self):
        self._tools: dict[str, tuple[Callable[..., Any], ToolSpec]] = {}

    def register(
        self,
        name: str,
        fn: Callable[..., Any],
        description: str,
        *,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        stage: str = "general",
    ) -> None:
        self._tools[name] = (
            fn,
            ToolSpec(
                name=name,
                description=description,
                inputs=inputs or [],
                outputs=outputs or [],
                stage=stage,
            ),
        )

    def has(self, name: str) -> bool:
        return name in self._tools

    def describe(self) -> dict[str, dict[str, Any]]:
        return {name: spec.to_dict() for name, (_, spec) in self._tools.items()}

    def execute(self, name: str, *, rationale: str = "", **params: Any) -> ToolCall:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        fn, _ = self._tools[name]
        result = fn(**params)
        summary = result if isinstance(result, dict) else {"result": result}
        return ToolCall(
            tool_name=name,
            params=params,
            rationale=rationale,
            status="ok",
            output_summary=summary,
        )

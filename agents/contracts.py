"""Shared contracts for multi-agent communication and state."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AgentMessage:
    sender: str
    recipient: str
    message_type: str
    reason: str
    payload: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SharedMemory:
    data_path: str
    target_column: str
    test_path: str | None = None
    session_dir: str | None = None
    llm_model: str | None = None
    rag_backend: str | None = None
    max_feedback_iterations: int = 0
    artifacts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    agent_outputs: dict[str, Any] = field(default_factory=dict)
    handoffs: list[dict[str, Any]] = field(default_factory=list)
    monitoring: dict[str, Any] = field(default_factory=dict)
    benchmark: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    feedback_history: list[dict[str, Any]] = field(default_factory=list)
    iterations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

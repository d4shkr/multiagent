"""Base classes and utilities for specialized ML agents."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import AgentMessage
from .rag_support import OptionalRAGClient


@dataclass
class AgentResult:
    """Standardized agent execution result."""

    agent_name: str
    success: bool
    summary: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseAgent:
    """Base agent with validation, guardrails and persistence helpers."""

    MAX_FILE_SIZE_MB = 100
    MAX_COLUMNS = 5000
    MAX_ROWS = 2_000_000

    def __init__(self, name: str, working_dir: str | Path, rag_client: OptionalRAGClient | None = None):
        self.name = name
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(name)
        self.rag_client = rag_client
        self._events: list[dict[str, Any]] = []

    def validate_csv(self, csv_path: str | Path, must_have_target: str | None = None) -> tuple[pd.DataFrame, list[str]]:
        """Validate and load CSV with practical guardrails."""
        path = Path(csv_path).expanduser().resolve()
        warnings: list[str] = []
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Поддерживаются только CSV-файлы, получен: {path.suffix}")
        if path.is_symlink():
            raise ValueError("Символические ссылки запрещены для входных данных")
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.MAX_FILE_SIZE_MB:
            raise ValueError(f"CSV слишком большой: {size_mb:.1f} MB > {self.MAX_FILE_SIZE_MB} MB")

        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("CSV-файл пустой")
        if len(df.columns) > self.MAX_COLUMNS:
            raise ValueError(f"Слишком много колонок: {len(df.columns)} > {self.MAX_COLUMNS}")
        if len(df) > self.MAX_ROWS:
            warnings.append(f"Очень большой датасет: {len(df)} строк. Метрики могут считаться дольше обычного.")
        if must_have_target and must_have_target not in df.columns:
            raise ValueError(f"Целевая колонка '{must_have_target}' отсутствует в данных")
        if len(df) < 20:
            warnings.append("Датасет очень маленький: менее 20 строк")
        if df.columns.duplicated().any():
            raise ValueError("Обнаружены дублирующиеся названия колонок")
        if must_have_target and df[must_have_target].nunique(dropna=False) < 2:
            raise ValueError("В целевой колонке меньше двух различных значений")
        if df.isna().all(axis=0).any():
            warnings.append("Есть полностью пустые колонки")
        if df.duplicated().any():
            warnings.append(f"Обнаружены полные дубликаты строк: {int(df.duplicated().sum())}")
        return df, warnings

    def _write_json(self, filename: str, payload: dict[str, Any]) -> str:
        path = self.working_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        return str(path)

    def _write_text(self, filename: str, content: str) -> str:
        path = self.working_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path)

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    def _event_path(self) -> Path:
        return self.working_dir / self.name.lower().replace("agent", "") / "events.jsonl"

    def log_event(self, event_type: str, **payload: Any) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": self.name,
            "event_type": event_type,
            **payload,
        }
        self._events.append(event)
        path = self._event_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def retrieve_context(self, query: str, k: int = 3, cell_type_filter: str | None = None) -> dict[str, Any]:
        if self.rag_client is None:
            evidence = {
                "query": query,
                "hits": [],
                "available": False,
                "backend": "disabled",
                "error": "rag client not configured",
            }
        else:
            evidence = self.rag_client.retrieve(query=query, k=k, cell_type_filter=cell_type_filter).to_dict()
        self.log_event(
            "rag_lookup",
            query=query,
            hits=len(evidence.get("hits", [])),
            available=evidence.get("available", False),
            error=evidence.get("error"),
        )
        return evidence

    def timed(self, fn, *args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        return result, time.perf_counter() - start

    def build_message(self, recipient: str, reason: str, payload: dict[str, Any], message_type: str = "handoff") -> dict[str, Any]:
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            reason=reason,
            payload=payload,
        )
        self.log_event("message_created", recipient=recipient, message_type=message_type, payload_keys=sorted(payload.keys()))
        return message.to_dict()

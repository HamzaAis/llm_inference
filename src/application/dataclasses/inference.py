from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.domain.enums import InferenceStatus


@dataclass(frozen=True, slots=True)
class InferenceCreateRequest:
    query: str | None
    images: list[str] | None
    guided_json: dict | None = None


@dataclass(frozen=True, slots=True)
class InferenceRecord:
    id: int
    status: InferenceStatus
    query: str | None
    images: list[str] | None
    output: str
    guided_json: dict | None
    latency_ms: float | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class InferenceDraft:
    status: InferenceStatus
    query: str | None
    images: list[str] | None
    output: str
    guided_json: dict | None
    latency_ms: float


@dataclass(frozen=True, slots=True)
class InferencePage:
    items: list[InferenceRecord]
    page: int
    page_size: int
    total: int

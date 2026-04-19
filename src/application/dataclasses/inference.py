from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.application.dataclasses.image import ImageUpload


@dataclass(frozen=True, slots=True)
class InferenceCreateRequest:
    prompt: str
    image: ImageUpload | None
    max_new_tokens: int


@dataclass(frozen=True, slots=True)
class InferenceRecord:
    id: int
    prompt: str
    response: str
    image_relative_path: str | None
    image_filename: str | None
    image_mime: str | None
    max_new_tokens: int
    latency_ms: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class InferenceDraft:
    prompt: str
    response: str
    image_relative_path: str | None
    image_filename: str | None
    image_mime: str | None
    max_new_tokens: int
    latency_ms: int


@dataclass(frozen=True, slots=True)
class InferencePage:
    items: list[InferenceRecord]
    page: int
    page_size: int
    total: int

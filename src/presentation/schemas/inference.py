from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from src.application.dataclasses.inference import (
    InferencePage,
    InferenceRecord,
)


_PREVIEW_MAX_CHARS = 160


class InferenceCreateForm(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1, max_length=32_000)
    max_new_tokens: int = Field(..., ge=1, le=8_192)


class InferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    prompt: str
    response: str
    has_image: bool
    image_url: str | None
    image_filename: str | None
    image_mime: str | None
    max_new_tokens: int
    latency_ms: int
    created_at: datetime

    @classmethod
    def from_record(cls, record: InferenceRecord) -> "InferenceResponse":
        has_image = record.image_relative_path is not None
        return cls(
            id=record.id,
            prompt=record.prompt,
            response=record.response,
            has_image=has_image,
            image_url=f"/inferences/{record.id}/image" if has_image else None,
            image_filename=record.image_filename,
            image_mime=record.image_mime,
            max_new_tokens=record.max_new_tokens,
            latency_ms=record.latency_ms,
            created_at=record.created_at,
        )


class InferenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    prompt_preview: str
    response_preview: str
    has_image: bool
    image_url: str | None
    created_at: datetime
    latency_ms: int

    @classmethod
    def from_record(cls, record: InferenceRecord) -> "InferenceSummary":
        has_image = record.image_relative_path is not None
        return cls(
            id=record.id,
            prompt_preview=_preview(record.prompt),
            response_preview=_preview(record.response),
            has_image=has_image,
            image_url=f"/inferences/{record.id}/image" if has_image else None,
            created_at=record.created_at,
            latency_ms=record.latency_ms,
        )


class InferenceListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[InferenceSummary]
    page: int
    page_size: int
    total: int

    @classmethod
    def from_page(cls, page: InferencePage) -> "InferenceListResponse":
        return cls(
            items=[InferenceSummary.from_record(r) for r in page.items],
            page=page.page,
            page_size=page.page_size,
            total=page.total,
        )


def _preview(text: str) -> str:
    cleaned = text.strip().replace("\n", " ")
    if len(cleaned) <= _PREVIEW_MAX_CHARS:
        return cleaned
    return cleaned[: _PREVIEW_MAX_CHARS - 1].rstrip() + "\u2026"

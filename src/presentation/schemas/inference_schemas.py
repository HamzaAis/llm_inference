from __future__ import annotations

import base64
from datetime import datetime
from io import BytesIO
from PIL import Image

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.application.dataclasses.inference import (
    InferencePage,
    InferenceRecord,
)
from src.infrastructure.config.settings import get_settings


_PREVIEW_MAX_CHARS = 160


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(None, description="Text query for the model")
    images: list[str] | None = Field(
        default=None,
        description="List of base64-encoded images",
        max_length=3
    )
    guided_json: dict | None = Field(
        default=None,
        description="JSON schema for structured output"
    )

    @field_validator('query')
    @classmethod
    def validate_query_length(cls, v):
        settings = get_settings()
        if v is not None and len(v) > settings.max_query_length:
            raise ValueError(f"Query length exceeds maximum of {settings.max_query_length} characters")
        return v

    @field_validator('images')
    @classmethod
    def validate_images(cls, v):
        if v is None:
            return v

        settings = get_settings()
        if len(v) > settings.max_images:
            raise ValueError(f"Maximum {settings.max_images} images allowed")

        max_size_bytes = settings.max_image_mb * 1024 * 1024

        for idx, img_b64 in enumerate(v):
            try:
                img_data = base64.b64decode(img_b64)

                if len(img_data) > max_size_bytes:
                    raise ValueError(f"Image {idx + 1} exceeds maximum size of {settings.max_image_mb}MB")

                img = Image.open(BytesIO(img_data))

                if img.format and img.format.upper() not in ['PNG', 'JPEG', 'JPG']:
                    raise ValueError(f"Image {idx + 1} must be PNG, JPEG, or JPG format")

            except ValueError:
                raise
            except Exception as e:
                raise ValueError(f"Invalid base64 image at index {idx}: {str(e)}")

        return v

    @field_validator('query')
    @classmethod
    def validate_query_or_images(cls, v, info):
        if v is None and (not info.data.get('images') or len(info.data.get('images', [])) == 0):
            raise ValueError("Either query or images must be provided")
        return v


class InferenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    output: str
    latency_ms: float | None

    @classmethod
    def from_record(cls, record: InferenceRecord) -> "InferenceResponse":
        return cls(
            id=record.id,
            output=record.output,
            latency_ms=record.latency_ms,
        )


class InferenceDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    query: str | None
    images: list[str] | None
    output: str
    guided_json: dict | None
    latency_ms: float | None
    created_at: datetime

    @classmethod
    def from_record(cls, record: InferenceRecord) -> "InferenceDetail":
        return cls(
            id=record.id,
            query=record.query,
            images=record.images,
            output=record.output,
            guided_json=record.guided_json,
            latency_ms=record.latency_ms,
            created_at=record.created_at,
        )


class InferenceSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    query_preview: str | None
    output_preview: str
    has_images: bool
    created_at: datetime
    latency_ms: float | None

    @classmethod
    def from_record(cls, record: InferenceRecord) -> "InferenceSummary":
        return cls(
            id=record.id,
            query_preview=_preview(record.query) if record.query else None,
            output_preview=_preview(record.output),
            has_images=record.images is not None and len(record.images) > 0,
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

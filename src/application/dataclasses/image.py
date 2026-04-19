from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImageUpload:
    content: bytes
    filename: str
    mime_type: str


@dataclass(frozen=True, slots=True)
class StoredImage:
    relative_path: str
    absolute_path: str
    filename: str
    mime_type: str
    size_bytes: int

from __future__ import annotations

from pathlib import Path

from src.application.dataclasses.image import ImageUpload, StoredImage
from src.domain.repositories.image_file_repository import ImageFileRepository


class ImageFileService:
    def __init__(
        self,
        repository: ImageFileRepository,
        allowed_mimes: tuple[str, ...],
        max_bytes: int,
    ) -> None:
        self._repository = repository
        self._allowed_mimes = allowed_mimes
        self._max_bytes = max_bytes

    def store(self, upload: ImageUpload) -> StoredImage:
        self._validate(upload)
        return self._repository.save(upload)

    def resolve_absolute_path(self, relative_path: str) -> Path | None:
        return self._repository.absolute_path_for(relative_path)

    def _validate(self, upload: ImageUpload) -> None:
        if upload.mime_type not in self._allowed_mimes:
            raise UnsupportedImageMimeError(upload.mime_type, self._allowed_mimes)
        if len(upload.content) == 0:
            raise EmptyImageError()
        if len(upload.content) > self._max_bytes:
            raise ImageTooLargeError(len(upload.content), self._max_bytes)


class UnsupportedImageMimeError(ValueError):
    def __init__(self, mime: str, allowed: tuple[str, ...]) -> None:
        super().__init__(f"unsupported image mime '{mime}', allowed: {', '.join(allowed)}")
        self.mime = mime
        self.allowed = allowed


class EmptyImageError(ValueError):
    def __init__(self) -> None:
        super().__init__("uploaded image is empty")


class ImageTooLargeError(ValueError):
    def __init__(self, size: int, limit: int) -> None:
        super().__init__(f"image size {size} bytes exceeds limit {limit} bytes")
        self.size = size
        self.limit = limit

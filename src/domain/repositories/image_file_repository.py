from __future__ import annotations

import uuid
from pathlib import Path

from src.application.dataclasses.image import ImageUpload, StoredImage


_MIME_EXTENSIONS: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}


class ImageFileRepository:
    def __init__(self, files_dir: Path) -> None:
        self._files_dir = files_dir
        self._files_dir.mkdir(parents=True, exist_ok=True)

    def save(self, upload: ImageUpload) -> StoredImage:
        extension = _MIME_EXTENSIONS.get(upload.mime_type, Path(upload.filename).suffix or ".bin")
        relative_name = f"{uuid.uuid4().hex}{extension}"
        absolute_path = self._files_dir / relative_name
        absolute_path.write_bytes(upload.content)
        return StoredImage(
            relative_path=relative_name,
            absolute_path=str(absolute_path),
            filename=upload.filename,
            mime_type=upload.mime_type,
            size_bytes=len(upload.content),
        )

    def absolute_path_for(self, relative_path: str) -> Path | None:
        candidate = (self._files_dir / relative_path).resolve()
        files_root = self._files_dir.resolve()
        if files_root not in candidate.parents and candidate != files_root:
            return None
        if not candidate.is_file():
            return None
        return candidate

    def delete(self, relative_path: str) -> bool:
        target = self.absolute_path_for(relative_path)
        if target is None:
            return False
        target.unlink(missing_ok=True)
        return True

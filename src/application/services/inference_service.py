from __future__ import annotations

from pathlib import Path

from src.application.dataclasses.generation import GenerationRequest
from src.application.dataclasses.inference import (
    InferenceCreateRequest,
    InferenceDraft,
    InferencePage,
    InferenceRecord,
)
from src.application.services.image_file_service import ImageFileService
from src.application.services.model_service import ModelService
from src.domain.repositories.inference_repository import InferenceRepository


class InferenceService:
    def __init__(
        self,
        repository: InferenceRepository,
        image_service: ImageFileService,
        model_service: ModelService,
    ) -> None:
        self._repository = repository
        self._image_service = image_service
        self._model_service = model_service

    async def create(self, request: InferenceCreateRequest) -> InferenceRecord:
        stored_image = None
        image_absolute_path: str | None = None
        if request.image is not None:
            stored_image = self._image_service.store(request.image)
            image_absolute_path = stored_image.absolute_path

        generation = await self._model_service.generate(
            GenerationRequest(
                prompt=request.prompt,
                image_absolute_path=image_absolute_path,
                max_new_tokens=request.max_new_tokens,
            )
        )

        draft = InferenceDraft(
            prompt=request.prompt,
            response=generation.text,
            image_relative_path=stored_image.relative_path if stored_image else None,
            image_filename=stored_image.filename if stored_image else None,
            image_mime=stored_image.mime_type if stored_image else None,
            max_new_tokens=request.max_new_tokens,
            latency_ms=generation.latency_ms,
        )
        return await self._repository.create(draft)

    async def get(self, inference_id: int) -> InferenceRecord | None:
        return await self._repository.get_by_id(inference_id)

    async def list_page(self, page: int, page_size: int) -> InferencePage:
        return await self._repository.list_paginated(page=page, page_size=page_size)

    def resolve_image_path(self, relative_path: str) -> Path | None:
        return self._image_service.resolve_absolute_path(relative_path)

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from src.application.dataclasses.inference import (
    InferenceCreateRequest,
    InferenceDraft,
    InferencePage,
    InferenceRecord,
)
from src.application.utils.onnx_client import OnnxClient
from src.application.utils.image_processor import ImageProcessor
from src.domain.repositories.inference_repository import InferenceRepository
from src.domain.enums import InferenceStatus
from src.infrastructure.config.settings import get_settings


class InferenceService:
    def __init__(
        self,
        repository: InferenceRepository,
        onnx_client: OnnxClient,
        image_processor: ImageProcessor,
    ) -> None:
        self._repository = repository
        self._onnx_client = onnx_client
        self._image_processor = image_processor

    async def create(self, request: InferenceCreateRequest) -> InferenceRecord:
        start_time = time.time()

        try:
            preprocessed_images = None
            if request.images:
                preprocessed_images = await self._image_processor.preprocess_images_async(request.images)

            settings = get_settings()

            output = await self._onnx_client.generate(
                query=request.query,
                images=preprocessed_images,
                guided_json=request.guided_json,
                max_new_tokens=settings.default_max_new_tokens,
            )

            latency_ms = (time.time() - start_time) * 1000

            draft = InferenceDraft(
                status=InferenceStatus.COMPLETED,
                query=request.query,
                images=preprocessed_images,
                output=output,
                guided_json=request.guided_json,
                latency_ms=latency_ms,
            )
            return await self._repository.create(draft)

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_output = f"Error: {str(e)}"

            draft = InferenceDraft(
                status=InferenceStatus.FAILED,
                query=request.query,
                images=request.images,
                output=error_output,
                guided_json=request.guided_json,
                latency_ms=latency_ms,
            )
            record = await self._repository.create(draft)
            raise

    async def get(self, inference_id: int) -> InferenceRecord | None:
        return await self._repository.get_by_id(inference_id)

    async def list_page(self, page: int, page_size: int) -> InferencePage:
        return await self._repository.list_paginated(page=page, page_size=page_size)

    async def delete_by_id(self, inference_id: int) -> bool:
        return await self._repository.delete_by_id(inference_id)

    async def bulk_delete(self, older_than_days: int) -> tuple[int, datetime]:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        deleted_count = await self._repository.delete_older_than(cutoff_date)
        return deleted_count, cutoff_date

    async def get_metrics(self) -> dict:
        total_runs = await self._repository.count_all()
        successful_runs = await self._repository.count_successful()
        failed_runs = total_runs - successful_runs

        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0

        avg_latency = await self._repository.get_average_latency() or 0
        min_latency = await self._repository.get_min_latency() or 0
        max_latency = await self._repository.get_max_latency() or 0

        return {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate": round(success_rate, 2),
            "average_latency_ms": round(avg_latency, 2),
            "min_latency_ms": round(min_latency, 2),
            "max_latency_ms": round(max_latency, 2)
        }

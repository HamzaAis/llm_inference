from __future__ import annotations

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.application.dataclasses.inference import (
    InferenceDraft,
    InferencePage,
    InferenceRecord,
)
from src.domain.models.inference import Inference


class InferenceRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, draft: InferenceDraft) -> InferenceRecord:
        entity = Inference(
            prompt=draft.prompt,
            response=draft.response,
            image_path=draft.image_relative_path,
            image_filename=draft.image_filename,
            image_mime=draft.image_mime,
            max_new_tokens=draft.max_new_tokens,
            latency_ms=draft.latency_ms,
        )
        self._session.add(entity)
        await self._session.flush()
        await self._session.refresh(entity)
        await self._session.commit()
        return self._to_record(entity)

    async def get_by_id(self, inference_id: int) -> InferenceRecord | None:
        entity = await self._session.get(Inference, inference_id)
        if entity is None:
            return None
        return self._to_record(entity)

    async def list_paginated(self, page: int, page_size: int) -> InferencePage:
        offset = (page - 1) * page_size
        items_stmt = (
            select(Inference)
            .order_by(desc(Inference.created_at))
            .offset(offset)
            .limit(page_size)
        )
        total_stmt = select(func.count()).select_from(Inference)

        items_result = await self._session.execute(items_stmt)
        total_result = await self._session.execute(total_stmt)

        items = [self._to_record(row) for row in items_result.scalars().all()]
        total = int(total_result.scalar_one())
        return InferencePage(
            items=items,
            page=page,
            page_size=page_size,
            total=total,
        )

    @staticmethod
    def _to_record(entity: Inference) -> InferenceRecord:
        return InferenceRecord(
            id=entity.id,
            prompt=entity.prompt,
            response=entity.response,
            image_relative_path=entity.image_path,
            image_filename=entity.image_filename,
            image_mime=entity.image_mime,
            max_new_tokens=entity.max_new_tokens,
            latency_ms=entity.latency_ms,
            created_at=entity.created_at,
        )

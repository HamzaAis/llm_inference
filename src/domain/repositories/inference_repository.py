from __future__ import annotations

import json
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
            query=draft.query,
            images=json.dumps(draft.images) if draft.images else None,
            output=draft.output,
            guided_json=json.dumps(draft.guided_json) if draft.guided_json else None,
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

    async def count_all(self) -> int:
        stmt = select(func.count()).select_from(Inference)
        result = await self._session.execute(stmt)
        return int(result.scalar_one())

    async def count_successful(self) -> int:
        stmt = select(func.count()).select_from(Inference).where(
            ~Inference.output.startswith("Error:")
        )
        result = await self._session.execute(stmt)
        return int(result.scalar_one())

    async def get_average_latency(self) -> float | None:
        stmt = select(func.avg(Inference.latency_ms)).where(Inference.latency_ms.isnot(None))
        result = await self._session.execute(stmt)
        return result.scalar_one()

    async def get_min_latency(self) -> float | None:
        stmt = select(func.min(Inference.latency_ms)).where(Inference.latency_ms.isnot(None))
        result = await self._session.execute(stmt)
        return result.scalar_one()

    async def get_max_latency(self) -> float | None:
        stmt = select(func.max(Inference.latency_ms)).where(Inference.latency_ms.isnot(None))
        result = await self._session.execute(stmt)
        return result.scalar_one()

    @staticmethod
    def _to_record(entity: Inference) -> InferenceRecord:
        return InferenceRecord(
            id=entity.id,
            query=entity.query,
            images=json.loads(entity.images) if entity.images else None,
            output=entity.output,
            guided_json=json.loads(entity.guided_json) if entity.guided_json else None,
            latency_ms=entity.latency_ms,
            created_at=entity.created_at,
        )

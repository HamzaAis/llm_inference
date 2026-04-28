from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, Float, Text, String, Index
from sqlalchemy.orm import Mapped, mapped_column

from src.domain.entities.base import Base
from src.domain.enums import InferenceStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Inference(Base):
    __tablename__ = "inferences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=InferenceStatus.PENDING,
        index=True,
    )
    query: Mapped[str | None] = mapped_column(Text, nullable=True)
    images: Mapped[str | None] = mapped_column(Text, nullable=True)
    output: Mapped[str] = mapped_column(Text, nullable=False)
    guided_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        index=True,
    )

    __table_args__ = (
        Index('idx_created_at_desc', created_at.desc()),
        Index('idx_latency_created', latency_ms, created_at),
        Index('idx_status_created', status, created_at),
    )

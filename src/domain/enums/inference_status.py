from __future__ import annotations

from enum import StrEnum


class InferenceStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

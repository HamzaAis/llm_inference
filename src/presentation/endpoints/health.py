from __future__ import annotations

from fastapi import APIRouter

from src.infrastructure.config.dependency import ModelServiceDep, SettingsDep
from src.presentation.schemas.common import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(model_service: ModelServiceDep, settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded,
        model_name=settings.model_name,
    )

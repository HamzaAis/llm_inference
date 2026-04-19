from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.application.dataclasses.generation import SamplingProfile
from src.application.services.image_file_service import ImageFileService
from src.application.services.inference_service import InferenceService
from src.application.services.model_service import ModelService
from src.domain.repositories.image_file_repository import ImageFileRepository
from src.domain.repositories.inference_repository import InferenceRepository
from src.infrastructure.config.database import get_session
from src.infrastructure.config.settings import Settings, get_settings


_model_service: ModelService | None = None


def build_model_service(settings: Settings) -> ModelService:
    text_profile = SamplingProfile(
        temperature=settings.text_temperature,
        top_p=settings.text_top_p,
        top_k=settings.text_top_k,
        min_p=settings.text_min_p,
        presence_penalty=settings.text_presence_penalty,
        repeat_penalty=settings.text_repeat_penalty,
    )
    vl_profile = SamplingProfile(
        temperature=settings.vl_temperature,
        top_p=settings.vl_top_p,
        top_k=settings.vl_top_k,
        min_p=settings.vl_min_p,
        presence_penalty=settings.vl_presence_penalty,
        repeat_penalty=settings.vl_repeat_penalty,
    )
    return ModelService(
        model_name=settings.model_name,
        hf_cache_dir=settings.hf_cache_dir,
        provider=settings.model_provider,
        precision=settings.model_precision,
        verbose=settings.model_verbose,
        image_max_side=settings.image_max_side,
        text_sampling=text_profile,
        vl_sampling=vl_profile,
        vl_system_prompt=settings.vl_system_prompt,
    )


def set_model_service(service: ModelService) -> None:
    global _model_service
    _model_service = service


def clear_model_service() -> None:
    global _model_service
    _model_service = None


def provide_settings() -> Settings:
    return get_settings()


def provide_model_service() -> ModelService:
    if _model_service is None:
        raise RuntimeError("model service is not initialised")
    return _model_service


def provide_inference_repository(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> InferenceRepository:
    return InferenceRepository(session)


def provide_image_file_repository(
    settings: Annotated[Settings, Depends(provide_settings)],
) -> ImageFileRepository:
    return ImageFileRepository(settings.files_dir)


def provide_image_file_service(
    settings: Annotated[Settings, Depends(provide_settings)],
    repository: Annotated[ImageFileRepository, Depends(provide_image_file_repository)],
) -> ImageFileService:
    return ImageFileService(
        repository=repository,
        allowed_mimes=settings.allowed_image_mimes,
        max_bytes=settings.max_image_mb * 1024 * 1024,
    )


def provide_inference_service(
    repository: Annotated[InferenceRepository, Depends(provide_inference_repository)],
    image_service: Annotated[ImageFileService, Depends(provide_image_file_service)],
    model_service: Annotated[ModelService, Depends(provide_model_service)],
) -> InferenceService:
    return InferenceService(
        repository=repository,
        image_service=image_service,
        model_service=model_service,
    )


SettingsDep = Annotated[Settings, Depends(provide_settings)]
InferenceServiceDep = Annotated[InferenceService, Depends(provide_inference_service)]
ModelServiceDep = Annotated[ModelService, Depends(provide_model_service)]

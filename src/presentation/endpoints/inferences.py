from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import ValidationError

from src.application.dataclasses.image import ImageUpload
from src.application.dataclasses.inference import InferenceCreateRequest
from src.application.services.image_file_service import (
    EmptyImageError,
    ImageTooLargeError,
    UnsupportedImageMimeError,
)
from src.infrastructure.config.dependency import InferenceServiceDep, SettingsDep
from src.presentation.schemas.common import ErrorResponse
from src.presentation.schemas.inference import (
    InferenceCreateForm,
    InferenceListResponse,
    InferenceResponse,
)


router = APIRouter(prefix="/inferences", tags=["inferences"])


@router.post(
    "",
    response_model=InferenceResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def create_inference(
    inference_service: InferenceServiceDep,
    settings: SettingsDep,
    prompt: Annotated[str, Form(...)],
    image: Annotated[UploadFile | None, File()] = None,
    max_new_tokens: Annotated[int | None, Form()] = None,
) -> InferenceResponse:
    try:
        form = InferenceCreateForm(
            prompt=prompt,
            max_new_tokens=max_new_tokens
            if max_new_tokens is not None
            else settings.default_max_new_tokens,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors())

    upload_dto: ImageUpload | None = None
    if image is not None and image.filename:
        content = await image.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="uploaded image is empty",
            )
        mime = image.content_type or "application/octet-stream"
        upload_dto = ImageUpload(content=content, filename=image.filename, mime_type=mime)

    create_request = InferenceCreateRequest(
        prompt=form.prompt,
        image=upload_dto,
        max_new_tokens=form.max_new_tokens,
    )

    try:
        record = await inference_service.create(create_request)
    except UnsupportedImageMimeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except EmptyImageError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except ImageTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return InferenceResponse.from_record(record)


@router.get("", response_model=InferenceListResponse)
async def list_inferences(
    inference_service: InferenceServiceDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> InferenceListResponse:
    page_dto = await inference_service.list_page(page=page, page_size=page_size)
    return InferenceListResponse.from_page(page_dto)


@router.get(
    "/{inference_id}",
    response_model=InferenceResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_inference(
    inference_id: int,
    inference_service: InferenceServiceDep,
) -> InferenceResponse:
    record = await inference_service.get(inference_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="inference not found")
    return InferenceResponse.from_record(record)


@router.get(
    "/{inference_id}/image",
    responses={404: {"model": ErrorResponse}},
)
async def get_inference_image(
    inference_id: int,
    inference_service: InferenceServiceDep,
) -> FileResponse:
    record = await inference_service.get(inference_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="inference not found")
    if record.image_relative_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="inference has no image",
        )
    absolute_path = inference_service.resolve_image_path(record.image_relative_path)
    if absolute_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="image file is missing on disk",
        )
    return FileResponse(
        path=str(absolute_path),
        media_type=record.image_mime or "application/octet-stream",
        filename=record.image_filename or absolute_path.name,
    )

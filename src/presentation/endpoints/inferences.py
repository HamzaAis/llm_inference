from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from src.application.dataclasses.inference import InferenceCreateRequest
from src.infrastructure.config.dependency import InferenceServiceDep
from src.presentation.schemas.common import ErrorResponse
from src.presentation.schemas.inference import (
    InferenceDetail,
    InferenceListResponse,
    InferenceRequest,
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
    request: InferenceRequest,
    inference_service: InferenceServiceDep,
) -> InferenceResponse:
    create_request = InferenceCreateRequest(
        query=request.query,
        images=request.images,
        guided_json=request.guided_json,
    )

    try:
        record = await inference_service.create(create_request)
        return InferenceResponse.from_record(record)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}"
        )


@router.get("/metrics")
async def get_metrics(inference_service: InferenceServiceDep):
    return await inference_service.get_metrics()


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
    response_model=InferenceDetail,
    responses={404: {"model": ErrorResponse}},
)
async def get_inference(
    inference_id: int,
    inference_service: InferenceServiceDep,
) -> InferenceDetail:
    record = await inference_service.get(inference_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="inference not found")
    return InferenceDetail.from_record(record)

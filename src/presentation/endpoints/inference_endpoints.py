import json
import time
from typing import Annotated, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status

from src.application.dataclasses.inference import InferenceCreateRequest
from src.infrastructure.config.dependency import InferenceServiceDep, ImageProcessorDep
from src.infrastructure.config.settings import get_settings
from src.infrastructure.middleware import setup_logger
from src.presentation.schemas.common import ErrorResponse
from src.presentation.schemas.inference_schemas import (
    InferenceDetail,
    InferenceListResponse,
    InferenceRequest,
    InferenceResponse,
)

logger = setup_logger(__name__)


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
    t0 = time.perf_counter()
    logger.info("endpoint: request received query_len=%d num_images=%s",
               len(request.query or ""), len(request.images or []))

    create_request = InferenceCreateRequest(
        query=request.query,
        images=request.images,
        guided_json=request.guided_json,
    )
    logger.info("endpoint: request converted to InferenceCreateRequest elapsed_ms=%.2f",
               (time.perf_counter() - t0) * 1000)

    try:
        record = await inference_service.create(create_request)
        t_resp = time.perf_counter()
        response = InferenceResponse.from_record(record)
        logger.info("endpoint: response serialized elapsed_ms=%.2f",
                   (time.perf_counter() - t_resp) * 1000)
        logger.info("endpoint: total request duration_ms=%.2f",
                   (time.perf_counter() - t0) * 1000)
        return response
    except Exception as exc:
        logger.error("endpoint: request failed elapsed_ms=%.2f error=%s",
                    (time.perf_counter() - t0) * 1000, str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}"
        )


@router.post(
    "/multipart",
    response_model=InferenceResponse,
    status_code=status.HTTP_201_CREATED,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def create_inference_multipart(
    query: str | None = Form(default=None),
    guided_json: str | None = Form(default=None),
    images: list[bytes] = File(default=[]),
    inference_service: InferenceServiceDep = None,
    image_processor: ImageProcessorDep = None,
) -> InferenceResponse:
    t0 = time.perf_counter()
    images = images or []
    logger.info("endpoint: multipart request received query_len=%d num_images=%d",
               len(query or ""), len(images))

    try:
        settings = get_settings()

        if not query and not images:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either query or images must be provided"
            )

        if len(images) > settings.max_images:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Maximum {settings.max_images} images allowed"
            )

        t_step = time.perf_counter()
        image_bytes_list = []
        max_size_bytes = settings.max_image_mb * 1024 * 1024

        for idx, file in enumerate(images):
            if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Image {idx + 1} must be JPEG or PNG format"
                )

            file_bytes = await file.read()
            if len(file_bytes) > max_size_bytes:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Image {idx + 1} exceeds maximum size of {settings.max_image_mb}MB"
                )
            image_bytes_list.append(file_bytes)

        logger.info("endpoint: file read elapsed_ms=%.2f",
                   (time.perf_counter() - t_step) * 1000)

        t_step = time.perf_counter()
        preprocessed_images = None
        if image_bytes_list:
            preprocessed_images = await image_processor.preprocess_uploaded_images_async(image_bytes_list)
        logger.info("endpoint: image preprocessing elapsed_ms=%.2f",
                   (time.perf_counter() - t_step) * 1000)

        t_step = time.perf_counter()
        parsed_guided_json = None
        if guided_json:
            try:
                parsed_guided_json = json.loads(guided_json)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid JSON in guided_json field"
                )

        create_request = InferenceCreateRequest(
            query=query,
            images=preprocessed_images,
            guided_json=parsed_guided_json,
        )
        logger.info("endpoint: request converted to InferenceCreateRequest elapsed_ms=%.2f",
                   (time.perf_counter() - t_step) * 1000)

        record = await inference_service.create(create_request, store_images=False)

        t_resp = time.perf_counter()
        response = InferenceResponse.from_record(record)
        logger.info("endpoint: response serialized elapsed_ms=%.2f",
                   (time.perf_counter() - t_resp) * 1000)
        logger.info("endpoint: total multipart request duration_ms=%.2f",
                   (time.perf_counter() - t0) * 1000)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("endpoint: multipart request failed elapsed_ms=%.2f error=%s",
                    (time.perf_counter() - t0) * 1000, str(exc))
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


@router.delete("/{inference_id}")
async def delete_inference(
    inference_id: int,
    inference_service: InferenceServiceDep,
):
    success = await inference_service.delete_by_id(inference_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="inference not found")
    return {"message": "Inference deleted successfully"}


@router.delete("")
async def bulk_delete_inferences(
    older_than_days: Annotated[int, Query(ge=1)],
    inference_service: InferenceServiceDep,
):
    deleted_count, cutoff_date = await inference_service.bulk_delete(older_than_days)
    return {
        "deleted_count": deleted_count,
        "cutoff_date": cutoff_date
    }

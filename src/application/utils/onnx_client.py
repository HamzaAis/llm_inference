from __future__ import annotations

import base64
import tempfile
import time
from pathlib import Path
from typing import Optional

import aiofiles
import aiofiles.os
import anyio

from src.application.dataclasses.generation import GenerationRequest
from src.application.services.model_service import ModelService
from src.infrastructure.middleware import setup_logger

logger = setup_logger(__name__)


class OnnxClient:
    """
    Client wrapper for ONNX Runtime model service.
    Provides a clean interface similar to VLLMClient, handling temp file management.
    """

    def __init__(self, model_service: ModelService):
        self._model_service = model_service

    async def generate(
        self,
        query: Optional[str] = None,
        images: Optional[list[str]] = None,
        guided_json: Optional[dict] = None,
        max_new_tokens: int = 512
    ) -> str:
        """
        Generate text output from the ONNX model.

        Args:
            query: Text query for the model
            images: List of base64-encoded images (only first image is used)
            guided_json: JSON schema for structured output (not yet implemented)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text output

        Raises:
            OnnxGenerationError: If generation fails
        """
        t0 = time.perf_counter()
        temp_image_path = None

        try:
            if images and len(images) > 0:
                t_step = time.perf_counter()
                temp_image_path = await self._save_temp_image(images[0])
                logger.info("onnx_client: save temp image elapsed_ms=%.2f",
                           (time.perf_counter() - t_step) * 1000)

            t_step = time.perf_counter()
            generation_request = GenerationRequest(
                prompt=query or "",
                image_absolute_path=str(temp_image_path) if temp_image_path else None,
                max_new_tokens=max_new_tokens,
            )
            logger.info("onnx_client: build generation request elapsed_ms=%.2f",
                       (time.perf_counter() - t_step) * 1000)

            t_step = time.perf_counter()
            result = await self._model_service.generate(generation_request)
            logger.info("onnx_client: model generate elapsed_ms=%.2f",
                       (time.perf_counter() - t_step) * 1000)

            logger.info("onnx_client: generate total elapsed_ms=%.2f",
                       (time.perf_counter() - t0) * 1000)
            return result.text

        except Exception as e:
            logger.error("onnx_client: generate failed elapsed_ms=%.2f error=%s",
                        (time.perf_counter() - t0) * 1000, str(e))
            raise OnnxGenerationError(f"ONNX generation failed: {str(e)}")

        finally:
            if temp_image_path and await aiofiles.os.path.exists(str(temp_image_path)):
                t_step = time.perf_counter()
                await aiofiles.os.remove(str(temp_image_path))
                logger.info("onnx_client: cleanup temp image elapsed_ms=%.2f",
                           (time.perf_counter() - t_step) * 1000)

    async def _save_temp_image(self, image_b64: str) -> Path:
        """Save base64 image to temporary file asynchronously."""
        img_data = base64.b64decode(image_b64)

        def _create_temp() -> str:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            path = temp_file.name
            temp_file.close()
            return path

        temp_path = Path(await anyio.to_thread.run_sync(_create_temp))

        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(img_data)

        return temp_path


class OnnxGenerationError(Exception):
    """Raised when ONNX model generation fails."""
    pass

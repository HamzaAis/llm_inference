from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Optional

import aiofiles

from src.application.dataclasses.generation import GenerationRequest
from src.application.services.model_service import ModelService


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
        temp_image_path = None

        try:
            if images and len(images) > 0:
                temp_image_path = await self._save_temp_image(images[0])

            generation_request = GenerationRequest(
                prompt=query or "",
                image_absolute_path=str(temp_image_path) if temp_image_path else None,
                max_new_tokens=max_new_tokens,
            )

            result = await self._model_service.generate(generation_request)

            return result.text

        except Exception as e:
            raise OnnxGenerationError(f"ONNX generation failed: {str(e)}")

        finally:
            if temp_image_path and temp_image_path.exists():
                temp_image_path.unlink()

    async def _save_temp_image(self, image_b64: str) -> Path:
        """Save base64 image to temporary file asynchronously."""
        img_data = base64.b64decode(image_b64)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        temp_path = Path(temp_file.name)
        temp_file.close()

        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(img_data)

        return temp_path


class OnnxGenerationError(Exception):
    """Raised when ONNX model generation fails."""
    pass

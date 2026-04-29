import base64
import time
from io import BytesIO
from PIL import Image
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.infrastructure.middleware import setup_logger

logger = setup_logger(__name__)


class ImageProcessor:
    def __init__(self, max_width: int = 1280, max_height: int = 720, dpi: int = 150, jpeg_quality: int = 80):
        self.max_width = max_width
        self.max_height = max_height
        self.dpi = dpi
        self.jpeg_quality = jpeg_quality

    def preprocess_image(self, image_b64: str) -> str:
        t0 = time.perf_counter()

        t_step = time.perf_counter()
        img_data = base64.b64decode(image_b64)
        img = Image.open(BytesIO(img_data))
        logger.info("image_processor: decode+open image elapsed_ms=%.2f",
                   (time.perf_counter() - t_step) * 1000)

        t_step = time.perf_counter()
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        logger.info("image_processor: color conversion elapsed_ms=%.2f",
                   (time.perf_counter() - t_step) * 1000)

        t_step = time.perf_counter()
        img.thumbnail((self.max_width, self.max_height), Image.Resampling.LANCZOS)
        logger.info("image_processor: thumbnail resize elapsed_ms=%.2f",
                   (time.perf_counter() - t_step) * 1000)

        t_step = time.perf_counter()
        output_buffer = BytesIO()
        img.save(
            output_buffer,
            format='JPEG',
            quality=self.jpeg_quality,
            optimize=True,
            dpi=(self.dpi, self.dpi)
        )
        logger.info("image_processor: jpeg encode+save elapsed_ms=%.2f",
                   (time.perf_counter() - t_step) * 1000)

        t_step = time.perf_counter()
        optimized_b64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        logger.info("image_processor: base64 encode elapsed_ms=%.2f",
                   (time.perf_counter() - t_step) * 1000)

        logger.info("image_processor: preprocess_image total elapsed_ms=%.2f",
                   (time.perf_counter() - t0) * 1000)
        return optimized_b64

    async def preprocess_images_async(self, images: Optional[list[str]]) -> Optional[list[str]]:
        if not images:
            return images

        t0 = time.perf_counter()
        logger.info("image_processor: preprocessing %d images", len(images))

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=3) as executor:
            tasks = [
                loop.run_in_executor(executor, self.preprocess_image, img)
                for img in images
            ]
            results = await asyncio.gather(*tasks)

        logger.info("image_processor: all images preprocessed total_elapsed_ms=%.2f",
                   (time.perf_counter() - t0) * 1000)
        return results

    def preprocess_uploaded_image(self, image_bytes: bytes) -> str:
        """Process binary image bytes from multipart upload, return base64 for internal use."""
        t0 = time.perf_counter()

        t_step = time.perf_counter()
        img = Image.open(BytesIO(image_bytes))
        logger.info("image_processor: open uploaded image elapsed_ms=%.2f",
                   (time.perf_counter() - t_step) * 1000)

        t_step = time.perf_counter()
        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        logger.info("image_processor: color conversion elapsed_ms=%.2f",
                   (time.perf_counter() - t_step) * 1000)

        t_step = time.perf_counter()
        img.thumbnail((self.max_width, self.max_height), Image.Resampling.LANCZOS)
        logger.info("image_processor: thumbnail resize elapsed_ms=%.2f",
                   (time.perf_counter() - t_step) * 1000)

        t_step = time.perf_counter()
        output_buffer = BytesIO()
        img.save(
            output_buffer,
            format='JPEG',
            quality=self.jpeg_quality,
            optimize=True,
            dpi=(self.dpi, self.dpi)
        )
        logger.info("image_processor: jpeg encode+save elapsed_ms=%.2f",
                   (time.perf_counter() - t_step) * 1000)

        t_step = time.perf_counter()
        optimized_b64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        logger.info("image_processor: base64 encode elapsed_ms=%.2f",
                   (time.perf_counter() - t_step) * 1000)

        logger.info("image_processor: preprocess_uploaded_image total elapsed_ms=%.2f",
                   (time.perf_counter() - t0) * 1000)
        return optimized_b64

    async def preprocess_uploaded_images_async(self, image_files: list[bytes]) -> list[str]:
        """Async batch processing of uploaded binary images."""
        if not image_files:
            return []

        t0 = time.perf_counter()
        logger.info("image_processor: preprocessing %d uploaded images", len(image_files))

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=3) as executor:
            tasks = [
                loop.run_in_executor(executor, self.preprocess_uploaded_image, img_bytes)
                for img_bytes in image_files
            ]
            results = await asyncio.gather(*tasks)

        logger.info("image_processor: all uploaded images preprocessed total_elapsed_ms=%.2f",
                   (time.perf_counter() - t0) * 1000)
        return results

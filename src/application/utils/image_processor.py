import base64
from io import BytesIO
from PIL import Image
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor


class ImageProcessor:
    def __init__(self, max_width: int = 1280, max_height: int = 720, dpi: int = 150, jpeg_quality: int = 80):
        self.max_width = max_width
        self.max_height = max_height
        self.dpi = dpi
        self.jpeg_quality = jpeg_quality

    def preprocess_image(self, image_b64: str) -> str:
        img_data = base64.b64decode(image_b64)
        img = Image.open(BytesIO(img_data))

        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        img.thumbnail((self.max_width, self.max_height), Image.Resampling.LANCZOS)

        output_buffer = BytesIO()
        img.save(
            output_buffer,
            format='JPEG',
            quality=self.jpeg_quality,
            optimize=True,
            dpi=(self.dpi, self.dpi)
        )

        optimized_b64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')
        return optimized_b64

    async def preprocess_images_async(self, images: Optional[list[str]]) -> Optional[list[str]]:
        if not images:
            return images

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=3) as executor:
            tasks = [
                loop.run_in_executor(executor, self.preprocess_image, img)
                for img in images
            ]
            return await asyncio.gather(*tasks)

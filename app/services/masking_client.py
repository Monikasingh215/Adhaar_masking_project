# app/services/masking_client.py
import httpx
from pathlib import Path
from typing import Optional
from ..core.logging import get_logger

logger = get_logger(__name__)

class MaskingClient:
    def __init__(self, endpoint: str = "http://localhost:8000/mask/"):
        self.endpoint = endpoint

    async def mask_image(self, image_path: Path) -> Optional[Path]:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                with image_path.open("rb") as f:
                    files = {"file": (image_path.name, f, "image/jpeg")}
                    resp = await client.post(self.endpoint, files=files)
                    resp.raise_for_status()           # ← raises on 4xx/5xx
                    # Save masked image
                    masked_path = image_path.with_name(
                        f"{image_path.stem}_masked{image_path.suffix}"
                    )
                    masked_path.write_bytes(resp.content)
                    return masked_path
        except Exception as e:
            logger.error(f"[MaskingClient] Masking failed for {image_path.name}: {e}")
            return None

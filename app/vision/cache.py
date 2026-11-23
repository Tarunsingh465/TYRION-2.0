"""
Persistent Disk & Memory Cache for Multimodal Vision & OCR Intelligence.
TYRION-2.0 — Final-Year Academic & Portfolio Project

Eliminates redundant vision API calls by indexing image findings by SHA-256 byte hash.
Ensures identical images uploaded repeatedly consume 0 Gemini API credits or quota tokens.
"""

import hashlib
import json
import threading
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from app.core.config import settings
from app.core.logger import get_logger
from app.schemas.vision import VisionExtractionResult

logger = get_logger()


class ImageVisionCache:
    """
    Two-tier (In-Memory LRU + Persistent Disk) Cache for Vision & OCR Extractions.
    Uses SHA-256 of original raw bytes paired with model identifier.
    """

    _instance: Optional["ImageVisionCache"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        if getattr(self, "_initialized", False):
            return

        self.cache_dir = Path(cache_dir or getattr(settings, "VISION_CACHE_DIR", ".cache/vision"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, VisionExtractionResult] = {}
        self._rw_lock = threading.Lock()
        self._initialized = True
        logger.info(f"ImageVisionCache initialized at '{self.cache_dir}'.")

    @staticmethod
    def compute_sha256(raw_bytes: bytes) -> str:
        """Calculate hexadecimal SHA-256 digest of binary image payload."""
        return hashlib.sha256(raw_bytes).hexdigest()

    def _build_key(self, image_hash: str, model_name: str, prompt_version: str = "v1") -> str:
        safe_model = model_name.replace("/", "_").replace(":", "_").replace("-", "_")
        return f"{image_hash}_{safe_model}_{prompt_version}"

    def get(
        self,
        raw_bytes: bytes,
        model_name: str,
        prompt_version: str = "v1",
    ) -> Optional[VisionExtractionResult]:
        """
        Look up cached vision extraction by raw bytes and model name.
        Checks in-memory cache first, then disk storage.
        """
        image_hash = self.compute_sha256(raw_bytes)
        cache_key = self._build_key(image_hash, model_name, prompt_version)

        with self._rw_lock:
            # 1. Tier 1: In-memory hit
            if cache_key in self._memory_cache:
                logger.info(f"Vision Cache HIT (Memory): {image_hash[:12]}... (0 API tokens consumed)")
                res = self._memory_cache[cache_key]
                meta = res.metadata.copy() if res.metadata else {}
                meta["from_cache"] = True
                meta["image_sha256"] = image_hash
                return res.model_copy(update={"metadata": meta})

            # 2. Tier 2: Disk cache hit
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                    res = VisionExtractionResult(**data)
                    meta = res.metadata.copy() if res.metadata else {}
                    meta["from_cache"] = True
                    meta["image_sha256"] = image_hash
                    res = res.model_copy(update={"metadata": meta})
                    self._memory_cache[cache_key] = res
                    logger.info(f"Vision Cache HIT (Disk): {image_hash[:12]}... (0 API tokens consumed)")
                    return res
                except Exception as e:
                    logger.warning(f"Failed to read disk cache for {cache_key}: {e}")

        return None

    def set(
        self,
        raw_bytes: bytes,
        model_name: str,
        result: VisionExtractionResult,
        prompt_version: str = "v1",
    ) -> None:
        """
        Persist a vision extraction result to memory and disk.
        """
        image_hash = self.compute_sha256(raw_bytes)
        cache_key = self._build_key(image_hash, model_name, prompt_version)

        meta = result.metadata.copy() if result.metadata else {}
        meta["image_sha256"] = image_hash
        persisted_result = result.model_copy(update={"metadata": meta})

        with self._rw_lock:
            self._memory_cache[cache_key] = persisted_result
            cache_file = self.cache_dir / f"{cache_key}.json"
            try:
                cache_file.write_text(persisted_result.model_dump_json(indent=2), encoding="utf-8")
                logger.debug(f"Vision Cache STORED: {image_hash[:12]}... -> {cache_file.name}")
            except Exception as e:
                logger.warning(f"Could not write cache file {cache_file.name}: {e}")

    def clear(self) -> None:
        """Flush in-memory and disk cache (used primarily in tests)."""
        with self._rw_lock:
            self._memory_cache.clear()
            for f in self.cache_dir.glob("*.json"):
                try:
                    f.unlink()
                except Exception:
                    pass


def get_vision_cache() -> ImageVisionCache:
    """Helper factory returning singleton ImageVisionCache."""
    return ImageVisionCache()

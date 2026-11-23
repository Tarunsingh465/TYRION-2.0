"""
Image Preprocessing and Validation Module for TYRION-2.0.
Provides format verification, magic byte checks, EXIF orientation correction,
resizing, decompression bomb safeguards, and multi-frame TIFF handling.
"""

import io
from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import settings
from app.core.constants import SUPPORTED_IMAGE_EXTENSIONS
from app.core.logger import get_logger

logger = get_logger()

# Protect against decompression bomb exploits (e.g., zip bombs or gigantic canvases)
Image.MAX_IMAGE_PIXELS = settings.IMAGE_MAX_PIXELS

# Standard magic byte signatures for image validation
MAGIC_SIGNATURES = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"GIF87a": "gif",
    b"GIF89a": "gif",
    b"RIFF": "webp",
    b"BM": "bmp",
    b"II*\x00": "tiff",  # Little-endian TIFF
    b"MM\x00*": "tiff",  # Big-endian TIFF
}

MAX_EDGE_DIMENSION = 2048  # Downscale dimension for very large images while preserving aspect ratio


class ImagePreprocessingError(ValueError):
    """Raised when an image fails format, size, or integrity checks."""
    pass


class ImagePreprocessor:
    """
    Validates, sanitizes, and normalizes images before vision API or OCR ingestion.
    """

    def __init__(
        self,
        max_size_mb: int = settings.IMAGE_MAX_SIZE_MB,
        max_pixels: int = settings.IMAGE_MAX_PIXELS,
        max_edge: int = MAX_EDGE_DIMENSION,
    ):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_pixels = max_pixels
        self.max_edge = max_edge

    def validate_file_signature(self, raw_bytes: bytes) -> str:
        """
        Validate raw bytes against known image file signatures (magic bytes).
        Prevents executable files masquerading as images.
        """
        if len(raw_bytes) < 8:
            raise ImagePreprocessingError("File data is too small to be a valid image.")

        for sig, fmt in MAGIC_SIGNATURES.items():
            if raw_bytes.startswith(sig):
                if fmt == "webp" and len(raw_bytes) >= 12 and raw_bytes[8:12] != b"WEBP":
                    continue
                return fmt

        # Check for webp specifically (RIFF....WEBP)
        if raw_bytes[:4] == b"RIFF" and len(raw_bytes) >= 12 and raw_bytes[8:12] == b"WEBP":
            return "webp"

        raise ImagePreprocessingError(
            "Invalid image signature. The file header does not match any recognized image format."
        )

    def load_and_preprocess(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
    ) -> List[Tuple[Image.Image, dict]]:
        """
        Load an image from a filepath, raw bytes, or PIL Image.
        Applies:
        - Format & size validation
        - EXIF transpose orientation fix
        - Multi-frame TIFF extraction
        - High-resolution downscaling preserving aspect ratio
        - Conversion to RGB mode

        Returns a list of tuples: [(processed_pil_image, frame_metadata), ...]
        """
        raw_bytes: Optional[bytes] = None
        source_name = "in_memory"

        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            source_name = path.name
            ext = path.suffix.lower()
            if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                raise ImagePreprocessingError(
                    f"Unsupported image extension '{ext}'. Supported: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
                )

            if not path.exists():
                raise ImagePreprocessingError(f"Image file not found: {path}")

            file_size = path.stat().st_size
            if file_size > self.max_size_bytes:
                raise ImagePreprocessingError(
                    f"Image file size ({file_size / (1024*1024):.1f} MB) exceeds maximum allowed limit ({settings.IMAGE_MAX_SIZE_MB} MB)."
                )

            raw_bytes = path.read_bytes()
        elif isinstance(image_input, bytes):
            raw_bytes = image_input
            if len(raw_bytes) > self.max_size_bytes:
                raise ImagePreprocessingError(
                    f"Image byte size exceeds maximum allowed limit ({settings.IMAGE_MAX_SIZE_MB} MB)."
                )
        elif isinstance(image_input, Image.Image):
            return self._process_pil_image(image_input, {"source": source_name, "frame_index": 0})
        else:
            raise ImagePreprocessingError(f"Unsupported image input type: {type(image_input)}")

        # Verify magic bytes
        self.validate_file_signature(raw_bytes)

        try:
            pil_image = Image.open(io.BytesIO(raw_bytes))
        except UnidentifiedImageError as e:
            raise ImagePreprocessingError(f"Cannot identify or decode image: {e}")
        except Exception as e:
            raise ImagePreprocessingError(f"Corrupted or invalid image data: {e}")

        # Check total pixel count safeguard
        w, h = pil_image.size
        if w * h > self.max_pixels:
            raise ImagePreprocessingError(
                f"Image dimensions ({w}x{h} = {w*h} pixels) exceed maximum pixel limit ({self.max_pixels})."
            )

        # Multi-frame handling (e.g. multi-page TIFFs)
        frames: List[Tuple[Image.Image, dict]] = []
        try:
            n_frames = getattr(pil_image, "n_frames", 1)
        except Exception:
            n_frames = 1

        for frame_idx in range(n_frames):
            try:
                pil_image.seek(frame_idx)
                frame_copy = pil_image.copy()
                meta = {
                    "source": source_name,
                    "frame_index": frame_idx,
                    "total_frames": n_frames,
                    "original_format": pil_image.format or "JPEG",
                    "original_width": w,
                    "original_height": h,
                }
                frames.extend(self._process_pil_image(frame_copy, meta))
            except EOFError:
                break
            except Exception as e:
                logger.warning(f"Error processing frame {frame_idx} of image '{source_name}': {e}")
                if not frames:
                    raise ImagePreprocessingError(f"Failed to read image frames: {e}")

        return frames

    def _process_pil_image(self, img: Image.Image, meta: dict) -> List[Tuple[Image.Image, dict]]:
        """
        Apply EXIF transposition, RGB conversion, and proportional resizing to a single PIL frame.
        """
        # 1. Correct orientation using EXIF tags
        try:
            img = ImageOps.exif_transpose(img) or img
        except Exception as e:
            logger.debug(f"EXIF transpose skipped: {e}")

        # 2. Convert to RGB (handles RGBA, Palette, CMYK, Grayscale)
        if img.mode != "RGB":
            # For RGBA or images with transparency, composite onto white background
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                bg.paste(img, mask=img.split()[3])
                img = bg
            else:
                img = img.convert("RGB")

        # 3. Proportional resizing if oversized
        w, h = img.size
        if max(w, h) > self.max_edge:
            if w >= h:
                new_w = self.max_edge
                new_h = int(h * (self.max_edge / w))
            else:
                new_h = self.max_edge
                new_w = int(w * (self.max_edge / h))

            logger.info(f"Resizing oversized image from {w}x{h} to {new_w}x{new_h} preserving aspect ratio.")
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            meta["resized"] = True
            meta["processed_width"] = new_w
            meta["processed_height"] = new_h
        else:
            meta["resized"] = False
            meta["processed_width"] = w
            meta["processed_height"] = h

        return [(img, meta)]

    @staticmethod
    def to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
        """Helper to convert a PIL Image into JPEG bytes for API transport."""
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

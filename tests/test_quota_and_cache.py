"""
Unit and Integration Tests for API Usage Optimization, Caching & Quota Protection.
TYRION-2.0 — Final-Year Academic & Portfolio Project

Verifies:
1. SHA-256 byte caching for images (identical images trigger 0 API calls).
2. Centralized daily budget ceiling enforcement and local OCR fallback.
3. Per-upload maximum vision call limits.
4. HTTP 429 rate limit backoff and finite loop termination.
5. Embedding chunk caching (unchanged text chunks make 0 Google API calls).
6. Selective image filtering (ignoring banners, tiny icons, and solid-color graphics).
7. Budget monitoring API endpoint telemetry.
"""

import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import settings
from app.ingestion.embedder import EmbeddingCache, RateLimitedEmbeddings
from app.main import app
from app.schemas.vision import VisionExtractionResult
from app.vision.cache import ImageVisionCache
from app.vision.client import GeminiVisionProvider, LocalOcrProvider, VisionClient
from app.vision.preprocessor import ImagePreprocessor
from app.vision.quota_manager import VisionQuotaManager


class TestQuotaAndCache(unittest.TestCase):
    """
    Test suite validating client-side quota protection, caching, and rate-limit safety.
    """

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(tempfile.mkdtemp(prefix="tyrion_quota_test_"))
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        self.cache = ImageVisionCache(cache_dir=self.test_dir / "vision_cache")
        self.cache.clear()

        self.quota = VisionQuotaManager(state_file=self.test_dir / "quota_state.json")
        self.quota.reset_all_counters_for_testing()

    # =========================================================================
    # 1. Image Vision SHA-256 Caching
    # =========================================================================

    def test_image_vision_cache_hit_and_persistence(self):
        """Verify image results are persisted by SHA-256 and retrieved without API calls."""
        img_bytes = b"\x89PNG\r\n\x1a\n" + b"mock_png_image_bytes_12345"
        mock_result = VisionExtractionResult(
            ocr_text="SAMPLE TAG: V-200",
            visual_description="Gas turbine isolation valve",
            technical_labels=["V-200"],
            processing_status="success",
        )

        # 1. Miss initially
        self.assertIsNone(self.cache.get(img_bytes, model_name="gemini-3.5-flash-lite"))

        # 2. Store in cache
        self.cache.set(img_bytes, model_name="gemini-3.5-flash-lite", result=mock_result)

        # 3. Hit from cache
        cached = self.cache.get(img_bytes, model_name="gemini-3.5-flash-lite")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.ocr_text, "SAMPLE TAG: V-200")
        self.assertTrue(cached.metadata.get("from_cache"))

    def test_uploading_same_image_twice_makes_zero_duplicate_api_calls(self):
        """Verify uploading the same image twice serves the second call 100% from cache."""
        img = Image.new("RGB", (150, 150), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw_jpeg = buf.getvalue()

        mock_extraction = VisionExtractionResult(
            ocr_text="PRESSURE TRANSMITTER: 10.0 bar",
            visual_description="Pressure sensor",
            technical_labels=["PT-10"],
            processing_status="success",
        )

        client = VisionClient.__new__(VisionClient)
        client.cloud_provider = MagicMock()
        client.cloud_provider.model_name = "gemini-3.5-flash-lite"
        client.cloud_provider.api_key = "mock_key"
        client.cloud_provider.analyze_image.return_value = mock_extraction
        client.local_provider = MagicMock()
        client.local_provider._tesseract_available = False
        client.cache = self.cache
        client.quota_manager = self.quota

        # First call: invokes cloud provider
        res1 = client.analyze_image(img, prompt_hint="technical diagram", metadata={"normalized_jpeg_bytes": raw_jpeg})
        self.assertEqual(client.cloud_provider.analyze_image.call_count, 1)
        self.assertEqual(self.quota.cached_hits, 0)
        self.assertEqual(self.quota.cloud_vision_calls, 1)

        # Second call with identical image: hits cache, 0 new API calls!
        res2 = client.analyze_image(img, prompt_hint="technical diagram", metadata={"normalized_jpeg_bytes": raw_jpeg})
        self.assertEqual(client.cloud_provider.analyze_image.call_count, 1)  # Still 1!
        self.assertTrue(res2.metadata.get("from_cache"))
        self.assertEqual(self.quota.cached_hits, 1)

    # =========================================================================
    # 2. Daily Budget & Per-Upload Call Controls
    # =========================================================================

    def test_daily_call_budget_enforcement(self):
        """Verify cloud vision calls are blocked when daily budget ceiling is reached."""
        self.quota.daily_budget = 5
        self.quota.calls_today = 5  # Budget fully consumed

        can_call, reason = self.quota.can_make_cloud_call(for_upload=True)
        self.assertFalse(can_call)
        self.assertIn("daily vision budget reached", reason.lower())

    def test_per_upload_limit_enforcement(self):
        """Verify per-upload ceiling prevents single large documents from draining daily quota."""
        self.quota.max_calls_per_upload = 3
        self.quota.current_upload_calls = 3

        can_call, reason = self.quota.can_make_cloud_call(for_upload=True)
        self.assertFalse(can_call)
        self.assertIn("Per-upload cloud vision limit reached", reason)

    # =========================================================================
    # 3. Rate-Limit 429 Backoff & Finite Retries
    # =========================================================================

    def test_rate_limit_429_backoff_and_finite_termination(self):
        """Verify provider does NOT loop infinitely on 429 and terminates at retry limit."""
        provider = GeminiVisionProvider(api_key="mock_key", model_name="gemini-3.5-flash-lite")
        provider.max_retries = 2

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for model")
        provider._client = mock_client

        test_img = Image.new("RGB", (100, 100), color="white")

        with patch.object(settings, "ALLOW_CLOUD_VISION_PROCESSING", True):
            with patch("time.sleep") as mock_sleep:
                res = provider.analyze_image(test_img)

        # Verified: retried exactly max_retries times and returned rate_limited
        self.assertEqual(mock_client.models.generate_content.call_count, 2)
        self.assertEqual(res.processing_status, "rate_limited")
        self.assertIn("rate-limited", res.visual_description)
        self.assertGreater(mock_sleep.call_count, 0)

    # =========================================================================
    # 4. Embedding Chunk Caching
    # =========================================================================

    def test_embedding_chunk_caching(self):
        """Verify unchanged text chunks are served from disk cache with 0 API calls."""
        emb_cache = EmbeddingCache(cache_dir=str(self.test_dir / "embed_cache"))
        emb_cache.clear()
        mock_base = MagicMock()
        mock_base.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        rate_emb = RateLimitedEmbeddings(base_embeddings=mock_base)
        rate_emb.cache = emb_cache

        chunks = ["Pump specification chunk A", "Centrifugal pump chunk B"]

        # First pass: calls base embeddings
        vectors1 = rate_emb.embed_documents(chunks)
        self.assertEqual(mock_base.embed_documents.call_count, 1)
        self.assertEqual(len(vectors1), 2)

        # Second pass with identical chunks: served 100% from cache!
        vectors2 = rate_emb.embed_documents(chunks)
        self.assertEqual(mock_base.embed_documents.call_count, 1)  # Still 1!
        self.assertEqual(vectors1, vectors2)

    # =========================================================================
    # 5. Selective Image Filtering
    # =========================================================================

    def test_selective_filtering_decorative_and_banner_graphics(self):
        """Verify tiny icons, divider banners, and low-entropy graphics are skipped."""
        # 1. Tiny icon (30x30) -> Ignorable
        tiny_img = Image.new("RGB", (30, 30), color="gray")
        is_dec, reason = ImagePreprocessor.is_decorative_or_ignorable(tiny_img)
        self.assertTrue(is_dec)
        self.assertIn("smaller than minimum", reason)

        # 2. Extreme aspect banner (1200x150, aspect=8:1) -> Ignorable
        banner_img = Image.new("RGB", (1200, 150), color="blue")
        is_dec, reason = ImagePreprocessor.is_decorative_or_ignorable(banner_img)
        self.assertTrue(is_dec)
        self.assertIn("Extreme aspect ratio", reason)

        # 3. Solid uniform block (150x150 solid white) -> Ignorable
        solid_img = Image.new("RGB", (150, 150), color=(255, 255, 255))
        is_dec, reason = ImagePreprocessor.is_decorative_or_ignorable(solid_img)
        self.assertTrue(is_dec)
        self.assertIn("Low visual variance", reason)

        # 4. Authentic diagram with high variance and standard dimensions -> Valid
        diagram_img = Image.new("RGB", (300, 300), color="white")
        # Draw some content across pixels
        pixels = diagram_img.load()
        for i in range(100):
            pixels[i, i] = (0, 0, 0)
            pixels[i + 50, i] = (255, 0, 0)
        is_dec, _ = ImagePreprocessor.is_decorative_or_ignorable(diagram_img)
        self.assertFalse(is_dec)

    # =========================================================================
    # 6. Budget API Endpoint
    # =========================================================================

    def test_budget_api_endpoint(self):
        """Verify GET /api/v1/vision/budget returns real-time quota telemetry."""
        response = self.client.get("/api/v1/vision/budget")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("calls_today", data)
        self.assertIn("daily_budget", data)
        self.assertIn("remaining_budget", data)
        self.assertIn("cached_hits", data)
        self.assertIn("local_ocr_calls", data)
        self.assertIn("cloud_vision_calls", data)


if __name__ == "__main__":
    unittest.main()

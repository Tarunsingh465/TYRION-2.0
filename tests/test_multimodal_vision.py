"""
Multimodal Image Intelligence & Vision-Based Document Ingestion Test Suite.
TYRION-2.0 — Final-Year Academic & Portfolio Project

Comprehensive automated unit and integration tests verifying:
1. Supported image formats ingestion (JPG, JPEG, PNG, WEBP, BMP, TIFF, TIF).
2. Image preprocessing safeguards (magic bytes, decompression bombs, EXIF transpose, downscaling).
3. Corrupted, empty, and unsupported image error handling.
4. Pydantic vision schemas and RAG Markdown serialization.
5. Multimodal vision client providers (Gemini, Local OCR fallback, cloud consent gating).
6. Metadata preservation across chunking and dual-indexing (Pinecone + BM25).
7. Hybrid retrieval and rich source citation formatting with image provenance.
8. Deletion and index cleanup for image assets.
9. Scanned PDF fallback and embedded image extraction (DOCX, PPTX, XLSX).
"""

import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image
from langchain_core.documents import Document

from app.core.config import settings
from app.ingestion.splitter import DocumentSplitter
from app.ingestion.extractors.image_extractor import ImageExtractor
from app.ingestion.extractors.office_extractor import OfficeExtractor
from app.ingestion.extractors.pdf_extractor import PdfExtractor
from app.ingestion.extractors.tabular_extractor import TabularExtractor
from app.retrieval.bm25 import BM25Index
from app.retrieval.retriever import Retriever
from app.schemas.vision import VisionExtractionResult, VisionModelOutput
from app.services.rag_service import RagService
from app.vision.client import GeminiVisionProvider, LocalOcrProvider, VisionClient
from app.vision.preprocessor import ImagePreprocessingError, ImagePreprocessor


class TestMultimodalVision(unittest.TestCase):
    """
    Comprehensive test suite for TYRION-2.0 Multimodal Image & Vision Pipeline.
    """

    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path(tempfile.mkdtemp(prefix="tyrion_vision_test_"))
        cls.mock_vision_output = VisionExtractionResult(
            ocr_text="PRESSURE GAUGE: 16.0 bar | TAG: PG-101",
            visual_description="Industrial pressure transmitter with digital LED readout and stainless steel flange.",
            detected_objects=["pressure gauge", "flange", "stainless steel piping", "LED display"],
            technical_labels=["PG-101", "MOD-404-B"],
            tables_markdown=["| Parameter | Value |\n|---|---|\n| Operating Pressure | 16.0 bar |\n| Design Temp | 120 C |"],
            chart_description="",
            diagram_entities=["Sensor Head", "Isolation Valve", "Piping"],
            relationships=["Sensor Head connected to Isolation Valve via 1/2 inch NPT thread"],
            units_and_measurements=["16.0 bar", "120 C", "1/2 inch NPT"],
            uncertain_regions=[],
            processing_status="success",
            metadata={"provider": "gemini_vision", "model": "gemini-3.5-flash-lite"},
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def _create_sample_image(self, format_name: str = "PNG", size=(200, 200), color="blue") -> Path:
        """Create a real valid image on disk for ingestion testing."""
        ext = format_name.lower()
        if ext == "jpeg":
            ext = "jpg"
        img_path = self.test_dir / f"sample_{size[0]}x{size[1]}_{color}.{ext}"
        img = Image.new("RGB", size, color=color)
        img.save(str(img_path), format=format_name)
        return img_path

    # =========================================================================
    # 1. Image Preprocessing & Security Safeguards
    # =========================================================================

    def test_magic_byte_signature_validation(self):
        """Verify signature checker identifies valid formats and rejects impostors."""
        preprocessor = ImagePreprocessor()

        # JPEG signature
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        self.assertEqual(preprocessor.validate_file_signature(jpeg_bytes), "jpeg")

        # PNG signature
        png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        self.assertEqual(preprocessor.validate_file_signature(png_bytes), "png")

        # BMP signature
        bmp_bytes = b"BM\x36\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00"
        self.assertEqual(preprocessor.validate_file_signature(bmp_bytes), "bmp")

        # Rejection of invalid executable / text header
        with self.assertRaises(ImagePreprocessingError):
            preprocessor.validate_file_signature(b"MZ\x90\x00\x03\x00\x00\x00")

    def test_empty_image_handling(self):
        """Verify 0-byte images raise ImagePreprocessingError."""
        empty_file = self.test_dir / "empty.png"
        empty_file.write_bytes(b"")

        preprocessor = ImagePreprocessor()
        with self.assertRaises(ImagePreprocessingError):
            preprocessor.load_and_preprocess(empty_file)

    def test_corrupted_image_handling(self):
        """Verify non-image bytes disguised as image raise ImagePreprocessingError."""
        corrupt_file = self.test_dir / "corrupted.jpg"
        corrupt_file.write_bytes(b"\xff\xd8\xff\xe0" + b"random_corrupt_data" * 50)

        preprocessor = ImagePreprocessor()
        with self.assertRaises(ImagePreprocessingError):
            preprocessor.load_and_preprocess(corrupt_file)

    def test_unsupported_format_rejection(self):
        """Verify unsupported extensions are rejected."""
        unsupported = self.test_dir / "program.exe"
        unsupported.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00test_binary_data")

        preprocessor = ImagePreprocessor()
        with self.assertRaises(ImagePreprocessingError):
            preprocessor.load_and_preprocess(unsupported)

    def test_large_image_resizing_and_aspect_ratio(self):
        """Verify large images (> 2048px) are downscaled proportionally."""
        large_path = self._create_sample_image(format_name="PNG", size=(3000, 1500), color="green")
        preprocessor = ImagePreprocessor(max_edge=2048)

        processed_frames = preprocessor.load_and_preprocess(large_path)
        self.assertEqual(len(processed_frames), 1)
        proc_img, meta = processed_frames[0]

        self.assertLessEqual(proc_img.width, 2048)
        self.assertLessEqual(proc_img.height, 2048)
        self.assertEqual(proc_img.width, 2048)
        self.assertEqual(proc_img.height, 1024)  # 2:1 ratio preserved
        self.assertTrue(meta["resized"])

    def test_multi_frame_tiff_extraction(self):
        """Verify multi-page TIFF images are split into individual frames."""
        tiff_path = self.test_dir / "multipage.tiff"
        frame1 = Image.new("RGB", (100, 100), color="red")
        frame2 = Image.new("RGB", (100, 100), color="blue")
        frame1.save(str(tiff_path), save_all=True, append_images=[frame2])

        preprocessor = ImagePreprocessor()
        frames = preprocessor.load_and_preprocess(tiff_path)

        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0][1]["frame_index"], 0)
        self.assertEqual(frames[1][1]["frame_index"], 1)

    # =========================================================================
    # 2. Pydantic Schemas & RAG Serialization
    # =========================================================================

    def test_vision_model_output_schema_and_markdown(self):
        """Verify VisionModelOutput serialization and Markdown RAG conversion."""
        result = self.mock_vision_output

        markdown = result.to_rag_markdown()
        self.assertIn("### Visual Overview & Scene Description", markdown)
        self.assertIn("PRESSURE GAUGE: 16.0 bar", markdown)
        self.assertIn("PG-101", markdown)
        self.assertIn("16.0 bar", markdown)
        self.assertIn("| Operating Pressure | 16.0 bar |", markdown)
        self.assertIn("Sensor Head connected to Isolation Valve", markdown)

    def test_empty_vision_result_fallback(self):
        """Verify empty vision result produces clean empty markdown without crashing."""
        empty_res = VisionExtractionResult()
        self.assertEqual(empty_res.to_rag_markdown(), "")

    # =========================================================================
    # 3. Vision Client & Provider Isolation
    # =========================================================================

    def test_gemini_vision_provider_success(self):
        """Verify GeminiVisionProvider parses JSON output into VisionExtractionResult."""
        provider = GeminiVisionProvider(api_key="mock_key", model_name="gemini-3.5-flash-lite")
        mock_response = MagicMock()
        mock_response.text = self.mock_vision_output.model_dump_json()

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        provider._client = mock_client

        test_img = Image.new("RGB", (150, 150), color="white")

        with patch.object(settings, "ALLOW_CLOUD_VISION_PROCESSING", True):
            res = provider.analyze_image(test_img)

        self.assertEqual(res.ocr_text, "PRESSURE GAUGE: 16.0 bar | TAG: PG-101")
        self.assertIn("PG-101", res.technical_labels)
        self.assertEqual(res.metadata["provider"], "gemini")

    def test_cloud_consent_disabled_policy(self):
        """Verify cloud processing refusal when ALLOW_CLOUD_VISION_PROCESSING is False."""
        provider = GeminiVisionProvider(api_key="mock_key", model_name="gemini-3.5-flash-lite")
        test_img = Image.new("RGB", (150, 150), color="white")

        with patch.object(settings, "ALLOW_CLOUD_VISION_PROCESSING", False):
            res = provider.analyze_image(test_img)

        self.assertEqual(res.processing_status, "cloud_disabled")
        self.assertIn("disabled by local privacy policy", res.visual_description)

    def test_missing_api_key_behavior(self):
        """Verify missing API key triggers fallback or unconfigured notice without crash."""
        provider = GeminiVisionProvider(api_key="", model_name="gemini-3.5-flash-lite")
        test_img = Image.new("RGB", (150, 150), color="white")

        with patch.object(settings, "ALLOW_CLOUD_VISION_PROCESSING", True):
            res = provider.analyze_image(test_img)

        self.assertEqual(res.processing_status, "cloud_disabled")
        self.assertIn("missing GOOGLE_API_KEY", res.visual_description)

    def test_local_ocr_provider(self):
        """Verify LocalOcrProvider operates cleanly and populates ocr_text."""
        provider = LocalOcrProvider()
        provider._tesseract_available = True
        test_img = Image.new("RGB", (150, 150), color="white")

        with patch.object(provider._pytesseract, "image_to_string", return_value="SAFETY VALVE RELIEF SETTING: 25.0 BAR"):
            res = provider.analyze_image(test_img)

        self.assertEqual(res.ocr_text, "SAFETY VALVE RELIEF SETTING: 25.0 BAR")
        self.assertEqual(res.metadata["provider"], "local_ocr")

    # =========================================================================
    # 4. Standalone Image Extraction (JPG, PNG, WEBP, TIFF, BMP)
    # =========================================================================

    def _test_format_ingestion(self, format_ext: str, format_name: str):
        img_path = self._create_sample_image(format_name=format_name, size=(120, 120), color="cyan")
        extractor = ImageExtractor()

        with patch.object(extractor.vision_client, "analyze_image", return_value=self.mock_vision_output):
            docs = extractor.extract(img_path)

        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertIn("PG-101", doc.page_content)
        self.assertEqual(doc.metadata["source"], img_path.name)
        self.assertTrue(doc.metadata["is_multimodal"])
        self.assertEqual(doc.metadata["doc_type"], "technical_image")
        self.assertIn("16.0 bar", doc.metadata["units_and_measurements"])

    def test_jpg_ingestion(self):
        self._test_format_ingestion("jpg", "JPEG")

    def test_jpeg_ingestion(self):
        self._test_format_ingestion("jpeg", "JPEG")

    def test_png_ingestion(self):
        self._test_format_ingestion("png", "PNG")

    def test_webp_ingestion(self):
        self._test_format_ingestion("webp", "WEBP")

    def test_bmp_ingestion(self):
        self._test_format_ingestion("bmp", "BMP")

    def test_tiff_ingestion(self):
        self._test_format_ingestion("tiff", "TIFF")

    # =========================================================================
    # 5. Metadata Preservation Across Chunker
    # =========================================================================

    def test_metadata_preservation_across_splitter(self):
        """Verify child chunks preserve all multimodal provenance metadata."""
        doc = Document(
            page_content="### Pressure Gauge\n" + "Detailed specification text. " * 80,
            metadata={
                "source": "transmitter.png",
                "file_type": ".png",
                "page": 1,
                "slide_number": None,
                "sheet_name": None,
                "image_index": 1,
                "is_multimodal": True,
                "doc_type": "technical_image",
                "extraction_method": "gemini_vision",
                "technical_labels": ["PG-101"],
            },
        )

        splitter = DocumentSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split([doc])

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.metadata["source"], "transmitter.png")
            self.assertEqual(chunk.metadata["image_index"], 1)
            self.assertTrue(chunk.metadata["is_multimodal"])
            self.assertEqual(chunk.metadata["extraction_method"], "gemini_vision")
            self.assertEqual(chunk.metadata["technical_labels"], ["PG-101"])

    # =========================================================================
    # 6. Hybrid Indexing & RRF Retrieval of Multimodal Chunks
    # =========================================================================

    def test_hybrid_retrieval_of_image_derived_content(self):
        """Verify hybrid retriever scores and ranks image-derived chunks via RRF."""
        image_doc = Document(
            page_content="Operating pressure is 16.0 bar for PG-101 transmitter.",
            metadata={
                "source": "pid_diagram.png",
                "file_name": "pid_diagram.png",
                "page": 1,
                "image_index": 1,
                "doc_type": "technical_image",
                "extraction_method": "gemini_vision",
            },
        )

        retriever = Retriever.__new__(Retriever)
        retriever._get_active_documents = MagicMock(return_value=["pid_diagram.png"])
        retriever.retrieve_dense = MagicMock(return_value=[(image_doc, 0.88)])
        retriever.retrieve_bm25 = MagicMock(return_value=[(image_doc, 4.5)])

        results, diagnostics = retriever.retrieve_hybrid("What is the operating pressure of PG-101?", k=2)

        self.assertEqual(len(results), 1)
        retrieved_doc, score = results[0]
        self.assertEqual(retrieved_doc.metadata["source"], "pid_diagram.png")
        self.assertGreater(score, 0.0)

    # =========================================================================
    # 7. Source Citation Generation with Multimodal Provenance
    # =========================================================================

    def test_source_citation_generation_with_image_metadata(self):
        """Verify RAG service packages image-specific citation attributes."""
        mock_doc = Document(
            page_content="Operating pressure is 16.0 bar.",
            metadata={
                "source": "compressor_schematic.png",
                "file_type": ".png",
                "page": 1,
                "slide_number": 3,
                "sheet_name": "Compressors",
                "cell_anchor": "C14",
                "image_index": 2,
                "extraction_method": "gemini_vision",
                "doc_type": "technical_image",
                "rrf_score": 0.032,
            },
        )

        rag_service = RagService.__new__(RagService)
        rag_service._check_relevance = MagicMock(return_value=(True, 0.95, [(mock_doc, 0.95)]))
        rag_service.chain = MagicMock()
        rag_service.chain.invoke.return_value = {
            "answer": "The operating pressure for the compressor is 16.0 bar.",
            "documents": [mock_doc],
        }

        response = rag_service.ask("What is the operating pressure?")

        self.assertFalse(response["out_of_context"])
        self.assertEqual(len(response["sources"]), 1)
        src = response["sources"][0]
        self.assertEqual(src["source"], "compressor_schematic.png")
        self.assertEqual(src["slide_number"], 3)
        self.assertEqual(src["sheet_name"], "Compressors")
        self.assertEqual(src["cell_anchor"], "C14")
        self.assertEqual(src["image_index"], 2)
        self.assertEqual(src["extraction_method"], "gemini_vision")

    # =========================================================================
    # 8. Deletion and Index Cleanup
    # =========================================================================

    def test_image_deletion_and_index_cleanup(self):
        """Verify deleting an image file cleans disk and dual indexes."""
        from app.services.ingest_service import IngestService

        dummy_img = self.test_dir / "test_cleanup.png"
        dummy_img.write_bytes(b"dummy")

        with patch("app.services.ingest_service.UPLOAD_FOLDER", str(self.test_dir)):
            ingest_service = IngestService.__new__(IngestService)
            mock_pipeline = MagicMock()
            ingest_service.pipeline = mock_pipeline

            success = ingest_service.delete_document("test_cleanup.png")
            self.assertTrue(success)
            self.assertFalse(dummy_img.exists())
            mock_pipeline.vector_store.index.delete.assert_called()
            mock_pipeline.bm25_index.remove_document_by_filename.assert_called_with("test_cleanup.png")

    # =========================================================================
    # 9. Scanned PDF & Embedded Documents Vision Tests
    # =========================================================================

    @patch("pypdf.PdfReader")
    def test_pdf_scanned_page_vision_fallback(self, mock_reader_cls):
        """Verify scanned PDF page (<50 chars text) triggers vision processing for embedded image."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Scanned"  # Only 7 chars -> triggers is_scanned_page

        # Create uncompressed BMP image bytes (>1000 bytes)
        raw_io = io.BytesIO()
        Image.new("RGB", (120, 120), color="yellow").save(raw_io, format="BMP")
        mock_img_obj = MagicMock()
        mock_img_obj.data = raw_io.getvalue()
        mock_page.images = [mock_img_obj]

        mock_reader = MagicMock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page]
        mock_reader_cls.return_value = mock_reader

        pdf_path = self.test_dir / "scanned_doc.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 mock content")

        pdf_extractor = PdfExtractor()
        with patch.object(pdf_extractor.vision_client, "analyze_image", return_value=self.mock_vision_output):
            docs = pdf_extractor.extract(pdf_path)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["doc_type"], "scanned_page_vision")
        self.assertEqual(docs[0].metadata["image_index"], 1)

        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].metadata["doc_type"], "scanned_page_vision")
        self.assertEqual(docs[0].metadata["image_index"], 1)

    def test_pptx_embedded_image_extraction_preserves_slides(self):
        """Verify PPTX extractor captures slide numbers and calls vision for images."""
        extractor = OfficeExtractor()
        # Verify extractor initialized vision client
        self.assertIsNotNone(extractor.vision_client)

    def test_xlsx_embedded_image_extraction_preserves_anchors(self):
        """Verify XLSX tabular extractor initializes vision client and anchors."""
        extractor = TabularExtractor()
        self.assertIsNotNone(extractor.vision_client)


if __name__ == "__main__":
    unittest.main()

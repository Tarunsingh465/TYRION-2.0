"""
Document and Multimodal Ingestion Loader for TYRION-2.0.
Routes incoming documents, spreadsheets, presentations, and technical images
to the appropriate specialized extractor.
"""

from pathlib import Path
from typing import List, Union

from langchain_core.documents import Document

from app.core.config import settings
from app.core.constants import SUPPORTED_FILE_EXTENSIONS
from app.core.logger import get_logger
from app.ingestion.extractors.image_extractor import ImageExtractor
from app.ingestion.extractors.office_extractor import OfficeExtractor
from app.ingestion.extractors.pdf_extractor import PdfExtractor
from app.ingestion.extractors.tabular_extractor import TabularExtractor
from app.ingestion.extractors.text_extractor import TextExtractor

logger = get_logger()


class DocumentLoader:
    """
    Routes documents and technical images to specialized extractors.

    Supported Formats:
    - PDF: Native offline text extraction with multimodal vision for scanned pages and embedded diagrams.
    - Multimodal Images: JPG, JPEG, PNG, WEBP, BMP, TIFF, TIF via Gemini Vision and local OCR fallback.
    - Office: DOCX, PPTX (hierarchical sections, tables, slide-by-slide indexing, embedded pictures).
    - Tabular: CSV, XLSX (row context preservation, multi-sheet, embedded spreadsheet charts).
    - Text: TXT, MD (clean UTF-8 parsing).
    """

    def __init__(self):
        pdf_extractor = PdfExtractor()
        image_extractor = ImageExtractor()
        office = OfficeExtractor()
        tabular = TabularExtractor()
        text = TextExtractor()

        self._extractors = {
            # PDF
            ".pdf": pdf_extractor,

            # Multimodal Images (Vision & OCR)
            ".png": image_extractor,
            ".jpg": image_extractor,
            ".jpeg": image_extractor,
            ".webp": image_extractor,
            ".bmp": image_extractor,
            ".tiff": image_extractor,
            ".tif": image_extractor,

            # Office
            ".docx": office,
            ".pptx": office,

            # Tabular
            ".csv": tabular,
            ".xlsx": tabular,

            # Text
            ".txt": text,
            ".md": text,
        }

    def load(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load a document or image file and return extracted LangChain Documents.
        """
        file_path = Path(file_path)
        self._validate(file_path)

        suffix = file_path.suffix.lower()
        extractor = self._extractors.get(suffix)

        if not extractor:
            raise ValueError(f"No extractor registered for file extension: '{suffix}'")

        logger.info(f"Loading '{file_path.name}' using {extractor.__class__.__name__}")
        documents = extractor.extract(file_path)

        # Validate extracted content
        if not documents:
            raise ValueError(f"Extractor returned 0 documents for file: {file_path.name}")

        logger.success(f"Loaded {len(documents)} document(s) from '{file_path.name}'.")
        return documents

    def _validate(self, file_path: Path):
        """
        Validate file existence and extension support.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a regular file: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_FILE_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_FILE_EXTENSIONS))}"
            )

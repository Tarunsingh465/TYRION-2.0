"""
Image and Vision Extractor for TYRION-2.0.
Extracts visible text, interprets diagrams, tables, charts, and technical objects
from image files (JPG, JPEG, PNG, WEBP, BMP, TIFF, TIF).
"""

from pathlib import Path
from typing import List, Union

from langchain_core.documents import Document

from app.core.logger import get_logger
from app.ingestion.cleaner import TextCleaner
from app.vision.client import get_vision_client
from app.vision.preprocessor import ImagePreprocessor, ImagePreprocessingError

logger = get_logger()


class ImageExtractor:
    """
    Extracts text and semantic visual understanding from standalone image files.
    Returns LangChain Document objects ready for splitting and vectorization.
    """

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.cleaner = TextCleaner()
        self.vision_client = get_vision_client()

    def extract(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Preprocess image, analyze with Vision/OCR provider, and construct
        LangChain Document objects with rich provenance metadata.
        """
        path = Path(file_path)
        logger.info(f"Extracting visual intelligence from image: {path.name}")

        try:
            frames = self.preprocessor.load_and_preprocess(path)
        except ImagePreprocessingError as e:
            logger.error(f"Image preprocessing validation failed for {path.name}: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error loading image {path.name}: {e}")
            raise ImagePreprocessingError(f"Corrupted or invalid image: {e}")

        documents: List[Document] = []

        for frame_idx, (frame_img, frame_meta) in enumerate(frames):
            frame_label = f"Frame {frame_idx + 1}/{len(frames)}" if len(frames) > 1 else "Main Image"
            logger.info(f"Analyzing {frame_label} of '{path.name}' ({frame_img.width}x{frame_img.height} px)...")

            extraction = self.vision_client.analyze_image(
                image=frame_img,
                prompt_hint=f"Industrial document or technical image: {path.name}",
                metadata=frame_meta,
            )

            rag_markdown = extraction.to_rag_markdown()
            clean_text = self.cleaner.clean(rag_markdown)

            if not clean_text.strip():
                clean_text = f"Image file '{path.name}' processed. Visual description: {extraction.visual_description}"

            page_number = frame_idx + 1 if len(frames) > 1 else 1

            doc_metadata = {
                "source": path.name,
                "file_name": path.name,
                "file_type": path.suffix.lower(),
                "page": page_number,
                "total_pages": len(frames),
                "image_index": frame_idx + 1,
                "extraction_method": extraction.metadata.get("provider", "vision_provider"),
                "vision_model": extraction.metadata.get("model", "unknown"),
                "doc_type": "technical_image",
                "is_multimodal": True,
                "has_ocr_text": bool(extraction.ocr_text.strip()),
                "has_technical_labels": bool(extraction.technical_labels),
                "has_tables": bool(extraction.tables_markdown),
                "has_charts": bool(extraction.chart_description.strip()),
                "has_diagram": bool(extraction.diagram_entities),
                "technical_labels": extraction.technical_labels[:15],
                "units_and_measurements": extraction.units_and_measurements[:15],
                "detected_objects": extraction.detected_objects[:15],
            }

            documents.append(
                Document(
                    page_content=clean_text,
                    metadata=doc_metadata,
                )
            )

        logger.success(f"Extracted {len(documents)} document(s) from image '{path.name}'.")
        return documents

"""
Local and Multimodal PDF Extractor for TYRION-2.0.
Extracts native text offline using pypdf. For scanned pages or image-rich sections,
automatically detects little/no extractable text and invokes multimodal vision / OCR.
Also extracts embedded technical diagrams and schematics with page provenance.
"""

import hashlib
import io
from pathlib import Path
from typing import List

from PIL import Image
import pypdf
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logger import get_logger
from app.ingestion.cleaner import TextCleaner
from app.vision.client import get_vision_client
from app.vision.preprocessor import ImagePreprocessor

logger = get_logger()


class PdfExtractor:
    """
    Extract text from PDFs locally using pypdf, with multimodal vision intelligence
    for scanned pages and embedded diagrams/charts.
    """

    def __init__(self):
        self.cleaner = TextCleaner()
        self.preprocessor = ImagePreprocessor()
        self.vision_client = get_vision_client()

    def extract(self, file_path: Path) -> List[Document]:
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Extracting PDF content (text + visual diagrams): {file_path.name}")
        try:
            reader = pypdf.PdfReader(str(file_path))
        except Exception as e:
            logger.error(f"Failed to open PDF '{file_path.name}': {e}")
            raise ValueError(f"Malformed or unreadable PDF: {e}")

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as e:
                logger.warning(f"Could not decrypt PDF '{file_path.name}': {e}")

        documents: List[Document] = []
        total_pages = len(reader.pages)
        seen_image_hashes: set[str] = set()

        for index, page in enumerate(reader.pages):
            page_num = index + 1
            raw_text = ""
            try:
                raw_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Error extracting text on page {page_num} of '{file_path.name}': {e}")

            text = self.cleaner.clean(raw_text)

            # 1. Normal text extraction if text exists
            if text and len(text.strip()) >= 50:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path.name,
                            "page": page_num,
                            "total_pages": total_pages,
                            "file_type": ".pdf",
                            "doc_type": "pdf_page_text",
                            "extraction_method": "pypdf",
                        },
                    )
                )

            # 2. Check for scanned page or embedded images if enabled
            if settings.ENABLE_IMAGE_ANALYSIS and hasattr(page, "images") and page.images:
                is_scanned_page = not text or len(text.strip()) < 50
                max_imgs_per_page = 4 if is_scanned_page else 2

                for img_idx, img_obj in enumerate(page.images[:max_imgs_per_page]):
                    try:
                        img_data = img_obj.data
                        if not img_data or len(img_data) < 1000:
                            # Skip tiny icons, stamps, or bullets
                            continue

                        pil_img = Image.open(io.BytesIO(img_data))
                        is_dec, reason = self.preprocessor.is_decorative_or_ignorable(pil_img, img_data)
                        if is_dec:
                            logger.debug(f"Skipping decorative PDF image on page {page_num}: {reason}")
                            continue

                        img_hash = hashlib.sha256(img_data).hexdigest()
                        if img_hash in seen_image_hashes:
                            logger.debug(f"Skipping duplicate PDF image on page {page_num} (hash={img_hash[:8]}).")
                            continue
                        seen_image_hashes.add(img_hash)

                        processed_frames = self.preprocessor.load_and_preprocess(pil_img)
                        for f_idx, (f_img, f_meta) in enumerate(processed_frames):
                            vision_res = self.vision_client.analyze_image(
                                image=f_img,
                                prompt_hint=f"PDF embedded image from {file_path.name}, Page {page_num}",
                                metadata=f_meta,
                            )

                            vision_md = vision_res.to_rag_markdown()
                            if vision_md.strip():
                                documents.append(
                                    Document(
                                        page_content=vision_md,
                                        metadata={
                                            "source": file_path.name,
                                            "page": page_num,
                                            "total_pages": total_pages,
                                            "image_index": img_idx + 1,
                                            "file_type": ".pdf",
                                            "doc_type": "scanned_page_vision" if is_scanned_page else "pdf_embedded_diagram",
                                            "extraction_method": vision_res.metadata.get("provider", "gemini-vision"),
                                            "technical_labels": vision_res.technical_labels[:10],
                                            "units_and_measurements": vision_res.units_and_measurements[:10],
                                        },
                                    )
                                )
                    except Exception as e:
                        logger.debug(f"Could not analyze image {img_idx + 1} on page {page_num}: {e}")

            elif text and len(text.strip()) < 50:
                # Small text, no images
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path.name,
                            "page": page_num,
                            "total_pages": total_pages,
                            "file_type": ".pdf",
                            "doc_type": "pdf_page_text",
                            "extraction_method": "pypdf",
                        },
                    )
                )

        if not documents:
            documents.append(
                Document(
                    page_content=f"[Scanned/Image-only PDF: No extractable embedded text found in '{file_path.name}']",
                    metadata={
                        "source": file_path.name,
                        "page": 1,
                        "total_pages": total_pages if total_pages else 1,
                        "file_type": ".pdf",
                        "doc_type": "empty_pdf",
                        "extraction_method": "none",
                    },
                )
            )

        logger.info(f"Extracted {len(documents)} document chunk(s) from '{file_path.name}'.")
        return documents

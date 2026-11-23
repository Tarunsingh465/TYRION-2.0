"""
Office Document Extractor for TYRION-2.0.
Extracts structured hierarchical text, tables, and embedded images from DOCX and PPTX.
Preserves slide numbers, image indices, and table associations.
"""

import io
from pathlib import Path
from typing import List

from PIL import Image
from docx import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from langchain_core.documents import Document
from pptx import Presentation

from app.core.config import settings
from app.core.logger import get_logger
from app.vision.client import get_vision_client
from app.vision.preprocessor import ImagePreprocessor

logger = get_logger()


class OfficeExtractor:
    """
    Structured extractor for Microsoft Office documents (.docx and .pptx).
    Extracts text, markdown tables, and embedded diagrams/photographs with provenance.
    """

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.vision_client = get_vision_client()

    def extract(self, file_path: Path) -> List[Document]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".docx":
            return self._extract_docx(file_path)
        elif suffix == ".pptx":
            return self._extract_pptx(file_path)
        else:
            raise ValueError(f"Unsupported Office document format: '{suffix}'")

    def _extract_docx(self, file_path: Path) -> List[Document]:
        try:
            doc = DocxDocument(file_path)
        except Exception as e:
            logger.error(f"Failed to open DOCX file '{file_path.name}': {e}")
            raise ValueError(f"Malformed or unreadable DOCX file: {e}")

        documents: List[Document] = []
        current_section = "General / Introduction"
        current_block: List[str] = []
        approx_page = 1
        words_count = 0

        def flush_block(section_name: str, has_tbl: bool = False):
            nonlocal words_count, approx_page
            content = "\n".join(current_block).strip()
            if content:
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": file_path.name,
                            "page": approx_page,
                            "section": section_name,
                            "has_table": has_tbl,
                            "file_type": ".docx",
                            "doc_type": "docx_section",
                        },
                    )
                )
                current_block.clear()
                words_count += len(content.split())
                if words_count > 450:
                    approx_page += 1
                    words_count = 0

        # 1. Traverse body elements in document order
        for child in doc.element.body:
            if isinstance(child, CT_P):
                p = Paragraph(child, doc)
                text = p.text.strip()
                if not text:
                    continue

                style_name = (p.style.name if p.style else "").lower()
                is_heading = style_name.startswith("heading") or (p.style and p.style.name in ["Title", "Subtitle"])

                if is_heading:
                    if current_block:
                        flush_block(current_section)
                    current_section = text
                    current_block.append(f"### {text}")
                else:
                    current_block.append(text)

            elif isinstance(child, CT_Tbl):
                tbl = Table(child, doc)
                md_table = self._table_to_markdown(tbl)
                if md_table:
                    current_block.append(f"\n{md_table}\n")
                    flush_block(current_section, has_tbl=True)

        if current_block:
            flush_block(current_section)

        # 2. Extract embedded images in DOCX if vision analysis enabled
        if settings.ENABLE_IMAGE_ANALYSIS and hasattr(doc, "part"):
            img_counter = 0
            for rel in doc.part.related_parts.values():
                if "image" in rel.content_type:
                    img_counter += 1
                    if img_counter > 8:  # Cap at 8 images per doc to avoid excessive token use
                        break
                    try:
                        raw_img_bytes = rel.blob
                        if len(raw_img_bytes) < 1500:
                            continue
                        pil_img = Image.open(io.BytesIO(raw_img_bytes))
                        if pil_img.width < 100 or pil_img.height < 100:
                            continue

                        frames = self.preprocessor.load_and_preprocess(pil_img)
                        for f_img, f_meta in frames:
                            v_res = self.vision_client.analyze_image(
                                image=f_img,
                                prompt_hint=f"DOCX embedded image {img_counter} from {file_path.name}",
                                metadata=f_meta,
                            )
                            v_md = v_res.to_rag_markdown()
                            if v_md.strip():
                                documents.append(
                                    Document(
                                        page_content=v_md,
                                        metadata={
                                            "source": file_path.name,
                                            "page": approx_page,
                                            "image_index": img_counter,
                                            "file_type": ".docx",
                                            "doc_type": "docx_embedded_image",
                                            "extraction_method": v_res.metadata.get("provider", "gemini-vision"),
                                            "technical_labels": v_res.technical_labels[:10],
                                            "units_and_measurements": v_res.units_and_measurements[:10],
                                        },
                                    )
                                )
                    except Exception as e:
                        logger.debug(f"Failed to process embedded image {img_counter} in DOCX: {e}")

        logger.info(f"Extracted {len(documents)} structured block(s) from DOCX '{file_path.name}'")
        return documents

    def _table_to_markdown(self, table: Table) -> str:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(cells):
                rows.append(cells)

        if not rows:
            return ""

        headers = rows[0]
        body = rows[1:] if len(rows) > 1 else []

        md_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in body:
            padded_row = row + [""] * (len(headers) - len(row))
            md_lines.append("| " + " | ".join(padded_row[:len(headers)]) + " |")

        return "\n".join(md_lines)

    def _extract_pptx(self, file_path: Path) -> List[Document]:
        try:
            prs = Presentation(file_path)
        except Exception as e:
            logger.error(f"Failed to open PPTX file '{file_path.name}': {e}")
            raise ValueError(f"Malformed or unreadable PPTX file: {e}")

        documents: List[Document] = []

        for slide_idx, slide in enumerate(prs.slides, start=1):
            slide_parts: List[str] = []
            has_table = False
            img_index_on_slide = 0

            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = paragraph.text.strip()
                        if text:
                            slide_parts.append(text)
                elif shape.has_table:
                    has_table = True
                    table_lines = []
                    for row in shape.table.rows:
                        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                        if any(cells):
                            table_lines.append("| " + " | ".join(cells) + " |")
                    if table_lines:
                        slide_parts.append("\n".join(table_lines))

                # Check for embedded image on slide
                elif settings.ENABLE_IMAGE_ANALYSIS and hasattr(shape, "image"):
                    img_index_on_slide += 1
                    try:
                        raw_blob = shape.image.blob
                        if len(raw_blob) > 1500:
                            pil_img = Image.open(io.BytesIO(raw_blob))
                            if pil_img.width >= 100 and pil_img.height >= 100:
                                frames = self.preprocessor.load_and_preprocess(pil_img)
                                for f_img, f_meta in frames:
                                    v_res = self.vision_client.analyze_image(
                                        image=f_img,
                                        prompt_hint=f"PPTX Slide {slide_idx} Image from {file_path.name}",
                                        metadata=f_meta,
                                    )
                                    v_md = v_res.to_rag_markdown()
                                    if v_md.strip():
                                        documents.append(
                                            Document(
                                                page_content=v_md,
                                                metadata={
                                                    "source": file_path.name,
                                                    "page": slide_idx,
                                                    "slide_number": slide_idx,
                                                    "image_index": img_index_on_slide,
                                                    "file_type": ".pptx",
                                                    "doc_type": "pptx_embedded_image",
                                                    "extraction_method": v_res.metadata.get("provider", "gemini-vision"),
                                                    "technical_labels": v_res.technical_labels[:10],
                                                    "units_and_measurements": v_res.units_and_measurements[:10],
                                                },
                                            )
                                        )
                    except Exception as e:
                        logger.debug(f"Failed to extract PPTX slide image: {e}")

            content = "\n".join(slide_parts).strip()
            if not content:
                content = f"[Slide {slide_idx}: Visual/Diagrammatic slide with no extractable text]"

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": file_path.name,
                        "page": slide_idx,
                        "slide_number": slide_idx,
                        "has_table": has_table,
                        "file_type": ".pptx",
                        "doc_type": "pptx_slide_text",
                    },
                )
            )

        if not documents:
            documents.append(
                Document(
                    page_content="[Empty presentation: No slides found]",
                    metadata={
                        "source": file_path.name,
                        "page": 1,
                        "slide_number": 1,
                        "has_table": False,
                        "file_type": ".pptx",
                    },
                )
            )

        logger.info(f"Extracted {len(documents)} item(s) from PPTX '{file_path.name}'")
        return documents

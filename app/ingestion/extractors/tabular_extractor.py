"""
Tabular Extractor for TYRION-2.0.
Extracts tabular rows and structured cell data from CSV and Excel (.xlsx, .xls) files.
Extracts embedded charts and images in spreadsheets, preserving worksheet names and cell coordinates.
"""

import io
from pathlib import Path
from typing import List

import pandas as pd
from PIL import Image
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logger import get_logger
from app.vision.client import get_vision_client
from app.vision.preprocessor import ImagePreprocessor

logger = get_logger()

DEFAULT_ROWS_PER_CHUNK = 35


class TabularExtractor:
    """
    Structured extractor for Tabular data (.csv and .xlsx).
    Extracts spreadsheet data, repeated headers, and embedded technical charts/diagrams.
    """

    def __init__(self, rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK):
        self.rows_per_chunk = rows_per_chunk
        self.preprocessor = ImagePreprocessor()
        self.vision_client = get_vision_client()

    def extract(self, file_path: Path) -> List[Document]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            return self._extract_csv(file_path)
        elif suffix in [".xlsx", ".xls"]:
            return self._extract_excel(file_path)
        else:
            raise ValueError(f"Unsupported tabular file format: '{suffix}'")

    def _extract_csv(self, file_path: Path) -> List[Document]:
        try:
            df = pd.read_csv(file_path, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="latin-1", low_memory=False)
        except Exception as e:
            logger.error(f"Failed to read CSV '{file_path.name}': {e}")
            raise ValueError(f"Malformed or unreadable CSV file: {e}")

        return self._dataframe_to_documents(
            df=df,
            source_name=file_path.name,
            sheet_name="Default",
            page_start=1,
            file_type=".csv",
        )

    def _extract_excel(self, file_path: Path) -> List[Document]:
        try:
            excel_file = pd.ExcelFile(file_path)
        except Exception as e:
            logger.error(f"Failed to open Excel workbook '{file_path.name}': {e}")
            raise ValueError(f"Malformed or unreadable Excel workbook: {e}")

        documents: List[Document] = []
        page_counter = 1

        try:
            for sheet_idx, sheet_name in enumerate(excel_file.sheet_names, start=1):
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                except Exception as e:
                    logger.warning(f"Could not read sheet '{sheet_name}' in '{file_path.name}': {e}")
                    continue

                sheet_docs = self._dataframe_to_documents(
                    df=df,
                    source_name=file_path.name,
                    sheet_name=sheet_name,
                    page_start=page_counter,
                    file_type=".xlsx",
                )
                documents.extend(sheet_docs)
                page_counter += max(1, len(sheet_docs))
        finally:
            excel_file.close()

        # Extract embedded charts / images in worksheets using openpyxl
        if settings.ENABLE_IMAGE_ANALYSIS and file_path.suffix.lower() == ".xlsx":
            try:
                import openpyxl
                wb = openpyxl.load_workbook(file_path, data_only=True)
                for sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    if hasattr(ws, "_images") and ws._images:
                        for img_idx, img in enumerate(ws._images[:4]):
                            try:
                                img_data = img._data()
                                if len(img_data) < 1500:
                                    continue
                                pil_img = Image.open(io.BytesIO(img_data))
                                if pil_img.width < 80 or pil_img.height < 80:
                                    continue

                                cell_anchor = "Unspecified Anchor"
                                if hasattr(img, "anchor") and hasattr(img.anchor, "_from"):
                                    col = getattr(img.anchor._from, "col", 0) + 1
                                    row = getattr(img.anchor._from, "row", 0) + 1
                                    cell_anchor = f"Cell (Row {row}, Column {col})"

                                frames = self.preprocessor.load_and_preprocess(pil_img)
                                for f_img, f_meta in frames:
                                    v_res = self.vision_client.analyze_image(
                                        f_img,
                                        prompt_hint=f"Excel Chart/Image from {file_path.name}, Sheet '{sheet_name}', Anchor {cell_anchor}",
                                        metadata=f_meta,
                                    )
                                    v_md = v_res.to_rag_markdown()
                                    if v_md.strip():
                                        documents.append(
                                            Document(
                                                page_content=v_md,
                                                metadata={
                                                    "source": file_path.name,
                                                    "sheet_name": sheet_name,
                                                    "cell_anchor": cell_anchor,
                                                    "page": page_counter,
                                                    "image_index": img_idx + 1,
                                                    "file_type": ".xlsx",
                                                    "doc_type": "xlsx_embedded_image",
                                                    "extraction_method": v_res.metadata.get("provider", "gemini-vision"),
                                                    "technical_labels": v_res.technical_labels[:10],
                                                    "units_and_measurements": v_res.units_and_measurements[:10],
                                                },
                                            )
                                        )
                            except Exception as e:
                                logger.debug(f"Could not extract spreadsheet image {img_idx + 1}: {e}")
            except Exception as e:
                logger.debug(f"openpyxl embedded image scan skipped: {e}")

        if not documents:
            documents.append(
                Document(
                    page_content="[Empty workbook: No readable sheets or data found]",
                    metadata={
                        "source": file_path.name,
                        "sheet_name": "None",
                        "page": 1,
                        "file_type": ".xlsx",
                    },
                )
            )

        logger.info(
            f"Extracted {len(documents)} tabular/image chunk(s) across "
            f"{len(excel_file.sheet_names)} sheet(s) from '{file_path.name}'"
        )
        return documents

    def _dataframe_to_documents(
        self,
        df: pd.DataFrame,
        source_name: str,
        sheet_name: str,
        page_start: int,
        file_type: str,
    ) -> List[Document]:
        if df.empty:
            return [
                Document(
                    page_content=f"[Empty Sheet: '{sheet_name}']",
                    metadata={
                        "source": source_name,
                        "sheet_name": sheet_name,
                        "page": page_start,
                        "total_rows": 0,
                        "file_type": file_type,
                    },
                )
            ]

        # Convert column names to string and replace newlines
        clean_columns = [str(col).strip().replace("\n", " ") for col in df.columns]
        df.columns = clean_columns

        # Fill NaNs with empty string
        df = df.fillna("")

        total_rows = len(df)
        documents: List[Document] = []
        num_chunks = max(1, (total_rows + self.rows_per_chunk - 1) // self.rows_per_chunk)

        for chunk_idx in range(num_chunks):
            start_row = chunk_idx * self.rows_per_chunk
            end_row = min(start_row + self.rows_per_chunk, total_rows)
            chunk_df = df.iloc[start_row:end_row]

            md_lines: List[str] = [
                f"### Sheet: {sheet_name} (Rows {start_row + 1} to {end_row} of {total_rows})\n",
                "| " + " | ".join(clean_columns) + " |",
                "| " + " | ".join(["---"] * len(clean_columns)) + " |",
            ]

            for _, row in chunk_df.iterrows():
                row_values = [str(val).strip().replace("\n", " ") for val in row.values]
                md_lines.append("| " + " | ".join(row_values) + " |")

            table_content = "\n".join(md_lines)

            documents.append(
                Document(
                    page_content=table_content,
                    metadata={
                        "source": source_name,
                        "sheet_name": sheet_name,
                        "page": page_start + chunk_idx,
                        "row_start": start_row + 1,
                        "row_end": end_row,
                        "total_rows": total_rows,
                        "file_type": file_type,
                        "doc_type": "tabular_data",
                    },
                )
            )

        return documents

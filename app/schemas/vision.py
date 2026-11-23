"""
Vision and Multimodal Ingestion Schemas for TYRION-2.0.
Defines structured data contracts for image OCR, technical diagram interpretation,
tables, chart descriptions, and visual entity detection.
"""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class VisionModelOutput(BaseModel):
    """Clean structured output schema for Gemini Developer API."""

    ocr_text: str = Field(
        default="",
        description="Complete visible printed and legible handwritten text extracted from the image.",
    )
    visual_description: str = Field(
        default="",
        description="Comprehensive objective description of the visual scene, machinery, equipment, or document layout.",
    )
    detected_objects: List[str] = Field(
        default_factory=list,
        description="List of detected industrial objects, components, equipment, devices, or tools.",
    )
    technical_labels: List[str] = Field(
        default_factory=list,
        description="Extracted model numbers, part identifiers, serial numbers, warning codes, or nameplate tags.",
    )
    tables_markdown: List[str] = Field(
        default_factory=list,
        description="Markdown-formatted tables extracted from the image.",
    )
    chart_description: str = Field(
        default="",
        description="Interpretation of charts/graphs including axis labels, units, data trends, legends, and plotted values.",
    )
    diagram_entities: List[str] = Field(
        default_factory=list,
        description="Components, valves, pumps, nodes, or instruments identified in schematics, flowcharts, or P&IDs.",
    )
    relationships: List[str] = Field(
        default_factory=list,
        description="Interconnections, flow directions, wiring lines, or structural hierarchies between diagram entities.",
    )
    units_and_measurements: List[str] = Field(
        default_factory=list,
        description="Specific visible measurement values with units (e.g. '16.0 bar', '1500 RPM', '415 V', '25.0 C').",
    )
    uncertain_regions: List[str] = Field(
        default_factory=list,
        description="Visual regions that are degraded, occluded, low-resolution, blurry, or illegible.",
    )
    processing_status: str = Field(
        default="success",
        description="Processing status: 'success', 'partial_fallback', 'cloud_disabled', or 'failed'.",
    )


class VisionExtractionResult(VisionModelOutput):
    """Full vision extraction result enriched with file provenance and runtime metadata."""

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provenance and technical metadata (resolution, dimensions, source, extraction method).",
    )

    def to_rag_markdown(self) -> str:
        """
        Normalize structured vision findings into dense and sparse searchable Markdown text.
        Ensures technical tokens and descriptions are indexed cleanly in Pinecone and BM25.
        """
        sections: List[str] = []

        if self.visual_description:
            sections.append(f"### Visual Overview & Scene Description\n{self.visual_description}")

        if self.ocr_text:
            sections.append(f"### Extracted Visible Text (OCR)\n{self.ocr_text}")

        if self.technical_labels:
            labels_str = ", ".join(self.technical_labels)
            sections.append(f"### Technical Identifiers & Labels\n{labels_str}")

        if self.units_and_measurements:
            units_str = ", ".join(self.units_and_measurements)
            sections.append(f"### Units, Measurements & Specifications\n{units_str}")

        if self.tables_markdown:
            tables_block = "\n\n".join(self.tables_markdown)
            sections.append(f"### Extracted Tables\n{tables_block}")

        if self.chart_description:
            sections.append(f"### Chart & Graph Analysis\n{self.chart_description}")

        if self.diagram_entities or self.relationships:
            diag_lines = []
            if self.diagram_entities:
                diag_lines.append(f"**Identified Entities/Components:** {', '.join(self.diagram_entities)}")
            if self.relationships:
                diag_lines.append(f"**Flow & Connections:** {'; '.join(self.relationships)}")
            sections.append("### Schematic & Diagram Analysis\n" + "\n\n".join(diag_lines))

        if self.detected_objects:
            sections.append(f"### Detected Physical Objects\n{', '.join(self.detected_objects)}")

        if self.uncertain_regions:
            sections.append(f"### Unclear / Low-Confidence Regions\n{', '.join(self.uncertain_regions)}")

        return "\n\n".join(sections)

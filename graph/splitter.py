"""
graph/splitter.py
─────────────────
Custom document splitter for Control Tower data.txt.

Splits on structural markers (section headers, widget entries, how-to blocks,
worked examples) so each logical unit stays as one atomic chunk.

Each chunk gets metadata:
    - section: parent section name (e.g., "SECTION 2: FORWARD MOVEMENT DASHBOARD")
    - block_type: "section_intro" | "widget" | "howto" | "example" | "reference"
    - block_title: the specific block name (e.g., "WIDGET: Stuck at Destination Hub")

Why not semantic chunking:
    Semantic chunking splits on embedding distance, which frequently breaks
    mid-widget — the widget name ends up in one chunk and the navigation path
    in another. The generator then has context without knowing which widget
    it belongs to. Structure-aware splitting keeps each entry atomic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from langchain_core.documents import Document


# ── Patterns ─────────────────────────────────────────────────────────────────

# Section headers: lines of === followed by SECTION N: TITLE followed by ===
SECTION_HEADER_RE = re.compile(
    r"^={3,}\s*\n(SECTION\s+\d+.*?)\n={3,}",
    re.MULTILINE,
)

# Block markers: --- WIDGET: X --- / --- HOW TO X --- / --- WORKED EXAMPLE: X --- / --- X ---
BLOCK_MARKER_RE = re.compile(
    r"^---\s+(.+?)\s+---\s*$",
    re.MULTILINE,
)


def _classify_block(title: str) -> str:
    """Classify a block by its marker title."""
    title_upper = title.upper()
    if title_upper.startswith("WIDGET:"):
        return "widget"
    if title_upper.startswith("HOW TO"):
        return "howto"
    if title_upper.startswith("WORKED EXAMPLE"):
        return "example"
    if title_upper.startswith("RECOMMENDED"):
        return "howto"
    if title_upper.startswith("SMART ALERT"):
        return "reference"
    if title_upper.startswith("IF YOU CANNOT"):
        return "howto"
    if title_upper.startswith("DEFAULT DASHBOARD"):
        return "reference"
    if title_upper.startswith("DATA VISUALIZATION"):
        return "reference"
    if title_upper.startswith("DRILL-DOWN"):
        return "howto"
    return "reference"


@dataclass
class ChunkMeta:
    section: str
    block_type: str
    block_title: str


def split_control_tower_docs(text: str) -> list[Document]:
    """Split data.txt into structured chunks.

    Args:
        text: Full content of data.txt as a string.

    Returns:
        List of LangChain Document objects, each with page_content and metadata.
    """
    chunks: list[Document] = []
    current_section = "GENERAL"

    # Split into sections first
    # Find all section header positions
    section_splits = list(SECTION_HEADER_RE.finditer(text))

    if not section_splits:
        # No section headers found — treat entire text as one section
        _split_section_into_blocks(text, "GENERAL", chunks)
        return chunks

    # Handle text before first section (if any)
    pre_text = text[: section_splits[0].start()].strip()
    if pre_text:
        _split_section_into_blocks(pre_text, "PREAMBLE", chunks)

    # Process each section
    for i, match in enumerate(section_splits):
        section_name = match.group(1).strip()

        # Section content runs from end of this header to start of next header (or end of file)
        content_start = match.end()
        if i + 1 < len(section_splits):
            content_end = section_splits[i + 1].start()
        else:
            content_end = len(text)

        section_content = text[content_start:content_end].strip()

        if section_content:
            _split_section_into_blocks(section_content, section_name, chunks)

    return chunks


def _split_section_into_blocks(
    section_text: str,
    section_name: str,
    chunks: list[Document],
) -> None:
    """Split a section's content into blocks based on --- markers."""

    block_markers = list(BLOCK_MARKER_RE.finditer(section_text))

    if not block_markers:
        # No block markers — entire section is one chunk
        content = _clean_content(section_text)
        if content and len(content) > 50:
            chunks.append(Document(
                page_content=content,
                metadata={
                    "section": section_name,
                    "block_type": "section_intro",
                    "block_title": section_name,
                },
            ))
        return

    # Text before the first block marker = section intro
    intro_text = section_text[: block_markers[0].start()].strip()
    if intro_text and len(intro_text) > 50:
        chunks.append(Document(
            page_content=_clean_content(intro_text),
            metadata={
                "section": section_name,
                "block_type": "section_intro",
                "block_title": section_name,
            },
        ))

    # Each block marker to the next
    for i, marker in enumerate(block_markers):
        block_title = marker.group(1).strip()
        block_type = _classify_block(block_title)

        content_start = marker.end()
        if i + 1 < len(block_markers):
            content_end = block_markers[i + 1].start()
        else:
            content_end = len(section_text)

        block_content = section_text[content_start:content_end].strip()

        if not block_content or len(block_content) < 30:
            continue

        # Prepend the block title to the content so the chunk is self-contained
        # This is critical — without this, a chunk might say "Where to find it:
        # Dashboard: Forward Movement" without ever naming the widget.
        full_content = f"{block_title}\n\n{block_content}"

        chunks.append(Document(
            page_content=_clean_content(full_content),
            metadata={
                "section": section_name,
                "block_type": block_type,
                "block_title": block_title,
            },
        ))


def _clean_content(text: str) -> str:
    """Clean up chunk content — remove excess whitespace and separator lines."""
    # Remove lines that are just === or ---
    lines = text.split("\n")
    cleaned = [
        line for line in lines
        if not re.match(r"^[=\-]{3,}\s*$", line.strip())
    ]
    result = "\n".join(cleaned).strip()
    # Collapse multiple blank lines into one
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result
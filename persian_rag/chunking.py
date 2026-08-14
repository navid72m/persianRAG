"""
Parent-child chunking for a single long Persian document.

Strategy:
- Parse page-by-page with PyMuPDF (handles Persian/RTL Unicode text fine —
  it extracts logical reading order, not visual order, which is what we want).
- Group pages into PARENT chunks of ~parent_chunk_tokens, breaking on the
  nearest heading boundary when we can detect one (lines that are short,
  bold, or larger font size), otherwise on plain page boundaries.
- Split each parent into overlapping CHILD chunks of ~child_chunk_tokens.
  Children carry a parent_id + page_range in metadata; children are what
  gets embedded and indexed, parents are what gets fetched for generation.
"""
from dataclasses import dataclass, field

import fitz  # PyMuPDF

# We use a plain whitespace word count as the chunk-size budgeting unit
# rather than tiktoken. tiktoken's cl100k_base BPE is English-centric and
# also requires a network fetch of its merge table on first use — neither
# is a good fit here. A word count is a fine *relative* size signal for
# Persian text; ~1 word ≈ 1.3-1.6 LLM tokens in Persian, so if you want to
# budget by actual LLM tokens, scale target_tokens down accordingly (the
# defaults in config.py already assume word-count units).


def _n_tokens(text: str) -> int:
    return len(text.split())


def _encode(text: str) -> list[str]:
    return text.split()


def _decode(tokens: list[str]) -> str:
    return " ".join(tokens)


@dataclass
class PageText:
    page_num: int  # 1-indexed
    text: str
    looks_like_heading_start: bool  # first line on the page is short/bold


@dataclass
class ParentChunk:
    parent_id: str
    text: str
    page_start: int
    page_end: int


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    text: str
    page_start: int
    page_end: int


def _ocr_page(page, lang: str, dpi: int) -> str:
    """OCR one rendered page. Lazy-imports pytesseract so text-layer PDFs
    never need it."""
    import io

    import pytesseract
    from PIL import Image

    pix = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, lang=lang).strip()


def parse_pdf(path: str, ocr_enabled: bool = True, ocr_lang: str = "fas+eng",
              ocr_dpi: int = 300) -> list[PageText]:
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        ocr_used = False
        if ocr_enabled and len(text) < 10:
            text = _ocr_page(page, ocr_lang, ocr_dpi)
            ocr_used = True
            print(f"  OCR page {i}/{len(doc)} ({len(text)} chars)")
        heading_like = False
        if text and not ocr_used:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                for line in b.get("lines", []):
                    spans = line.get("spans", [])
                    if spans and len(spans[0]["text"].strip()) > 0:
                        first = spans[0]
                        heading_like = first["size"] >= 13 and len(first["text"]) < 60
                        break
                break
        pages.append(PageText(page_num=i, text=text, looks_like_heading_start=heading_like))
    doc.close()
    return pages


def build_parent_chunks(pages: list[PageText], target_tokens: int) -> list[ParentChunk]:
    parents: list[ParentChunk] = []
    buf_text: list[str] = []
    buf_start = None
    buf_tokens = 0

    def flush(end_page: int):
        nonlocal buf_text, buf_start, buf_tokens
        if buf_text:
            parents.append(ParentChunk(
                parent_id=f"p{len(parents):05d}",
                text="\n\n".join(buf_text).strip(),
                page_start=buf_start,
                page_end=end_page,
            ))
        buf_text, buf_start, buf_tokens = [], None, 0

    for page in pages:
        if not page.text:
            continue
        if buf_start is None:
            buf_start = page.page_num

        page_tokens = _n_tokens(page.text)
        # break on heading boundary once we're already over ~70% of budget,
        # otherwise just cap on raw token budget so nothing balloons
        if buf_tokens >= target_tokens * 0.7 and page.looks_like_heading_start:
            flush(page.page_num - 1)
            buf_start = page.page_num

        buf_text.append(page.text)
        buf_tokens += page_tokens

        if buf_tokens >= target_tokens:
            flush(page.page_num)

    if buf_text:
        flush(pages[-1].page_num if pages else buf_start)

    return parents


def build_child_chunks(parents: list[ParentChunk], target_tokens: int, overlap_tokens: int) -> list[ChildChunk]:
    children: list[ChildChunk] = []
    for parent in parents:
        tokens = _encode(parent.text)
        step = max(1, target_tokens - overlap_tokens)
        idx = 0
        n_children_this_parent = 0
        for start in range(0, len(tokens), step):
            piece = tokens[start:start + target_tokens]
            if not piece:
                continue
            text = _decode(piece)
            children.append(ChildChunk(
                child_id=f"{parent.parent_id}-c{n_children_this_parent:03d}",
                parent_id=parent.parent_id,
                text=text,
                page_start=parent.page_start,
                page_end=parent.page_end,
            ))
            n_children_this_parent += 1
            if start + target_tokens >= len(tokens):
                break
    return children


def chunk_document(pdf_path: str, parent_tokens: int, child_tokens: int, child_overlap: int,
                   ocr_enabled: bool = True, ocr_lang: str = "fas+eng",
                   ocr_dpi: int = 300) -> tuple[list[ParentChunk], list[ChildChunk]]:
    pages = parse_pdf(pdf_path, ocr_enabled=ocr_enabled, ocr_lang=ocr_lang, ocr_dpi=ocr_dpi)
    parents = build_parent_chunks(pages, parent_tokens)
    children = build_child_chunks(parents, child_tokens, child_overlap)
    return parents, children

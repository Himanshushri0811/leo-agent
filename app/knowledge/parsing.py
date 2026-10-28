"""
Document parsing: extract plain text from various source types.

Supports:
- URL (HTML via BeautifulSoup)
- PDF (via pypdf)
- Plain text

All parsers return normalised UTF-8 text ready for chunking.
Retrieved content is treated as untrusted data - it is never executed.
"""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from pathlib import Path

import httpx
from bs4 import BeautifulSoup


class SourceType(StrEnum):
    URL = "url"
    PDF = "pdf"
    TEXT = "text"
    MARKDOWN = "markdown"
    DOCX = "docx"


def parse_url(url: str, timeout: int = 30) -> tuple[str, str]:
    """
    Fetch and parse HTML from a URL.

    Returns:
        (plain_text, content_hash)

    Raises:
        httpx.HTTPError on network failures.
    """
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url, headers={"User-Agent": "LEO-KnowledgeBot/1.0"})
        response.raise_for_status()
        html = response.text

    soup = BeautifulSoup(html, "lxml")

    # Remove script/style/nav noise
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    normalised = _normalise(text)
    return normalised, _hash(normalised)


def parse_pdf(path: Path) -> tuple[str, str]:
    """Extract text from a PDF file."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)

    text = "\n".join(pages)
    normalised = _normalise(text)
    return normalised, _hash(normalised)


def parse_text(content: str) -> tuple[str, str]:
    """Normalise plain text."""
    normalised = _normalise(content)
    return normalised, _hash(normalised)


def parse_docx(path: Path) -> tuple[str, str]:
    """
    Extract text from a Word document (.docx) using standard library zipfile and XML.
    Requires no external native binary dependencies.
    """
    import xml.etree.ElementTree as ET
    import zipfile

    paragraphs: list[str] = []
    with zipfile.ZipFile(str(path), "r") as docx_zip:
        xml_content = docx_zip.read("word/document.xml")
        root = ET.fromstring(xml_content)
        w_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        for p in root.iter(f"{{{w_ns}}}p"):
            texts = [node.text for node in p.iter(f"{{{w_ns}}}t") if node.text]
            if texts:
                paragraphs.append("".join(texts))

    text = "\n\n".join(paragraphs)
    normalised = _normalise(text)
    return normalised, _hash(normalised)


def parse_document(path: Path) -> tuple[str, str, str]:
    """
    Parse a local document by extension (.pdf, .docx, .md, .markdown, .txt).

    Returns:
        (normalised_text, content_hash, source_type)
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text, content_hash = parse_pdf(path)
        source_type = SourceType.PDF
    elif suffix == ".docx":
        text, content_hash = parse_docx(path)
        source_type = SourceType.DOCX
    elif suffix in (".md", ".markdown"):
        raw = path.read_text(encoding="utf-8", errors="replace")
        text, content_hash = parse_text(raw)
        source_type = SourceType.MARKDOWN
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")
        text, content_hash = parse_text(raw)
        source_type = SourceType.TEXT

    return text, content_hash, source_type


def _normalise(text: str) -> str:
    """Collapse whitespace, remove null bytes, strip leading/trailing spaces."""
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

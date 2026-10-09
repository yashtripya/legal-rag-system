"""
Document Loader Module for Legal RAG System.
Handles loading and parsing of PDF and TXT legal documents with metadata extraction
and scanned document detection.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from pypdf import PdfReader


class DocumentLoadingError(Exception):
    """Custom exception raised when a document fails validation or loading."""
    pass


class LegalDocument:
    """Represents a loaded legal document unit (file or page) with rich metadata."""
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self) -> str:
        doc_title = self.metadata.get("document_title", "Unknown")
        page = self.metadata.get("page_number", 1)
        return f"<LegalDocument title='{doc_title}' page={page} length={len(self.page_content)}>"


class LegalDocumentLoader:
    """
    Ingests PDF and TXT legal files and extracts text along with metadata
    such as document title, court, citation, and page numbers.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    def __init__(self):
        pass

    def load_file(self, file_path: str | Path) -> List[LegalDocument]:
        """
        Loads a single document file (.pdf or .txt) and returns a list of LegalDocument objects.
        
        Args:
            file_path: Path to the document.
            
        Returns:
            List of LegalDocument objects containing content and metadata.
            
        Raises:
            DocumentLoadingError: If the file does not exist, is unsupported, empty, or unreadable.
        """
        path = Path(file_path).resolve()

        if not path.exists():
            raise DocumentLoadingError(f"File not found: {path}")

        if not path.is_file():
            raise DocumentLoadingError(f"Path is not a regular file: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise DocumentLoadingError(
                f"Unsupported file format '{ext}'. Only {self.SUPPORTED_EXTENSIONS} are supported."
            )

        if path.stat().st_size == 0:
            raise DocumentLoadingError(f"Document file is empty (0 bytes): {path.name}")

        if ext == ".txt":
            return self._load_txt(path)
        elif ext == ".pdf":
            return self._load_pdf(path)
        else:
            raise DocumentLoadingError(f"Unhandled file extension: {ext}")

    def _load_txt(self, path: Path) -> List[LegalDocument]:
        """Loads a text document."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()
        except Exception as e:
            raise DocumentLoadingError(f"Failed to read text file '{path.name}': {str(e)}")

        if not content:
            raise DocumentLoadingError(f"Text document '{path.name}' contains no readable text.")

        metadata = self._extract_header_metadata(content, path)
        metadata["page_number"] = 1
        metadata["total_pages"] = 1
        metadata["file_type"] = "txt"
        metadata["source_path"] = str(path)
        metadata["filename"] = path.name

        return [LegalDocument(page_content=content, metadata=metadata)]

    def _load_pdf(self, path: Path) -> List[LegalDocument]:
        """Loads a PDF document page by page."""
        try:
            reader = PdfReader(str(path))
        except Exception as e:
            raise DocumentLoadingError(f"Failed to open PDF '{path.name}': {str(e)}")

        num_pages = len(reader.pages)
        if num_pages == 0:
            raise DocumentLoadingError(f"PDF document '{path.name}' has 0 pages.")

        documents: List[LegalDocument] = []
        total_extracted_length = 0

        # Attempt to read first page to extract document-level metadata
        first_page_text = ""
        try:
            first_page_text = reader.pages[0].extract_text() or ""
        except Exception:
            pass

        base_metadata = self._extract_header_metadata(first_page_text, path)
        base_metadata["total_pages"] = num_pages
        base_metadata["file_type"] = "pdf"
        base_metadata["source_path"] = str(path)
        base_metadata["filename"] = path.name

        for i, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""

            cleaned_page = page_text.strip()
            total_extracted_length += len(cleaned_page)

            page_metadata = base_metadata.copy()
            page_metadata["page_number"] = i

            if cleaned_page:
                documents.append(LegalDocument(page_content=cleaned_page, metadata=page_metadata))

        # Check for scanned PDF condition
        if total_extracted_length < 20:
            raise DocumentLoadingError(
                f"PDF '{path.name}' contains little to no extractable text ({total_extracted_length} chars). "
                f"The document may be scanned or image-based. OCR preprocessing is required."
            )

        return documents

    def _extract_header_metadata(self, text: str, path: Path) -> Dict[str, Any]:
        """Extracts structured legal metadata from header comments/fields if present."""
        metadata: Dict[str, Any] = {
            "document_title": path.stem.replace("_", " ").title(),
            "court_or_authority": "Indian Legal Authority",
            "jurisdiction": "Indian Law",
            "date": "Unknown",
            "source_url": "N/A",
        }

        # Parse Case Title / Citation / Document Title
        title_match = re.search(r"(?:Case Title|Statute Name|Document Title):\s*([^\n]+)", text, re.IGNORECASE)
        if title_match:
            metadata["document_title"] = title_match.group(1).strip()

        # Parse Issuing Authority / Court
        court_match = re.search(r"(?:Issuing Authority|Court|Bench):\s*([^\n]+)", text, re.IGNORECASE)
        if court_match:
            metadata["court_or_authority"] = court_match.group(1).strip()

        # Parse Date
        date_match = re.search(r"(?:Date|Date of Judgment):\s*([^\n]+)", text, re.IGNORECASE)
        if date_match:
            metadata["date"] = date_match.group(1).strip()

        # Parse Source URL
        url_match = re.search(r"(?:Source URL|URL):\s*(https?://[^\s]+)", text, re.IGNORECASE)
        if url_match:
            metadata["source_url"] = url_match.group(1).strip()

        return metadata

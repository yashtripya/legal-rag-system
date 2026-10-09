"""
Text Processor Module for Legal RAG System.
Provides legal text cleaning, normalization, and chunking with full metadata preservation.
"""

import re
import hashlib
from typing import List, Dict, Any

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.document_loader import LegalDocument


class LegalTextProcessor:
    """
    Cleans legal text and performs metadata-aware chunking optimized for legal document structure.
    """

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Custom separators prioritizing legal clause boundaries
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\nSECTION ",
                "\n\nARTICLE ",
                "\n\nEXCEPTIONS ",
                "\n\n",
                "\n",
                ". ",
                "; ",
                " ",
                "",
            ],
        )

    def clean_text(self, text: str) -> str:
        """
        Cleans unnecessary whitespace, page noise, and invalid characters while
        preserving legally meaningful symbols (§, AIR citations, section numbers).
        """
        if not text:
            return ""

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove header/footer page numbers (e.g. Page 1 of 12, -- Page 3 --)
        text = re.sub(r"(?i)^\s*(?:page|\-\-)\s*\d+\s*(?:of\s*\d+|\-\-)?\s*$", "", text, flags=re.MULTILINE)

        # Normalize multiple spaces (preserving newlines)
        text = re.sub(r"[ \t]+", " ", text)

        # Collapse 3+ newlines to 2 newlines (preserve paragraph boundaries)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def process_documents(self, documents: List[LegalDocument]) -> List[LegalDocument]:
        """
        Processes a list of LegalDocument objects by cleaning text and splitting into chunks.
        
        Args:
            documents: Raw LegalDocument instances from document loader.
            
        Returns:
            List of chunked LegalDocument instances with complete metadata.
        """
        all_chunks: List[LegalDocument] = []

        for doc in documents:
            cleaned_content = self.clean_text(doc.page_content)
            if not cleaned_content:
                continue

            # Split content using legal text splitter
            raw_chunks = self.splitter.split_text(cleaned_content)

            total_chunks = len(raw_chunks)
            for idx, chunk_str in enumerate(raw_chunks):
                chunk_str = chunk_str.strip()
                if not chunk_str:
                    continue

                # Build updated metadata
                chunk_metadata = doc.metadata.copy()
                chunk_metadata["chunk_index"] = idx
                chunk_metadata["total_chunks"] = total_chunks
                
                # Deterministic chunk ID based on filename, page, and chunk index
                filename = chunk_metadata.get("filename", "doc")
                page_num = chunk_metadata.get("page_number", 1)
                id_str = f"{filename}_p{page_num}_c{idx}_{hashlib.sha256(chunk_str.encode('utf-8')).hexdigest()[:8]}"
                chunk_metadata["chunk_id"] = id_str

                all_chunks.append(LegalDocument(page_content=chunk_str, metadata=chunk_metadata))

        return all_chunks

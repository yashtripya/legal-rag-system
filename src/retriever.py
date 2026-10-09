"""
Retriever Module for Legal RAG System.
Handles similarity search, score thresholding, passage deduplication, and context aggregation.
"""

from typing import List, Tuple, Dict, Any, Optional
from config import DEFAULT_TOP_K, DEFAULT_SCORE_THRESHOLD
from src.document_loader import LegalDocument
from src.vector_store import LegalVectorStore


class RetrievedContext:
    """Encapsulates retrieved evidence passages and query context."""

    def __init__(
        self,
        query: str,
        passages: List[Tuple[LegalDocument, float]],
        total_found: int,
    ):
        self.query = query
        self.passages = passages  # List of (LegalDocument, similarity_score)
        self.total_found = total_found

    def is_empty(self) -> bool:
        return len(self.passages) == 0

    def get_documents(self) -> List[LegalDocument]:
        return [doc for doc, _ in self.passages]

    def to_formatted_context(self) -> str:
        """Formats evidence passages into a clean context string for LLM prompt ingestion."""
        if not self.passages:
            return "NO RELEVANT LEGAL DOCUMENTS FOUND."

        formatted_blocks = []
        for idx, (doc, score) in enumerate(self.passages, start=1):
            meta = doc.metadata
            title = meta.get("document_title", "Legal Document")
            court = meta.get("court_or_authority", "Indian Legal Authority")
            page = meta.get("page_number", 1)
            filename = meta.get("filename", "N/A")
            
            block = (
                f"[EVIDENCE SOURCE {idx}]\n"
                f"Document: {title}\n"
                f"Authority/Court: {court}\n"
                f"Source File: {filename} (Page {page})\n"
                f"Relevance Score: {score:.4f}\n"
                f"Passage Content:\n{doc.page_content.strip()}\n"
            )
            formatted_blocks.append(block)

        return "\n----------------------------------------\n".join(formatted_blocks)


class LegalRetriever:
    """
    Executes context retrieval over the vector database with filtering and deduplication.
    """

    def __init__(self, vector_store: LegalVectorStore):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> RetrievedContext:
        """
        Retrieves relevant legal passages for a user query.
        
        Args:
            query: Question string.
            top_k: Max chunks to retrieve.
            score_threshold: Minimum similarity score threshold (0.0 to 1.0).
            
        Returns:
            RetrievedContext object containing filtered and deduplicated passages.
        """
        clean_query = query.strip()
        if not clean_query:
            return RetrievedContext(query="", passages=[], total_found=0)

        # Retrieve candidates from vector store
        raw_results = self.vector_store.similarity_search_with_score(query=clean_query, top_k=top_k * 2)

        # Filter by score threshold
        filtered_results = [(doc, score) for doc, score in raw_results if score >= score_threshold]

        # Deduplicate identical or near-identical passages
        deduped_results: List[Tuple[LegalDocument, float]] = []
        seen_texts = set()

        for doc, score in filtered_results:
            normalized_text = doc.page_content.strip().lower()
            if normalized_text in seen_texts:
                continue
            seen_texts.add(normalized_text)
            deduped_results.append((doc, score))

            if len(deduped_results) >= top_k:
                break

        return RetrievedContext(
            query=clean_query,
            passages=deduped_results,
            total_found=len(raw_results),
        )

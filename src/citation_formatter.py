"""
Citation Formatter Module for Legal RAG System.
Generates verified, source-traceable citations from retrieved chunk metadata.
"""

from typing import List, Tuple, Dict, Any
from src.document_loader import LegalDocument


class LegalCitationFormatter:
    """
    Formats retrieved metadata into verified, standard legal citations.
    """

    @staticmethod
    def format_citations(passages: List[Tuple[LegalDocument, float]]) -> List[Dict[str, Any]]:
        """
        Extracts and formats verified citation records from retrieved passages.
        
        Args:
            passages: List of (LegalDocument, score) tuples.
            
        Returns:
            List of structured citation dictionaries.
        """
        citations: List[Dict[str, Any]] = []

        for idx, (doc, score) in enumerate(passages, start=1):
            meta = doc.metadata
            title = meta.get("document_title", "Legal Document")
            court = meta.get("court_or_authority", "Indian Legal Authority")
            date = meta.get("date", "N/A")
            page = meta.get("page_number", 1)
            filename = meta.get("filename", "Unknown")
            source_url = meta.get("source_url", "N/A")
            
            # Short excerpt (first 250 chars)
            excerpt = doc.page_content.strip()
            if len(excerpt) > 250:
                excerpt = excerpt[:247] + "..."

            citations.append({
                "citation_id": f"CIT-{idx}",
                "title": title,
                "authority": court,
                "date": date,
                "page": page,
                "filename": filename,
                "source_url": source_url,
                "relevance_score": score,
                "excerpt": excerpt,
                "full_chunk_text": doc.page_content.strip(),
            })

        return citations

    @staticmethod
    def render_markdown_citations(citations: List[Dict[str, Any]]) -> str:
        """Renders citation dictionaries as a formatted Markdown section."""
        if not citations:
            return "_No verifiable citations available for this query._"

        lines = ["### Verifiable Legal Source Citations\n"]
        for cit in citations:
            lines.append(
                f"**[{cit['citation_id']}] {cit['title']}**\n"
                f"- **Authority / Court:** {cit['authority']}\n"
                f"- **Source File & Page:** `{cit['filename']}` (Page {cit['page']})\n"
                f"- **Date:** {cit['date']} | **Relevance Match:** {cit['relevance_score']*100:.1f}%\n"
                f"- **Verbatim Excerpt:**\n  > \"{cit['excerpt']}\"\n"
            )
            if cit["source_url"] != "N/A":
                lines.append(f"- **Official Link:** [{cit['source_url']}]({cit['source_url']})\n")
            lines.append("\n---\n")

        return "\n".join(lines)

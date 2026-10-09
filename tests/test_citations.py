"""
Unit Tests for Citation Formatter Module.
"""

from src.document_loader import LegalDocument
from src.citation_formatter import LegalCitationFormatter


def test_citation_formatting():
    doc = LegalDocument(
        page_content="Parliament cannot abrogate the basic structure of the Constitution.",
        metadata={
            "document_title": "Kesavananda Bharati v. State of Kerala",
            "court_or_authority": "Supreme Court of India",
            "date": "1973-04-24",
            "page_number": 12,
            "filename": "kes.txt",
            "source_url": "https://main.sci.gov.in/judgments",
        },
    )

    passages = [(doc, 0.92)]
    citations = LegalCitationFormatter.format_citations(passages)

    assert len(citations) == 1
    cit = citations[0]
    assert cit["title"] == "Kesavananda Bharati v. State of Kerala"
    assert cit["authority"] == "Supreme Court of India"
    assert cit["page"] == 12
    assert cit["relevance_score"] == 0.92

    markdown = LegalCitationFormatter.render_markdown_citations(citations)
    assert "Kesavananda Bharati v. State of Kerala" in markdown
    assert "Supreme Court of India" in markdown
    assert "https://main.sci.gov.in/judgments" in markdown

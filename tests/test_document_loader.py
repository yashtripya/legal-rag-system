"""
Unit Tests for Document Loader Module.
"""

import pytest
from pathlib import Path
from src.document_loader import LegalDocumentLoader, DocumentLoadingError


def test_load_valid_txt(tmp_path):
    txt_file = tmp_path / "sample_judgment.txt"
    txt_content = (
        "Case Title: Kesavananda Bharati v. State of Kerala\n"
        "Court: Supreme Court of India\n"
        "Date: 1973-04-24\n\n"
        "Parliament cannot alter the basic structure of the Constitution under Article 368."
    )
    txt_file.write_text(txt_content, encoding="utf-8")

    loader = LegalDocumentLoader()
    docs = loader.load_file(txt_file)

    assert len(docs) == 1
    assert docs[0].page_content == txt_content
    assert docs[0].metadata["document_title"] == "Kesavananda Bharati v. State of Kerala"
    assert docs[0].metadata["court_or_authority"] == "Supreme Court of India"
    assert docs[0].metadata["file_type"] == "txt"


def test_load_nonexistent_file():
    loader = LegalDocumentLoader()
    with pytest.raises(DocumentLoadingError, match="File not found"):
        loader.load_file("nonexistent_legal_doc.pdf")


def test_load_unsupported_format(tmp_path):
    docx_file = tmp_path / "test.docx"
    docx_file.write_text("Unsupported format content")

    loader = LegalDocumentLoader()
    with pytest.raises(DocumentLoadingError, match="Unsupported file format"):
        loader.load_file(docx_file)


def test_load_empty_file(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("")

    loader = LegalDocumentLoader()
    with pytest.raises(DocumentLoadingError, match="Document file is empty"):
        loader.load_file(empty_file)

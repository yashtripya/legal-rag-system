"""
Unit Tests for Text Processor Module.
"""

from src.document_loader import LegalDocument
from src.text_processor import LegalTextProcessor


def test_clean_text():
    processor = LegalTextProcessor()
    noisy_text = "Section 300  IPC   v.   State\n\n\n\nPage 1 of 5\n\nWhoever causes death..."
    cleaned = processor.clean_text(noisy_text)

    assert "Section 300 IPC v. State" in cleaned
    assert "\n\n\n" not in cleaned
    assert "Page 1 of 5" not in cleaned


def test_process_documents_chunking():
    processor = LegalTextProcessor(chunk_size=100, chunk_overlap=20)
    
    long_text = (
        "SECTION 300: MURDER\n"
        "Except in cases hereinafter excepted, culpable homicide is murder if the act by which death is caused "
        "is done with the intention of causing death or bodily injury sufficient in ordinary course of nature to cause death."
    )
    meta = {"document_title": "IPC Statute", "page_number": 1, "filename": "ipc.txt"}
    doc = LegalDocument(page_content=long_text, metadata=meta)

    chunks = processor.process_documents([doc])

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 120
        assert chunk.metadata["document_title"] == "IPC Statute"
        assert "chunk_id" in chunk.metadata
        assert "chunk_index" in chunk.metadata

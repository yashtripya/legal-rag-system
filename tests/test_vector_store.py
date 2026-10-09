"""
Unit Tests for Vector Store Module using isolated temp directory.
"""

import pytest
from src.document_loader import LegalDocument
from src.vector_store import LegalVectorStore


@pytest.fixture
def temp_vector_store(tmp_path):
    persist_dir = tmp_path / "chroma_test_db"
    store = LegalVectorStore(persist_dir=persist_dir, collection_name="test_legal_docs")
    yield store
    store.clear_collection()


def test_add_and_search_documents(temp_vector_store):
    doc1 = LegalDocument(
        page_content="Parliament cannot alter the basic structure of the Constitution under Article 368.",
        metadata={"document_title": "Kesavananda Bharati", "page_number": 1, "chunk_id": "c1", "filename": "kes.txt"},
    )
    doc2 = LegalDocument(
        page_content="Right to Privacy is a fundamental right under Article 21 of the Indian Constitution.",
        metadata={"document_title": "Puttaswamy Privacy", "page_number": 1, "chunk_id": "c2", "filename": "put.txt"},
    )

    result = temp_vector_store.add_documents([doc1, doc2])
    assert result["added_count"] == 2
    assert result["total_chunks"] == 2

    # Similarity search query
    matches = temp_vector_store.similarity_search_with_score("basic structure doctrine", top_k=2)
    assert len(matches) > 0
    top_doc, score = matches[0]
    assert "Kesavananda" in top_doc.metadata["document_title"]
    assert score > 0.0


def test_deduplication(temp_vector_store):
    doc = LegalDocument(
        page_content="Section 300 IPC defines the offense of Murder in India.",
        metadata={"document_title": "IPC", "page_number": 1, "chunk_id": "same_chunk_id", "filename": "ipc.txt"},
    )

    res1 = temp_vector_store.add_documents([doc])
    assert res1["added_count"] == 1

    # Adding exact same chunk ID again should be skipped
    res2 = temp_vector_store.add_documents([doc])
    assert res2["added_count"] == 0
    assert res2["skipped_count"] == 1

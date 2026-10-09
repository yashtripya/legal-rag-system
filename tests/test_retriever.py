"""
Unit Tests for Retriever Module.
"""

import pytest
from src.document_loader import LegalDocument
from src.vector_store import LegalVectorStore
from src.retriever import LegalRetriever


@pytest.fixture
def retriever_with_data(tmp_path):
    store = LegalVectorStore(persist_dir=tmp_path / "chroma_retriever_db", collection_name="test_retriever")
    doc1 = LegalDocument(
        page_content="Article 21 guarantees protection of life and personal liberty except according to procedure established by law.",
        metadata={"document_title": "Constitution Article 21", "page_number": 1, "chunk_id": "r1", "filename": "const.txt"},
    )
    doc2 = LegalDocument(
        page_content="Section 302 of the IPC prescribes death or life imprisonment for murder.",
        metadata={"document_title": "IPC Section 302", "page_number": 1, "chunk_id": "r2", "filename": "ipc.txt"},
    )
    store.add_documents([doc1, doc2])
    return LegalRetriever(vector_store=store)


def test_retrieval(retriever_with_data):
    context = retriever_with_data.retrieve("What is Article 21?", top_k=2, score_threshold=0.1)
    assert not context.is_empty()
    docs = context.get_documents()
    assert any("Article 21" in d.page_content for d in docs)


def test_empty_query_retrieval(retriever_with_data):
    context = retriever_with_data.retrieve("   ", top_k=2)
    assert context.is_empty()
    assert len(context.passages) == 0

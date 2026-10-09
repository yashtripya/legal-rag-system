"""
Integration Tests for RAG Pipeline Master Module.
"""

from src.rag_pipeline import LegalRAGPipeline


def test_rag_pipeline_end_to_end(tmp_path):
    pipeline = LegalRAGPipeline(
        persist_dir=tmp_path / "chroma_pipeline_db",
        collection_name="test_pipeline",
    )

    # Create temporary legal document
    sample_file = tmp_path / "constitution_test.txt"
    sample_file.write_text(
        "Document Title: Constitution Test\n"
        "Issuing Authority: Supreme Court\n\n"
        "Article 21 protects life and personal liberty of all citizens and non-citizens in India.",
        encoding="utf-8",
    )

    # 1. Test Ingestion
    ingest_res = pipeline.ingest_file(sample_file)
    assert ingest_res["status"] == "success"
    assert ingest_res["chunks_added"] > 0

    # 2. Test Query & Fallback Synthesis
    query_res = pipeline.query("What does Article 21 protect?", top_k=1)
    assert query_res["query"] == "What does Article 21 protect?"
    assert len(query_res["citations"]) == 1
    assert "Article 21" in query_res["citations"][0]["full_chunk_text"]
    assert query_res["answer"] is not None

    # 3. Test System Status
    status = pipeline.get_system_status()
    assert status["vector_store"]["total_chunks"] > 0

"""
RAG Pipeline Master Orchestrator for Legal RAG System.
Unifies document ingestion, text processing, vector indexing, retrieval, local LLM generation, and citation formatting.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

from config import CHROMA_DB_DIR, COLLECTION_NAME, SAMPLE_DATA_DIR, DEFAULT_TOP_K, DEFAULT_SCORE_THRESHOLD
from src.document_loader import LegalDocumentLoader, LegalDocument, DocumentLoadingError
from src.text_processor import LegalTextProcessor
from src.embeddings import LocalLegalEmbeddings
from src.vector_store import LegalVectorStore
from src.retriever import LegalRetriever, RetrievedContext
from src.llm_service import LocalLegalLLMService


class LegalRAGPipeline:
    """
    Master Legal RAG Pipeline controller.
    Provides end-to-end APIs for document ingestion, collection management, and legal Q&A.
    """

    def __init__(
        self,
        persist_dir: str | Path = CHROMA_DB_DIR,
        collection_name: str = COLLECTION_NAME,
        llm_model: str = "llama3.2:1b",
    ):
        self.loader = LegalDocumentLoader()
        self.processor = LegalTextProcessor()
        self.embeddings = LocalLegalEmbeddings()
        self.vector_store = LegalVectorStore(
            persist_dir=persist_dir,
            collection_name=collection_name,
            embedding_model=self.embeddings,
        )
        self.retriever = LegalRetriever(vector_store=self.vector_store)
        self.llm_service = LocalLegalLLMService(model_name=llm_model)

    def ingest_file(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Loads, cleans, chunks, and indexes a single legal document file.
        
        Args:
            file_path: Path to PDF or TXT file.
            
        Returns:
            Dictionary summary with status, filename, page_count, chunks_added, and error details.
        """
        path = Path(file_path).resolve()
        try:
            # 1. Load document
            docs = self.loader.load_file(path)
            
            # 2. Clean & chunk document
            chunks = self.processor.process_documents(docs)
            
            # 3. Index into vector database
            index_result = self.vector_store.add_documents(chunks)
            
            return {
                "status": "success",
                "filename": path.name,
                "total_pages": docs[0].metadata.get("total_pages", 1) if docs else 0,
                "chunks_processed": len(chunks),
                "chunks_added": index_result.get("added_count", 0),
                "chunks_skipped": index_result.get("skipped_count", 0),
            }
        except DocumentLoadingError as dle:
            return {"status": "error", "filename": path.name, "error_type": "loading", "error": str(dle)}
        except Exception as e:
            return {"status": "error", "filename": path.name, "error_type": "system", "error": str(e)}

    def ingest_directory(self, dir_path: str | Path = SAMPLE_DATA_DIR) -> Dict[str, Any]:
        """
        Ingests all supported legal documents (.pdf, .txt) from a directory.
        
        Args:
            dir_path: Directory path.
            
        Returns:
            Batch summary dictionary.
        """
        target_dir = Path(dir_path).resolve()
        if not target_dir.exists() or not target_dir.is_dir():
            return {"status": "error", "error": f"Directory not found: {target_dir}"}

        results = []
        supported_files = [
            f for f in target_dir.glob("*") if f.suffix.lower() in LegalDocumentLoader.SUPPORTED_EXTENSIONS
        ]

        for file_p in supported_files:
            res = self.ingest_file(file_p)
            results.append(res)

        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "error"]

        return {
            "status": "completed",
            "total_files_found": len(supported_files),
            "successful_files": len(successful),
            "failed_files": len(failed),
            "details": results,
            "vector_store_stats": self.vector_store.get_stats(),
        }

    def query(
        self,
        user_query: str,
        history: Optional[List[Dict[str, str]]] = None,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        model_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end legal Q&A: query retrieval -> prompt assembly -> LLM inference -> citation formatting.
        
        Args:
            user_query: User's legal question.
            history: Session conversation history.
            top_k: Retrieval depth.
            score_threshold: Minimum similarity threshold.
            model_override: Optional model name.
            
        Returns:
            Dictionary containing answer, citations, retrieved_context, and model metadata.
        """
        # 1. Retrieve relevant evidence chunks
        retrieved_context = self.retriever.retrieve(
            query=user_query,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        # 2. Generate grounded answer
        llm_response = self.llm_service.generate_answer(
            query=user_query,
            retrieved_context=retrieved_context,
            history=history,
            model_override=model_override,
        )

        return {
            "query": user_query,
            "answer": llm_response["answer"],
            "citations": llm_response["citations"],
            "is_fallback": llm_response.get("is_fallback", False),
            "model_used": llm_response.get("model_used", "unknown"),
            "total_chunks_retrieved": len(retrieved_context.passages),
            "raw_passages": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                }
                for doc, score in retrieved_context.passages
            ],
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Returns comprehensive vector store and LLM service status."""
        vector_stats = self.vector_store.get_stats()
        llm_health = self.llm_service.check_health()
        return {
            "vector_store": vector_stats,
            "llm_service": llm_health,
        }

    def clear_index(self) -> bool:
        """Clears vector database collection."""
        return self.vector_store.clear_collection()

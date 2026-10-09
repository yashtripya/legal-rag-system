"""
Vector Store Module for Legal RAG System.
Manages persistent ChromaDB vector storage, document indexing, deduplication, and similarity search.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import chromadb
from chromadb.config import Settings

from config import CHROMA_DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME
from src.document_loader import LegalDocument
from src.embeddings import LocalLegalEmbeddings


class VectorStoreError(Exception):
    """Custom exception raised when vector store operations fail."""
    pass


class LegalVectorStore:
    """
    Manages persistent vector storage using ChromaDB.
    Handles adding chunks, querying top-k matches, deduplication, and collection stats.
    """

    def __init__(
        self,
        persist_dir: str | Path = CHROMA_DB_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_model: Optional[LocalLegalEmbeddings] = None,
    ):
        self.persist_dir = str(Path(persist_dir).resolve())
        self.collection_name = collection_name
        self.embeddings = embedding_model or LocalLegalEmbeddings()

        # Initialize ChromaDB persistent client
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        # Get or create collection
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> chromadb.Collection:
        """Initializes or retrieves the ChromaDB collection with model metadata validation."""
        metadata = {"embedding_model": EMBEDDING_MODEL_NAME, "domain": "Indian Legal Law"}
        
        try:
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata=metadata,
            )
            return collection
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB collection '{self.collection_name}': {str(e)}")

    def add_documents(self, documents: List[LegalDocument]) -> Dict[str, Any]:
        """
        Adds chunked LegalDocument instances to the ChromaDB vector store with deduplication.
        
        Args:
            documents: List of LegalDocument chunks to index.
            
        Returns:
            Dictionary summary with added_count, skipped_count, and total_chunks.
        """
        if not documents:
            return {"added_count": 0, "skipped_count": 0, "total_chunks": 0}

        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        added_count = 0
        skipped_count = 0

        # Existing IDs in collection for deduplication
        existing_ids = set()
        try:
            existing_records = self.collection.get(include=[])
            if existing_records and "ids" in existing_records:
                existing_ids = set(existing_records["ids"])
        except Exception:
            pass

        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                h = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:16]
                chunk_id = f"chunk_{h}"

            if chunk_id in existing_ids:
                skipped_count += 1
                continue

            ids.append(chunk_id)
            texts.append(doc.page_content)
            
            # Clean metadata values to primitive types supported by Chroma (str, int, float, bool)
            clean_meta = {}
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                else:
                    clean_meta[k] = str(v)
            metadatas.append(clean_meta)
            added_count += 1

        if ids:
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)

            # Upsert into ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

        return {
            "added_count": added_count,
            "skipped_count": skipped_count,
            "total_chunks": self.collection.count(),
        }

    def similarity_search_with_score(
        self, query: str, top_k: int = 4
    ) -> List[Tuple[LegalDocument, float]]:
        """
        Performs semantic similarity search for a query string.
        
        Args:
            query: Question or legal phrase.
            top_k: Number of nearest matches to retrieve.
            
        Returns:
            List of tuples: (LegalDocument chunk, distance score). Lower distance = higher similarity.
        """
        if not query.strip():
            return []

        query_vector = self.embeddings.embed_query(query)
        if not query_vector:
            return []

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, max(1, self.collection.count())),
            include=["documents", "metadatas", "distances"],
        )

        output: List[Tuple[LegalDocument, float]] = []

        if not results or "documents" not in results or not results["documents"][0]:
            return output

        docs = results["documents"][0]
        metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
        distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)

        for doc_text, meta, dist in zip(docs, metas, distances):
            # Convert Chroma distance to similarity score (for L2/cosine distances)
            similarity_score = max(0.0, 1.0 - float(dist)) if dist <= 1.0 else 1.0 / (1.0 + float(dist))
            output.append((LegalDocument(page_content=doc_text, metadata=meta), similarity_score))

        # Sort by similarity score descending
        output.sort(key=lambda x: x[1], reverse=True)
        return output

    def get_stats(self) -> Dict[str, Any]:
        """Returns metadata statistics about the vector collection."""
        try:
            count = self.collection.count()
            records = self.collection.get(include=["metadatas"])
            
            source_files = set()
            document_titles = set()
            
            if records and "metadatas" in records and records["metadatas"]:
                for meta in records["metadatas"]:
                    if "filename" in meta:
                        source_files.add(meta["filename"])
                    if "document_title" in meta:
                        document_titles.add(meta["document_title"])

            return {
                "total_chunks": count,
                "total_documents": len(source_files),
                "source_files": list(source_files),
                "document_titles": list(document_titles),
                "collection_name": self.collection_name,
                "embedding_model": EMBEDDING_MODEL_NAME,
            }
        except Exception as e:
            return {
                "total_chunks": 0,
                "total_documents": 0,
                "source_files": [],
                "document_titles": [],
                "error": str(e),
            }

    def clear_collection(self) -> bool:
        """Clears all indexed chunks from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection()
            return True
        except Exception:
            return False

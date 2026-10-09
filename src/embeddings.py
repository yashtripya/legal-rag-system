"""
Embeddings Engine Module for Legal RAG System.
Provides lightweight local semantic embeddings using SentenceTransformers (all-MiniLM-L6-v2).
"""

from typing import List
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE


class LocalLegalEmbeddings(Embeddings):
    """
    Wrapper around SentenceTransformer providing local vector embeddings
    compatible with LangChain and ChromaDB.
    """

    _instance = None

    def __new__(cls, model_name: str = EMBEDDING_MODEL_NAME, device: str = EMBEDDING_DEVICE):
        if cls._instance is None:
            cls._instance = super(LocalLegalEmbeddings, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.device = device
            cls._instance.model = SentenceTransformer(model_name, device=device)
        return cls._instance

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of document strings into numerical dense vectors.
        
        Args:
            texts: List of text chunks to embed.
            
        Returns:
            List of float vector lists (384 dimensions for all-MiniLM-L6-v2).
        """
        if not texts:
            return []
        
        # Normalize embeddings for cosine similarity
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embeds a single query string into a numerical vector.
        
        Args:
            text: Query string.
            
        Returns:
            Float vector list.
        """
        if not text:
            return []
        
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()

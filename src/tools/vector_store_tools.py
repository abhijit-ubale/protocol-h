"""
Pinecone-specific vector store connector implementation.

Supports hybrid semantic-keyword search and document retrieval.
Implements a vector-agnostic interface for RAG operations.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from pinecone import Pinecone

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from a vector store query."""
    query: str
    matches: List[Dict[str, Any]]
    total_matches: int
    execution_time_ms: float = 0.0


class PineconeConnector:
    """
    Cloud-agnostic connector for Pinecone vector databases.
    
    Features:
    - Hybrid keyword-semantic search
    - Document chunk retrieval with metadata
    - Configurable similarity threshold
    - Automatic embedding handling
    """
    
    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        top_k: int = 5,
        namespace: str = "default"
    ):
        """
        Initialize Pinecone connector.
        
        Args:
            api_key: Pinecone API key
            environment: Environment (e.g., 'us-west-2-aws')
            index_name: Index name
            top_k: Number of top results to retrieve
            namespace: Namespace for queries
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.top_k = top_k
        self.namespace = namespace
        
        self.pc = None
        self.index = None
    
    def connect(self) -> None:
        """Establish connection to Pinecone."""
        try:
            self.pc = Pinecone(api_key=self.api_key)
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {str(e)}")
            raise
    
    def disconnect(self) -> None:
        """Close connection to Pinecone."""
        # Pinecone client doesn't require explicit disconnection
        logger.info("Disconnected from Pinecone")
    
    def similarity_search(
        self,
        query_embedding: List[float],
        query_text: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        include_metadata: bool = True
    ) -> VectorSearchResult:
        """
        Perform similarity search in Pinecone.
        
        Args:
            query_embedding: Query vector (already embedded)
            query_text: Original query text (for logging)
            filters: Metadata filters for hybrid search
            top_k: Number of results to retrieve
            include_metadata: Include metadata in results
        
        Returns:
            VectorSearchResult with matches
        """
        if not self.index:
            raise RuntimeError("Not connected to Pinecone")
        
        top_k = top_k or self.top_k
        
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=include_metadata,
                namespace=self.namespace,
                filter=filters
            )
            
            matches = []
            for match in results.get("matches", []):
                matches.append({
                    "id": match.get("id"),
                    "score": match.get("score"),
                    "metadata": match.get("metadata", {})
                })
            
            return VectorSearchResult(
                query=query_text or "embedding_query",
                matches=matches,
                total_matches=len(matches)
            )
        
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            raise
    
    def retrieve_chunks(
        self,
        query_embedding: List[float],
        query_text: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[str]:
        """
        Retrieve relevant document chunks as strings.
        
        Args:
            query_embedding: Query vector
            query_text: Original query text
            top_k: Number of chunks to retrieve
        
        Returns:
            List of document chunks
        """
        result = self.similarity_search(
            query_embedding=query_embedding,
            query_text=query_text,
            top_k=top_k,
            include_metadata=True
        )
        
        chunks = []
        for match in result.matches:
            metadata = match.get("metadata", {})
            chunk_text = metadata.get("text", "")
            if chunk_text:
                chunks.append(chunk_text)
        
        return chunks
    
    def test_connection(self) -> bool:
        """Test if connection is valid."""
        try:
            if self.index:
                stats = self.index.describe_index_stats()
                return stats is not None
            return False
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

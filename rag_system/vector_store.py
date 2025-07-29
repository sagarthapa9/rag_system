
# rag_system/vector_store.py
"""
Vector store implementation using Qdrant for the RAG system.
Handles vector storage, indexing, and similarity search.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """
    Vector store implementation using Qdrant database.
    Handles document storage, indexing, and similarity search.
    """
    
    def __init__(self, host: str = "localhost", port: int = 6333,
                 collection_name: str = "rag_documents", vector_size: int = 384,
                 distance_metric: str = "cosine"):
        """
        Initialize Qdrant vector store.
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to use
            vector_size: Dimension of the vectors
            distance_metric: Distance metric for similarity ("cosine", "euclidean", "dot")
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is required for vector storage. "
                "Install with: pip install qdrant-client"
            )
        
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Map string distance to Qdrant Distance enum
        distance_map = {
            "cosine": Distance.COSINE,
            "euclidean": Distance.EUCLID,
            "dot": Distance.DOT
        }
        self.distance = distance_map.get(distance_metric.lower(), Distance.COSINE)
        
        # Initialize client
        try:
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"Connected to Qdrant at {host}:{port}")
            
            # Create or verify collection
            self._setup_collection()
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise
    
    def _setup_collection(self):
        """Create collection if it doesn't exist or verify existing one."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create new collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance
                    )
                )
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                # Verify existing collection configuration
                collection_info = self.client.get_collection(self.collection_name)
                existing_size = collection_info.config.params.vectors.size
                
                if existing_size != self.vector_size:
                    logger.warning(
                        f"Collection {self.collection_name} exists with different vector size "
                        f"({existing_size} vs {self.vector_size}). Consider using a different collection name."
                    )
                
                logger.info(f"Using existing collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error setting up collection: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add documents with embeddings to the vector store.
        
        Args:
            documents: List of documents, each containing 'text', 'embedding', and metadata
            
        Returns:
            List of document IDs that were added
        """
        if not documents:
            return []
        
        points = []
        doc_ids = []
        
        for doc in documents:
            # Generate unique ID if not provided
            doc_id = doc.get('doc_id', str(uuid.uuid4()))
            doc_ids.append(doc_id)
            
            # Prepare payload (metadata)
            payload = {
                'text': doc['text'],
                'source': doc.get('source', ''),
                'chunk_id': doc.get('chunk_id', 0),
                'file_type': doc.get('file_type', ''),
                'chunk_size': doc.get('chunk_size', len(doc['text']))
            }
            
            # Create point
            point = PointStruct(
                id=doc_id,
                vector=doc['embedding'],
                payload=payload
            )
            points.append(point)
        
        try:
            # Upload points to Qdrant
            operation_info = self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Added {len(points)} documents to collection {self.collection_name}")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def search(self, query_vector: List[float], top_k: int = 5,
               score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of matching documents with scores and metadata
        """
        try:
            # Perform vector search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for hit in search_result:
                result = {
                    'id': hit.id,
                    'score': hit.score,
                    'text': hit.payload.get('text', ''),
                    'source': hit.payload.get('source', ''),
                    'chunk_id': hit.payload.get('chunk_id', 0),
                    'file_type': hit.payload.get('file_type', ''),
                    'chunk_size': hit.payload.get('chunk_size', 0)
                }
                results.append(result)
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise
    
    def search_with_filter(self, query_vector: List[float], 
                          filters: Dict[str, Any],
                          top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search with metadata filters.
        
        Args:
            query_vector: Query embedding vector
            filters: Dictionary of filters to apply
            top_k: Number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            # Build Qdrant filter
            filter_conditions = []
            
            for key, value in filters.items():
                if isinstance(value, str):
                    condition = models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                elif isinstance(value, list):
                    condition = models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value)
                    )
                else:
                    condition = models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                filter_conditions.append(condition)
            
            query_filter = models.Filter(
                must=filter_conditions
            ) if filter_conditions else None
            
            # Perform filtered search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k
            )
            
            # Format results
            results = []
            for hit in search_result:
                result = {
                    'id': hit.id,
                    'score': hit.score,
                    'text': hit.payload.get('text', ''),
                    'source': hit.payload.get('source', ''),
                    'chunk_id': hit.payload.get('chunk_id', 0),
                    'file_type': hit.payload.get('file_type', ''),
                    'chunk_size': hit.payload.get('chunk_size', 0)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in filtered search: {str(e)}")
            raise
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        Delete documents by their IDs.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=doc_ids
                )
            )
            logger.info(f"Deleted {len(doc_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def delete_by_source(self, source_path: str) -> bool:
        """
        Delete all documents from a specific source file.
        
        Args:
            source_path: Path of the source file
            
        Returns:
            True if successful
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value=source_path)
                            )
                        ]
                    )
                )
            )
            logger.info(f"Deleted documents from source: {source_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents by source: {str(e)}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                'name': self.collection_name,
                'vectors_count': collection_info.vectors_count,
                'points_count': collection_info.points_count,
                'segments_count': collection_info.segments_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance.name,
                'status': collection_info.status.name
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self._setup_collection()
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False
    
    def health_check(self) -> bool:
        """
        Check if the Qdrant service is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get collections - this will fail if service is down
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
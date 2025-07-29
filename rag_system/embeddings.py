# rag_system/embeddings.py
"""
Embedding generation module for the RAG system.
Handles text-to-vector conversion using sentence transformers.
"""

import logging
from typing import List, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for text using sentence-transformers models.
    Supports various pre-trained models with different sizes and capabilities.
    """
    
    # Popular models with their dimensions
    MODEL_DIMENSIONS = {
        'all-MiniLM-L6-v2': 384,
        'all-mpnet-base-v2': 768,
        'all-distilroberta-v1': 768,
        'all-MiniLM-L12-v2': 384,
        'paraphrase-MiniLM-L6-v2': 384,
        'paraphrase-mpnet-base-v2': 768,
    }
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run the model on ('cpu', 'cuda', etc.)
                   If None, automatically selects best available device
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for embedding generation. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading embedding model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.vector_size = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Model loaded successfully. Vector dimension: {self.vector_size}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floating point values representing the embedding
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # Ensure it's a list of floats
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str], 
                                 batch_size: int = 32,
                                 show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of input texts to embed
            batch_size: Number of texts to process in each batch
            show_progress: Whether to show progress bar
            
        Returns:
            List of embeddings, one for each input text
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        if len(valid_texts) != len(texts):
            logger.warning(f"Filtered out {len(texts) - len(valid_texts)} empty texts")
        
        if not valid_texts:
            return []
        
        try:
            logger.info(f"Generating embeddings for {len(valid_texts)} texts")
            
            # Generate embeddings in batches
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_tensor=False
            )
            
            # Convert to list format
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        emb1 = self.generate_embedding(text1)
        emb2 = self.generate_embedding(text2)
        
        return self._cosine_similarity(emb1, emb2)
    
    def compute_similarities_batch(self, query_text: str, 
                                  candidate_texts: List[str]) -> List[float]:
        """
        Compute similarities between a query and multiple candidates.
        
        Args:
            query_text: Query text to compare against
            candidate_texts: List of candidate texts
            
        Returns:
            List of similarity scores
        """
        query_emb = self.generate_embedding(query_text)
        candidate_embs = self.generate_embeddings_batch(candidate_texts)
        
        similarities = []
        for candidate_emb in candidate_embs:
            similarity = self._cosine_similarity(query_emb, candidate_emb)
            similarities.append(similarity)
        
        return similarities
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        # Convert to numpy arrays for easier computation
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'vector_size': self.vector_size,
            'device': str(self.model.device) if hasattr(self.model, 'device') else 'unknown',
            'max_sequence_length': getattr(self.model, 'max_seq_length', 'unknown')
        }
    
    @classmethod
    def list_available_models(cls) -> List[str]:
        """
        List commonly available sentence-transformer models.
        
        Returns:
            List of model names
        """
        return list(cls.MODEL_DIMENSIONS.keys())
    
    @classmethod
    def get_model_dimension(cls, model_name: str) -> int:
        """
        Get the vector dimension for a given model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Vector dimension, or None if unknown
        """
        return cls.MODEL_DIMENSIONS.get(model_name)
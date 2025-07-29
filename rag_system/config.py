# rag_system/config.py
"""
Configuration management for the RAG system.
Handles loading and validation of system settings.
"""

import os
import yaml
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration settings for the RAG system."""
    
    # Document processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "rag_documents"
    
    # LLM settings
    llm_model: str = "microsoft/DialoGPT-medium"  # Placeholder - can be replaced
    max_tokens: int = 512
    temperature: float = 0.7
    
    # Retrieval settings
    default_top_k: int = 5
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from file or environment variables.
        
        Args:
            config_path: Path to YAML configuration file
        """
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
        
        # Override with environment variables if present
        self._load_from_env()
    
    def _load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update attributes from config file
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {str(e)}")
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_mappings = {
            'RAG_CHUNK_SIZE': ('chunk_size', int),
            'RAG_CHUNK_OVERLAP': ('chunk_overlap', int),
            'RAG_EMBEDDING_MODEL': ('embedding_model', str),
            'RAG_QDRANT_HOST': ('qdrant_host', str),
            'RAG_QDRANT_PORT': ('qdrant_port', int),
            'RAG_COLLECTION_NAME': ('collection_name', str),
            'RAG_LLM_MODEL': ('llm_model', str),
            'RAG_MAX_TOKENS': ('max_tokens', int),
            'RAG_TEMPERATURE': ('temperature', float),
            'RAG_DEFAULT_TOP_K': ('default_top_k', int),
        }
        
        for env_var, (attr_name, attr_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if attr_type == bool:
                        setattr(self, attr_name, env_value.lower() in ('true', '1', 'yes'))
                    else:
                        setattr(self, attr_name, attr_type(env_value))
                except ValueError as e:
                    print(f"Warning: Invalid value for {env_var}: {env_value}")
    
    def save_to_file(self, config_path: str):
        """Save current configuration to YAML file."""
        config_dict = {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'embedding_model': self.embedding_model,
            'qdrant_host': self.qdrant_host,
            'qdrant_port': self.qdrant_port,
            'collection_name': self.collection_name,
            'llm_model': self.llm_model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'default_top_k': self.default_top_k,
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def validate(self):
        """Validate configuration settings."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        if self.qdrant_port <= 0 or self.qdrant_port > 65535:
            raise ValueError("qdrant_port must be between 1 and 65535")
        
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        
        if self.default_top_k <= 0:
            raise ValueError("default_top_k must be positive")
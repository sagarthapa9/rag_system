# rag_system/document_processor.py
"""
Document processing module for the RAG system.
Handles loading and chunking of various document formats.
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

# Document processing libraries
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document as DocxDocument
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Handles document loading and text chunking for various file formats.
    Supports: .txt, .pdf, .docx, .md
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Maximum size of text chunks in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Check available libraries
        if not PDF_AVAILABLE:
            logger.warning("PyPDF2 not available. PDF processing disabled.")
        if not DOCX_AVAILABLE:
            logger.warning("python-docx not available. DOCX processing disabled.")
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a document and return chunked text.
        
        Args:
            file_path: Path to the document to process
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        file_ext = Path(file_path).suffix.lower()
        
        # Extract text based on file type
        if file_ext == '.txt':
            text = self._load_text_file(file_path)
        elif file_ext == '.md':
            text = self._load_text_file(file_path)  # Markdown is plain text
        elif file_ext == '.pdf':
            text = self._load_pdf_file(file_path)
        elif file_ext == '.docx':
            text = self._load_docx_file(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        if not text.strip():
            raise ValueError(f"No text content found in {file_path}")
        
        # Chunk the text
        chunks = self._chunk_text(text)
        
        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk in enumerate(chunks):
            chunk_objects.append({
                'text': chunk,
                'chunk_id': i,
                'source_file': file_path,
                'file_type': file_ext,
                'chunk_size': len(chunk)
            })
        
        logger.info(f"Processed {file_path}: {len(chunk_objects)} chunks generated")
        return chunk_objects
    
    def _load_text_file(self, file_path: str) -> str:
        """Load text from a plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _load_pdf_file(self, file_path: str) -> str:
        """Load text from a PDF file."""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
        
        text = ""
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            raise Exception(f"Error reading PDF {file_path}: {str(e)}")
        
        return text
    
    def _load_docx_file(self, file_path: str) -> str:
        """Load text from a DOCX file."""
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx is required for DOCX processing. Install with: pip install python-docx")
        
        try:
            doc = DocxDocument(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            raise Exception(f"Error reading DOCX {file_path}: {str(e)}")
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end of the current chunk
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at a sentence or word boundary
            if end < len(text):
                # Look for sentence boundaries (. ! ?)
                sentence_end = max(
                    text.rfind('.', start, end),
                    text.rfind('!', start, end),
                    text.rfind('?', start, end)
                )
                
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Fall back to word boundary
                    word_end = text.rfind(' ', start, end)
                    if word_end > start + self.chunk_size // 2:
                        end = word_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            
            # Ensure we don't go backwards
            if start <= chunks and len(chunks) > 0:
                break
        
        return chunks
    
    def get_supported_formats(self) -> List[str]:
        """Return list of supported file formats."""
        formats = ['.txt', '.md']
        
        if PDF_AVAILABLE:
            formats.append('.pdf')
        
        if DOCX_AVAILABLE:
            formats.append('.docx')
        
        return formats
    
    def batch_process_directory(self, directory_path: str, 
                              recursive: bool = False) -> List[Dict[str, Any]]:
        """
        Process all supported documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            recursive: Whether to process subdirectories
            
        Returns:
            List of all chunks from all processed documents
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")
        
        supported_formats = self.get_supported_formats()
        all_chunks = []
        
        # Get file pattern
        pattern = "**/*" if recursive else "*"
        
        for file_path in Path(directory_path).glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                try:
                    chunks = self.process_document(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {str(e)}")
                    continue
        
        logger.info(f"Batch processed {directory_path}: {len(all_chunks)} total chunks")
        return all_chunks
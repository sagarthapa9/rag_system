#!/usr/bin/env python3
"""
RAG System with Qdrant Vector Database
A complete boilerplate for document ingestion, vectorization, and Q&A
"""

import os
import argparse
import logging
from typing import List, Dict, Any
from pathlib import Path

from rag_system.document_processor import DocumentProcessor
from rag_system.vector_store import QdrantVectorStore
from rag_system.embeddings import EmbeddingGenerator
from rag_system.llm_interface import LLMInterface
from rag_system.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Main RAG System class that orchestrates document processing,
    vector storage, and question-answering capabilities.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the RAG system with configuration."""
        self.config = Config(config_path)
        
        # Initialize components
        self.doc_processor = DocumentProcessor(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        self.embedding_generator = EmbeddingGenerator(
            model_name=self.config.embedding_model
        )
        
        self.vector_store = QdrantVectorStore(
            host=self.config.qdrant_host,
            port=self.config.qdrant_port,
            collection_name=self.config.collection_name,
            vector_size=self.embedding_generator.vector_size
        )
        
        self.llm = LLMInterface(
            model_name=self.config.llm_model,
            max_tokens=self.config.max_tokens
        )
        
        logger.info("RAG System initialized successfully")
    
    def ingest_documents(self, document_paths: List[str]) -> Dict[str, Any]:
        """
        Ingest documents into the vector database.
        
        Args:
            document_paths: List of paths to documents to ingest
            
        Returns:
            Dictionary with ingestion statistics
        """
        logger.info(f"Starting document ingestion for {len(document_paths)} documents")
        
        all_chunks = []
        processed_docs = 0
        
        for doc_path in document_paths:
            try:
                logger.info(f"Processing document: {doc_path}")
                
                # Load and chunk document
                chunks = self.doc_processor.process_document(doc_path)
                
                # Generate embeddings for chunks
                for i, chunk in enumerate(chunks):
                    embedding = self.embedding_generator.generate_embedding(chunk['text'])
                    chunk['embedding'] = embedding
                    chunk['doc_id'] = f"{Path(doc_path).stem}_{i}"
                    chunk['source'] = doc_path
                
                all_chunks.extend(chunks)
                processed_docs += 1
                
            except Exception as e:
                logger.error(f"Error processing {doc_path}: {str(e)}")
                continue
        
        # Store embeddings in Qdrant
        if all_chunks:
            self.vector_store.add_documents(all_chunks)
            logger.info(f"Successfully ingested {len(all_chunks)} chunks from {processed_docs} documents")
        
        return {
            "processed_documents": processed_docs,
            "total_chunks": len(all_chunks),
            "failed_documents": len(document_paths) - processed_docs
        }
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Process a query and return an answer with sources.
        
        Args:
            question: User's question
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary containing answer and metadata
        """
        logger.info(f"Processing query: {question}")
        
        try:
            # Generate embedding for the question
            question_embedding = self.embedding_generator.generate_embedding(question)
            
            # Retrieve relevant chunks
            relevant_chunks = self.vector_store.search(question_embedding, top_k=top_k)
            
            if not relevant_chunks:
                return {
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Prepare context for LLM
            context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
            
            # Generate answer using LLM
            answer = self.llm.generate_answer(question, context)
            
            # Prepare sources
            sources = [
                {
                    "source": chunk['source'],
                    "score": chunk['score'],
                    "text_preview": chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                }
                for chunk in relevant_chunks
            ]
            
            logger.info("Query processed successfully")
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": relevant_chunks[0]['score'] if relevant_chunks else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def interactive_mode(self):
        """Run the system in interactive Q&A mode."""
        print("\n🤖 RAG System Interactive Mode")
        print("Ask questions about your ingested documents. Type 'quit' to exit.\n")
        
        while True:
            try:
                question = input("❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\n🔍 Searching for relevant information...")
                result = self.query(question)
                
                print(f"\n💡 Answer: {result['answer']}")
                print(f"🎯 Confidence: {result['confidence']:.2f}")
                
                if result['sources']:
                    print(f"\n📚 Sources ({len(result['sources'])} found):")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"  {i}. {source['source']} (score: {source['score']:.3f})")
                        print(f"     Preview: {source['text_preview']}")
                
                print("\n" + "-" * 80 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")


def main():
    """Main CLI interface for the RAG system."""
    parser = argparse.ArgumentParser(description="RAG System with Qdrant")
    parser.add_argument(
        "--config", 
        default="config.yaml", 
        help="Path to configuration file"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument(
        "documents", 
        nargs="+", 
        help="Paths to documents to ingest"
    )
    
    # Query subcommand
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument(
        "--top-k", 
        type=int, 
        default=5, 
        help="Number of relevant chunks to retrieve"
    )
    
    # Interactive subcommand
    subparsers.add_parser("interactive", help="Start interactive Q&A mode")
    
    # Status subcommand
    subparsers.add_parser("status", help="Show system status")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        # Initialize RAG system
        rag = RAGSystem(args.config)
        
        if args.command == "ingest":
            # Validate document paths
            valid_paths = []
            for path in args.documents:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    logger.warning(f"Document not found: {path}")
            
            if not valid_paths:
                print("❌ No valid documents found to ingest")
                return
            
            # Ingest documents
            stats = rag.ingest_documents(valid_paths)
            print(f"✅ Ingestion complete!")
            print(f"   📄 Processed documents: {stats['processed_documents']}")
            print(f"   📝 Total chunks: {stats['total_chunks']}")
            if stats['failed_documents'] > 0:
                print(f"   ❌ Failed documents: {stats['failed_documents']}")
        
        elif args.command == "query":
            result = rag.query(args.question, args.top_k)
            print(f"\n💡 Answer: {result['answer']}")
            print(f"🎯 Confidence: {result['confidence']:.2f}")
            
            if result['sources']:
                print(f"\n📚 Sources:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"  {i}. {source['source']} (score: {source['score']:.3f})")
        
        elif args.command == "interactive":
            rag.interactive_mode()
        
        elif args.command == "status":
            # Show system status
            collection_info = rag.vector_store.get_collection_info()
            print(f"📊 RAG System Status:")
            print(f"   🗄️  Collection: {rag.config.collection_name}")
            print(f"   📄 Documents: {collection_info.get('vectors_count', 0)}")
            print(f"   🤖 Embedding model: {rag.config.embedding_model}")
            print(f"   🧠 LLM model: {rag.config.llm_model}")
            print(f"   🌐 Qdrant: {rag.config.qdrant_host}:{rag.config.qdrant_port}")
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
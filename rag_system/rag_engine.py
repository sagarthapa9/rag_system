# rag_system/rag_engine.py
"""
Simple RAG implementation without any caching.
Clean, straightforward, and easy to understand.
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

from .config import Config
from .document_processor import DocumentProcessor, Document
from .embeddings import EmbeddingModel
from .vector_store import QdrantVectorStore
from .llm_interface import OpenAIClient


class SimpleRAGEngine:
    """
    A clean RAG implementation without caching.
    Every operation is performed fresh each time.
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize components
        self.document_processor = DocumentProcessor(config)
        self.embedding_model = EmbeddingModel(
            model_name=config.embeddings["model_name"],
            device=config.embeddings["device"]
        )
        self.vector_store = QdrantVectorStore(config, self.embedding_model)
        self.llm_client = OpenAIClient(
            api_key=config.llm["api_key"],
            model=config.llm["model"],
            temperature=config.llm["temperature"]
        )
    
    async def initialize(self) -> None:
        """Initialize the RAG system."""
        await self.vector_store.create_collection()
        print("✅ RAG system initialized successfully")
    
    async def add_document(self, file_path: Path) -> List[str]:
        """
        Add a single document to the system.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document IDs that were added
        """
        print(f"📄 Processing document: {file_path.name}")
        
        # Process document into chunks
        documents = self.document_processor.process_file(file_path)
        print(f"📝 Created {len(documents)} chunks")
        
        # Add to vector store
        doc_ids = await self.vector_store.add_documents(documents)
        print(f"✅ Added {len(doc_ids)} document chunks to vector store")
        
        return doc_ids
    
    async def add_documents(self, file_paths: List[Path]) -> List[str]:
        """
        Add multiple documents to the system.
        
        Args:
            file_paths: List of paths to document files
            
        Returns:
            List of all document IDs that were added
        """
        all_doc_ids = []
        
        for file_path in file_paths:
            try:
                doc_ids = await self.add_document(file_path)
                all_doc_ids.extend(doc_ids)
            except Exception as e:
                print(f"❌ Error processing {file_path.name}: {e}")
        
        return all_doc_ids
    
    async def query(
        self, 
        question: str, 
        top_k: int = 5,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: The question to ask
            top_k: Number of relevant documents to retrieve
            include_sources: Whether to include source information
            
        Returns:
            Dictionary containing answer and context
        """
        print(f"🔍 Searching for: {question}")
        
        # 1. Search for relevant documents (no caching)
        search_results = await self.vector_store.search(
            query=question, 
            top_k=top_k
        )
        
        if not search_results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "context": [],
                "sources": []
            }
        
        print(f"📚 Found {len(search_results)} relevant documents")
        
        # 2. Prepare context from search results
        context_texts = []
        sources = []
        
        for result in search_results:
            context_texts.append(result["payload"]["content"])
            if include_sources:
                sources.append({
                    "content": result["payload"]["content"][:200] + "...",
                    "score": result["score"],
                    "metadata": result["payload"].get("metadata", {})
                })
        
        # 3. Create prompt with context
        context = "\n\n".join(context_texts)
        prompt = self._create_prompt(question, context)
        
        # 4. Get LLM response (no caching)
        print("🤖 Generating response...")
        answer = await self.llm_client.generate_response(prompt)
        
        return {
            "answer": answer,
            "context": context_texts,
            "sources": sources if include_sources else [],
            "total_sources": len(search_results)
        }
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create a prompt for the LLM with context."""
        return f"""You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer the question based on the provided context
- If the context doesn't contain enough information, say so
- Be concise but comprehensive
- Don't make up information not in the context

Answer:"""
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        return await self.vector_store.get_collection_info()
    
    async def delete_collection(self) -> None:
        """Delete the entire collection."""
        await self.vector_store.delete_collection()
        print("🗑️ Collection deleted")
    
    async def health_check(self) -> Dict[str, str]:
        """Check if all components are healthy."""
        status = {}
        
        try:
            # Check vector store
            await self.vector_store.client.get_collections()
            status["vector_store"] = "healthy"
        except Exception as e:
            status["vector_store"] = f"error: {e}"
        
        try:
            # Check embedding model
            test_embedding = self.embedding_model.encode(["test"])
            status["embedding_model"] = "healthy"
        except Exception as e:
            status["embedding_model"] = f"error: {e}"
        
        try:
            # Check LLM (simple test)
            test_response = await self.llm_client.generate_response("Say 'OK'")
            status["llm"] = "healthy" if test_response else "error: no response"
        except Exception as e:
            status["llm"] = f"error: {e}"
        
        return status


# Simple CLI without caching
class SimpleRAGCLI:
    """Simple command-line interface for the RAG system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = Config.from_yaml(config_path)
        self.rag_engine = SimpleRAGEngine(self.config)
    
    async def run_interactive(self):
        """Run interactive query session."""
        await self.rag_engine.initialize()
        
        print("\n🚀 Welcome to Simple RAG System!")
        print("Type 'quit' to exit, 'help' for commands\n")
        
        while True:
            try:
                user_input = input("❓ Ask a question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                if user_input.lower() == 'stats':
                    stats = await self.rag_engine.get_collection_stats()
                    print(f"📊 Collection stats: {stats}")
                    continue
                
                if not user_input:
                    continue
                
                # Process query
                print("🤔 Thinking...")
                result = await self.rag_engine.query(user_input)
                
                print(f"\n💡 Answer:\n{result['answer']}\n")
                
                if result['sources']:
                    print(f"📚 Found {result['total_sources']} relevant sources")
                    print("🔗 Top sources:")
                    for i, source in enumerate(result['sources'][:3], 1):
                        print(f"  {i}. {source['content']} (score: {source['score']:.3f})")
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_help(self):
        """Show help information."""
        print("""
Available commands:
- Type any question to get an answer
- 'stats' - Show collection statistics  
- 'help' - Show this help message
- 'quit' - Exit the system
        """)
    
    async def add_documents_from_directory(self, directory_path: str):
        """Add all documents from a directory."""
        directory = Path(directory_path)
        if not directory.exists():
            print(f"❌ Directory not found: {directory_path}")
            return
        
        # Find all supported files
        supported_extensions = {'.txt', '.pdf', '.docx', '.md', '.json'}
        files = [
            f for f in directory.rglob('*') 
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        
        if not files:
            print(f"❌ No supported files found in {directory_path}")
            return
        
        print(f"📁 Found {len(files)} files to process...")
        
        await self.rag_engine.initialize()
        doc_ids = await self.rag_engine.add_documents(files)
        
        print(f"✅ Successfully added {len(doc_ids)} document chunks")


# Example usage
async def main():
    """Example of how to use the simple RAG system."""
    
    # Initialize RAG system
    rag = SimpleRAGEngine(config=Config())
    await rag.initialize()
    
    # Add some documents
    document_files = [
        Path("sample_documents/doc1.txt"),
        Path("sample_documents/doc2.pdf"),
    ]
    
    for doc_file in document_files:
        if doc_file.exists():
            await rag.add_document(doc_file)
    
    # Query the system
    questions = [
        "What is machine learning?",
        "How does deep learning work?",
        "What are the benefits of AI?"
    ]
    
    for question in questions:
        print(f"\n🔍 Question: {question}")
        result = await rag.query(question)
        print(f"💡 Answer: {result['answer']}")
        print(f"📚 Used {result['total_sources']} sources")


if __name__ == "__main__":
    asyncio.run(main())
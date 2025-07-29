#!/usr/bin/env python3
"""
Modern CLI interface for RAG System using Click and Rich
Provides an enhanced user experience with better formatting and progress bars
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich import print as rprint

from .config import Config
from .document_processor import DocumentProcessor
from .embeddings import EmbeddingGenerator
from .vector_store import QdrantVectorStore
from .llm_interface import LLMInterface

console = Console()


class RAGSystem:
    """Enhanced RAG System with rich console output."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the RAG system with configuration."""
        self.config = Config(config_path)
        
        with console.status("[bold green]Initializing RAG System..."):
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
        
        console.print("✅ [bold green]RAG System initialized successfully[/bold green]")
    
    def ingest_documents_with_progress(self, document_paths: List[str]) -> dict:
        """Ingest documents with progress tracking."""
        
        # Validate paths first
        valid_paths = []
        for path in document_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                console.print(f"[yellow]⚠️ Document not found: {path}[/yellow]")
        
        if not valid_paths:
            console.print("[red]❌ No valid documents found to ingest[/red]")
            return {"processed_documents": 0, "total_chunks": 0, "failed_documents": len(document_paths)}
        
        all_chunks = []
        processed_docs = 0
        failed_docs = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            
            # Main ingestion task
            main_task = progress.add_task("[cyan]Processing documents...", total=len(valid_paths))
            
            for doc_path in valid_paths:
                try:
                    progress.update(main_task, description=f"[cyan]Processing: {Path(doc_path).name}")
                    
                    # Load and chunk document
                    chunks = self.doc_processor.process_document(doc_path)
                    
                    # Subtask for embeddings
                    embed_task = progress.add_task(f"[yellow]Generating embeddings...", total=len(chunks))
                    
                    # Generate embeddings for chunks
                    for i, chunk in enumerate(chunks):
                        embedding = self.embedding_generator.generate_embedding(chunk['text'])
                        chunk['embedding'] = embedding
                        chunk['doc_id'] = f"{Path(doc_path).stem}_{i}"
                        chunk['source'] = doc_path
                        progress.update(embed_task, advance=1)
                    
                    progress.remove_task(embed_task)
                    all_chunks.extend(chunks)
                    processed_docs += 1
                    
                except Exception as e:
                    console.print(f"[red]❌ Error processing {doc_path}: {str(e)}[/red]")
                    failed_docs += 1
                    continue
                finally:
                    progress.update(main_task, advance=1)
        
        # Store embeddings in Qdrant
        if all_chunks:
            with console.status("[bold green]Storing vectors in Qdrant..."):
                self.vector_store.add_documents(all_chunks)
            
            console.print(f"[green]✅ Successfully ingested {len(all_chunks)} chunks from {processed_docs} documents[/green]")
        
        return {
            "processed_documents": processed_docs,
            "total_chunks": len(all_chunks),
            "failed_documents": failed_docs
        }
    
    def query_with_formatting(self, question: str, top_k: int = 5) -> dict:
        """Process query with rich formatting."""
        
        with console.status(f"[bold yellow]Searching for: {question}"):
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
        
        with console.status("[bold blue]Generating answer..."):
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
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": relevant_chunks[0]['score'] if relevant_chunks else 0.0
        }


@click.group()
@click.option('--config', default='config.yaml', help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.pass_context
def cli(ctx, config, verbose):
    """🤖 RAG System with Qdrant Vector Database
    
    A complete system for document ingestion, vectorization, and Q&A.
    """
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['verbose'] = verbose
    
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)


@cli.command()
@click.argument('documents', nargs=-1, required=True)
@click.option('--recursive', '-r', is_flag=True, help='Process directories recursively')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table')
@click.pass_context
def ingest(ctx, documents, recursive, output_format):
    """📄 Ingest documents into the vector database.
    
    DOCUMENTS: Paths to documents or directories to ingest
    """
    
    console.print(Panel.fit("📄 [bold]Document Ingestion[/bold]", border_style="blue"))
    
    try:
        rag = RAGSystem(ctx.obj['config'])
        
        # Expand paths if directories
        all_paths = []
        for doc_path in documents:
            path = Path(doc_path)
            if path.is_dir():
                if recursive:
                    pattern = "**/*"
                else:
                    pattern = "*"
                
                # Get supported formats
                supported = rag.doc_processor.get_supported_formats()
                for file_path in path.glob(pattern):
                    if file_path.is_file() and file_path.suffix.lower() in supported:
                        all_paths.append(str(file_path))
            else:
                all_paths.append(str(path))
        
        if not all_paths:
            console.print("[red]❌ No documents found to process[/red]")
            return
        
        console.print(f"[cyan]📋 Found {len(all_paths)} documents to process[/cyan]")
        
        # Process documents
        stats = rag.ingest_documents_with_progress(all_paths)
        
        # Display results
        if output_format == 'table':
            table = Table(title="📊 Ingestion Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Processed Documents", str(stats['processed_documents']))
            table.add_row("Total Chunks", str(stats['total_chunks']))
            table.add_row("Failed Documents", str(stats['failed_documents']))
            
            console.print(table)
        else:
            rprint(stats)
            
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        if ctx.obj['verbose']:
            console.print_exception()


@cli.command()
@click.argument('question')
@click.option('--top-k', default=5, help='Number of relevant chunks to retrieve')
@click.option('--format', 'output_format', type=click.Choice(['pretty', 'json']), default='pretty')
@click.pass_context
def query(ctx, question, top_k, output_format):
    """❓ Ask a question about your documents.
    
    QUESTION: The question to ask
    """
    
    try:
        rag = RAGSystem(ctx.obj['config'])
        result = rag.query_with_formatting(question, top_k)
        
        if output_format == 'pretty':
            # Pretty format output
            console.print(Panel.fit(f"❓ [bold]Question:[/bold] {question}", border_style="yellow"))
            
            console.print(Panel(
                f"[green]{result['answer']}[/green]",
                title="💡 Answer",
                border_style="green"
            ))
            
            console.print(f"🎯 [bold]Confidence:[/bold] {result['confidence']:.3f}")
            
            if result['sources']:
                console.print("\n📚 [bold]Sources:[/bold]")
                for i, source in enumerate(result['sources'], 1):
                    source_text = Text()
                    source_text.append(f"{i}. ", style="bold cyan")
                    source_text.append(f"{Path(source['source']).name} ", style="blue")
                    source_text.append(f"(score: {source['score']:.3f})", style="dim")
                    console.print(source_text)
                    console.print(f"   [dim]{source['text_preview']}[/dim]\n")
        else:
            rprint(result)
            
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")
        if ctx.obj['verbose']:
            console.print_exception()


@cli.command()
@click.pass_context
def interactive(ctx):
    """💬 Start interactive Q&A mode."""
    
    console.print(Panel.fit("💬 [bold]Interactive Q&A Mode[/bold]", border_style="magenta"))
    console.print("Ask questions about your ingested documents. Type 'quit' or 'exit' to leave.\n")
    
    try:
        rag = RAGSystem(ctx.obj['config'])
        
        while True:
            try:
                question = Prompt.ask("❓ [bold cyan]Your question[/bold cyan]")
                
                if question.lower() in ['quit', 'exit', 'q']:
                    console.print("👋 [bold]Goodbye![/bold]")
                    break
                
                if not question.strip():
                    continue
                
                result = rag.query_with_formatting(question)
                
                # Display answer
                console.print(Panel(
                    f"[green]{result['answer']}[/green]",
                    title="💡 Answer",
                    border_style="green"
                ))
                
                console.print(f"🎯 Confidence: {result['confidence']:.3f}")
                
                # Show sources if available
                if result['sources']:
                    show_sources = Confirm.ask("📚 Show sources?", default=True)
                    if show_sources:
                        for i, source in enumerate(result['sources'], 1):
                            console.print(f"[cyan]{i}. {Path(source['source']).name}[/cyan] (score: {source['score']:.3f})")
                
                console.print("\n" + "-" * 60 + "\n")
                
            except KeyboardInterrupt:
                console.print("\n👋 [bold]Goodbye![/bold]")
                break
            except Exception as e:
                console.print(f"[red]❌ Error: {str(e)}[/red]")
                
    except Exception as e:
        console.print(f"[red]❌ Initialization error: {str(e)}[/red]")


@cli.command()
@click.option('--detailed', '-d', is_flag=True, help='Show detailed system information')
@click.pass_context
def status(ctx, detailed):
    """📊 Show system status and information."""
    
    console.print(Panel.fit("📊 [bold]System Status[/bold]", border_style="blue"))
    
    try:
        rag = RAGSystem(ctx.obj['config'])
        
        # Get collection info
        collection_info = rag.vector_store.get_collection_info()
        
        # Create status table
        table = Table(title="🤖 RAG System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="dim")
        
        # System info
        table.add_row("Collection", collection_info.get('name', 'Unknown'), f"{collection_info.get('vectors_count', 0)} documents")
        table.add_row("Embedding Model", rag.config.embedding_model, f"{rag.embedding_generator.vector_size}D vectors")
        table.add_row("LLM Model", rag.config.llm_model, f"Max tokens: {rag.config.max_tokens}")
        table.add_row("Qdrant", f"{rag.config.qdrant_host}:{rag.config.qdrant_port}", collection_info.get('status', 'Unknown'))
        
        console.print(table)
        
        if detailed:
            # Show detailed configuration
            config_table = Table(title="⚙️ Configuration Details")
            config_table.add_column("Setting", style="cyan")
            config_table.add_column("Value", style="yellow")
            
            config_table.add_row("Chunk Size", str(rag.config.chunk_size))
            config_table.add_row("Chunk Overlap", str(rag.config.chunk_overlap))
            config_table.add_row("Temperature", str(rag.config.temperature))
            config_table.add_row("Default Top-K", str(rag.config.default_top_k))
            
            console.print(config_table)
            
    except Exception as e:
        console.print(f"[red]❌ Error getting status: {str(e)}[/red]")


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to clear all documents?')
@click.pass_context
def clear(ctx):
    """🗑️ Clear all documents from the vector database."""
    
    try:
        rag = RAGSystem(ctx.obj['config'])
        
        with console.status("[bold red]Clearing collection..."):
            success = rag.vector_store.clear_collection()
        
        if success:
            console.print("[green]✅ Collection cleared successfully[/green]")
        else:
            console.print("[red]❌ Failed to clear collection[/red]")
            
    except Exception as e:
        console.print(f"[red]❌ Error: {str(e)}[/red]")


@cli.command()
@click.pass_context
def health(ctx):
    """🏥 Check system health."""
    
    console.print("🏥 [bold]Health Check[/bold]\n")
    
    try:
        # Check configuration
        config = Config(ctx.obj['config'])
        console.print("[green]✅ Configuration loaded[/green]")
        
        # Check Qdrant connection
        vector_store = QdrantVectorStore(
            host=config.qdrant_host,
            port=config.qdrant_port,
            collection_name=config.collection_name,
            vector_size=384  # Default size for health check
        )
        
        if vector_store.health_check():
            console.print("[green]✅ Qdrant connection healthy[/green]")
        else:
            console.print("[red]❌ Qdrant connection failed[/red]")
        
        # Check embedding model
        try:
            embedder = EmbeddingGenerator(model_name=config.embedding_model)
            console.print("[green]✅ Embedding model loaded[/green]")
        except Exception as e:
            console.print(f"[red]❌ Embedding model error: {str(e)}[/red]")
        
        # Check LLM
        try:
            llm = LLMInterface(model_name=config.llm_model)
            console.print("[green]✅ LLM model loaded[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠️ LLM model warning: {str(e)}[/yellow]")
        
        console.print("\n[bold green]System health check completed[/bold green]")
        
    except Exception as e:
        console.print(f"[red]❌ Health check failed: {str(e)}[/red]")


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
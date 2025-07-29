# RAG System PowerShell Commands
# Usage: .\rag-commands.ps1 <command>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet(
        "help", "build", "up", "down", "logs", "shell", "test", "clean",
        "cli-help", "cli-status", "cli-health", "cli-ingest", "cli-query", 
        "cli-interactive", "cli-clear", "health", "stats"
    )]
    [string]$Command,
    
    [string]$Query = ""
)

function Show-Help {
    Write-Host "Available commands:" -ForegroundColor Green
    Write-Host "  help          - Show this help message"
    Write-Host "  build         - Build all Docker images"
    Write-Host "  up            - Start Qdrant service"
    Write-Host "  down          - Stop all services"
    Write-Host "  logs          - View logs from all services"
    Write-Host "  shell         - Get shell access to RAG CLI container"
    Write-Host "  test          - Run tests"
    Write-Host "  clean         - Clean up containers and networks"
    Write-Host "  health        - Check health of all services"
    Write-Host "  stats         - Monitor resource usage"
    Write-Host ""
    Write-Host "CLI Commands:" -ForegroundColor Cyan
    Write-Host "  cli-help      - Show CLI help"
    Write-Host "  cli-status    - Check system status"
    Write-Host "  cli-health    - Check system health"
    Write-Host "  cli-ingest    - Ingest documents"
    Write-Host "  cli-query     - Query the system (use -Query parameter)"
    Write-Host "  cli-interactive - Start interactive mode"
    Write-Host "  cli-clear     - Clear all documents"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\rag-commands.ps1 up"
    Write-Host "  .\rag-commands.ps1 cli-query -Query 'What is machine learning?'"
    Write-Host "  .\rag-commands.ps1 cli-ingest"
}

switch ($Command) {
    "help" {
        Show-Help
    }
    "build" {
        Write-Host "Building Docker images..." -ForegroundColor Green
        docker-compose build --no-cache
    }
    "up" {
        Write-Host "Starting Qdrant service..." -ForegroundColor Green
        docker-compose up -d qdrant
    }
    "down" {
        Write-Host "Stopping all services..." -ForegroundColor Green
        docker-compose down
    }
    "logs" {
        docker-compose logs -f
    }
    "shell" {
        docker-compose run --rm rag-cli /bin/bash
    }
    "test" {
        docker-compose run --rm rag-cli uv run pytest
    }
    "clean" {
        Write-Host "Cleaning up containers and networks..." -ForegroundColor Green
        docker-compose down --volumes --remove-orphans
        docker network prune -f
        docker system prune -af
    }
    "health" {
        Write-Host "Checking service health..." -ForegroundColor Green
        docker-compose ps
        Write-Host "`nQdrant Health:" -ForegroundColor Cyan
        try {
            $response = Invoke-RestMethod -Uri "http://localhost:6333/health" -TimeoutSec 5
            Write-Host "✅ Qdrant is healthy" -ForegroundColor Green
        }
        catch {
            Write-Host "❌ Qdrant not responding" -ForegroundColor Red
        }
    }
    "stats" {
        docker stats
    }
    "cli-help" {
        docker-compose run --rm rag-cli uv run python -m rag_system.cli --help
    }
    "cli-status" {
        docker-compose run --rm rag-cli uv run python -m rag_system.cli status
    }
    "cli-health" {
        docker-compose run --rm rag-cli uv run python -m rag_system.cli health
    }
    "cli-ingest" {
        docker-compose run --rm rag-cli uv run python -m rag_system.cli ingest /app/documents/ --recursive
    }
    "cli-query" {
        if ([string]::IsNullOrEmpty($Query)) {
            Write-Host "Please provide a query using -Query parameter" -ForegroundColor Red
            Write-Host "Example: .\rag-commands.ps1 cli-query -Query 'What is machine learning?'" -ForegroundColor Yellow
            return
        }
        docker-compose run --rm rag-cli uv run python -m rag_system.cli query "$Query"
    }
    "cli-interactive" {
        docker-compose run --rm rag-cli uv run python -m rag_system.cli interactive
    }
    "cli-clear" {
        docker-compose run --rm rag-cli uv run python -m rag_system.cli clear
    }
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Show-Help
    }
}
# 🚀 RAG System with UV and pyproject.toml

This guide shows how to set up and use the RAG system with modern Python tooling using `uv` (ultra-fast Python package manager) and `pyproject.toml`.

## 📋 Prerequisites

### Install UV

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: using pip
pip install uv

# Alternative: using pipx
pipx install uv
```

### Install Docker (for Qdrant)

```bash
# Install Docker from https://docker.com
# Or using package managers:

# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io

# macOS
brew install --cask docker

# Windows: Download Docker Desktop
```

## 🏗️ Project Setup

### 1. Initialize the Project

```bash
# Create project directory
mkdir rag-system-qdrant
cd rag-system-qdrant

# Initialize with uv (creates basic structure)
uv init

# Or clone if you have a git repository
git clone <your-repo-url>
cd rag-system-qdrant
```

### 2. Project Structure
Create this directory structure:

```
rag-system-qdrant/
├── pyproject.toml          # Project configuration
├── README.md              # Documentation
├── .gitignore            # Git ignore rules
├── .python-version       # Python version specification
├── rag_system/           # Main package
│   ├── __init__.py
│   ├── cli.py           # Modern CLI with Click & Rich
│   ├── config.py        # Configuration management
│   ├── cache.py         # In-memory caching implementation
│   ├── document_processor.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── llm_interface.py
├── tests/               # Test suite
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_document_processor.py
│   ├── test_embeddings.py
│   └── test_integration.py
├── docs/                # Documentation
├── sample_documents/    # Sample files for testing
├── config.yaml         # Runtime configuration
└── docker-compose.yml  # Docker services
```

### 3. Set Python Version

```bash
# Specify Python version (creates .python-version file)
echo "3.11" > .python-version

# Or use uv to set it
uv python pin 3.11
```

## 🔧 Development Workflow with UV

### 1. Create Virtual Environment

```bash
# UV automatically creates and manages virtual environments
# Create a virtual environment with specified Python version
uv venv --python 3.11

# Activate the virtual environment
# On Unix/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install all dependencies from pyproject.toml
uv sync

# Install specific packages
uv add openai tiktoken qdrant-client sentence-transformers
uv add click rich pydantic pyyaml python-dotenv

# Install development dependencies
uv add --group dev pytest pytest-cov black isort mypy ruff
uv add --group dev pytest-asyncio httpx

# Install optional dependencies
uv add --optional docs sphinx sphinx-rtd-theme
```

### 3. Development Commands

```bash
# Run the application
uv run python -m rag_system.cli --help

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=rag_system --cov-report=html

# Format code
uv run black rag_system/ tests/
uv run isort rag_system/ tests/

# Lint code
uv run ruff check rag_system/ tests/
uv run mypy rag_system/

# Run all quality checks
uv run python -m pytest && uv run black --check . && uv run ruff check . && uv run mypy rag_system/
```

## 📄 Configuration Files

### pyproject.toml

```toml
[project]
name = "rag-system-qdrant"
version = "0.1.0"
description = "A comprehensive RAG system using Qdrant vector database"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
keywords = ["rag", "llm", "vector-database", "qdrant", "embeddings"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
requires-python = ">=3.11"

dependencies = [
    "openai>=1.0.0",
    "tiktoken>=0.5.0",
    "qdrant-client>=1.6.0",
    "sentence-transformers>=2.2.0",
    "click>=8.1.0",
    "rich>=13.0.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0.0",
    "python-dotenv>=1.0.0",
    "aiofiles>=23.0.0",
    "httpx>=0.25.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.5.0",
    "ruff>=0.0.290",
    "pre-commit>=3.4.0",
]
docs = [
    "sphinx>=7.0.0",
    "sphinx-rtd-theme>=1.3.0",
    "myst-parser>=2.0.0",
]
test = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "httpx>=0.25.0",
]

[project.scripts]
rag-system = "rag_system.cli:main"

[project.urls]
Homepage = "https://github.com/yourusername/rag-system-qdrant"
Documentation = "https://rag-system-qdrant.readthedocs.io/"
Repository = "https://github.com/yourusername/rag-system-qdrant.git"
Issues = "https://github.com/yourusername/rag-system-qdrant/issues"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["rag_system"]

[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["rag_system"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = [
    "qdrant_client.*",
    "sentence_transformers.*",
    "tiktoken.*",
]
ignore_missing_imports = true

[tool.ruff]
target-version = "py311"
line-length = 88
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
]
ignore = [
    "E501",  # line too long, handled by black
    "B008",  # do not perform function calls in argument defaults
    "C901",  # too complex
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]

[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --strict-markers --strict-config"
testpaths = ["tests"]
filterwarnings = [
    "error",
    "ignore::UserWarning",
    "ignore::DeprecationWarning",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]

[tool.coverage.run]
source = ["rag_system"]
omit = [
    "*/tests/*",
    "*/venv/*",
    "*/.venv/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\bProtocol\):",
    "@(abc\.)?abstractmethod",
]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: rag_qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  qdrant_data:
```

### config.yaml

```yaml
# RAG System Configuration
app:
  name: "RAG System with Qdrant"
  version: "0.1.0"
  debug: false
  log_level: "INFO"

# Vector Database Configuration
qdrant:
  host: "localhost"
  port: 6333
  collection_name: "documents"
  vector_size: 384  # sentence-transformers/all-MiniLM-L6-v2
  distance: "Cosine"
  timeout: 30
  prefer_grpc: false

# Embedding Model Configuration
embeddings:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"  # or "cuda" if GPU available
  batch_size: 32
  max_seq_length: 512

# LLM Configuration
llm:
  provider: "openai"  # openai, anthropic, local
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 1000
  timeout: 30

# Document Processing
document_processing:
  chunk_size: 1000
  chunk_overlap: 200
  supported_formats: ["txt", "pdf", "docx", "md", "json"]
  max_file_size_mb: 50

# Search Configuration
search:
  top_k: 5
  score_threshold: 0.7
  rerank: true
  hybrid_search: false

# Caching
cache:
  enabled: true
  backend: "memory"  # memory only (no redis)
  ttl: 3600  # seconds
  max_size: 1000  # for memory backend

# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  workers: 1
  reload: true
```

### .env.example

```bash
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Database URLs
QDRANT_URL=http://localhost:6333

# Application Settings
LOG_LEVEL=INFO
DEBUG=false
ENVIRONMENT=development

# Optional: Custom model paths
EMBEDDING_MODEL_PATH=/path/to/custom/model
LLM_MODEL_PATH=/path/to/local/llm
```

### .gitignore

```gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
poetry.lock

# pdm
.pdm.toml

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
.idea/

# VS Code
.vscode/

# Local data
data/
documents/
uploads/
*.db
*.sqlite

# Model files
models/
*.pkl
*.joblib

# Logs
logs/
*.log

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
```

## 🚀 Quick Start

### 1. Start Services

```bash
# Start only Qdrant (no Redis needed)
docker-compose up -d qdrant

# Check service is running
docker-compose ps
```

### 2. Install and Setup

```bash
# Install all dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Initialize the system
uv run python -m rag_system.cli init
```

### 3. Add Documents

```bash
# Add a single document
uv run python -m rag_system.cli add-document path/to/document.pdf

# Add multiple documents from a directory
uv run python -m rag_system.cli add-documents path/to/documents/

# Add with specific collection
uv run python -m rag_system.cli add-document document.pdf --collection my-docs
```

### 4. Query the System

```bash
# Simple query
uv run python -m rag_system.cli query "What is machine learning?"

# Query with options
uv run python -m rag_system.cli query "Explain neural networks" --top-k 10 --collection my-docs

# Interactive mode
uv run python -m rag_system.cli interactive
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=rag_system --cov-report=html

# Run specific test types
uv run pytest -m unit
uv run pytest -m integration
uv run pytest -m "not slow"

# Run tests in parallel
uv run pytest -n auto
```

### Test Structure

```bash
tests/
├── conftest.py              # Shared fixtures
├── test_unit/              # Unit tests
│   ├── test_config.py
│   ├── test_embeddings.py
│   └── test_document_processor.py
├── test_integration/       # Integration tests
│   ├── test_vector_store.py
│   └── test_end_to_end.py
└── test_fixtures/          # Test data
    ├── sample_documents/
    └── mock_responses/
```

## 🔄 Workflow Commands

### Development Workflow

```bash
# Start development environment
uv sync --dev
source .venv/bin/activate

# Pre-commit setup
uv run pre-commit install

# Run quality checks
uv run black .
uv run isort .
uv run ruff check .
uv run mypy rag_system/

# Test before commit
uv run pytest --cov=rag_system

# Build package
uv build

# Publish to PyPI (if configured)
uv publish
```

### Production Deployment

```bash
# Install production dependencies only
uv sync --no-dev

# Run with production config
uv run python -m rag_system.cli --config config.prod.yaml

# Or use Docker
docker build -t rag-system .
docker run -p 8000:8000 rag-system
```

## 📊 Monitoring and Maintenance

### Health Checks

```bash
# Check system health
uv run python -m rag_system.cli health

# Check vector store
uv run python -m rag_system.cli status --component qdrant

# View metrics
uv run python -m rag_system.cli metrics
```

### Maintenance

```bash
# Clean up old vectors
uv run python -m rag_system.cli cleanup --days 30

# Reindex documents
uv run python -m rag_system.cli reindex --collection my-docs

# Backup vector store
uv run python -m rag_system.cli backup --output backup.tar.gz

# Restore from backup
uv run python -m rag_system.cli restore --input backup.tar.gz
```

## 🎯 Best Practices

1. **Dependency Management**: Use `uv add` instead of pip for consistency
2. **Environment Isolation**: Always work within the UV-managed virtual environment
3. **Configuration**: Use `config.yaml` for application settings, `.env` for secrets
4. **Testing**: Write tests for all new features, maintain >80% coverage
5. **Code Quality**: Use pre-commit hooks and run quality checks before commits
6. **Documentation**: Keep README and docstrings updated
7. **Versioning**: Use semantic versioning and update `pyproject.toml`

## 🔧 Troubleshooting

### Common Issues

```bash
# UV not found
export PATH="$HOME/.cargo/bin:$PATH"

# Python version issues
uv python install 3.11
uv python pin 3.11

# Dependencies not syncing
uv sync --reinstall

# Qdrant connection issues
docker-compose logs qdrant
```

### Performance Optimization

```bash
# Use GPU for embeddings (if available)
# Edit config.yaml: embeddings.device = "cuda"

# Enable parallel processing
# Edit config.yaml: document_processing.batch_size = 64

# Use hybrid search for better results
# Edit config.yaml: search.hybrid_search = true
```

This comprehensive setup provides a modern, maintainable RAG system with excellent developer experience using UV and pyproject.toml!
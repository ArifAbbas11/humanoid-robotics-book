# Quickstart: Book RAG Content Ingestion Pipeline

## Prerequisites

- Python 3.11+
- `uv` package manager installed (`pip install uv` or via package manager)
- Cohere API key
- Qdrant Cloud account and API key

## Setup

1. **Create the backend directory:**
   ```bash
   mkdir backend
   cd backend
   ```

2. **Initialize the project:**
   ```bash
   uv init
   ```

3. **Create virtual environment and install dependencies:**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv add requests beautifulsoup4 cohere qdrant-client python-dotenv
   ```

4. **Create environment file:**
   ```bash
   touch .env
   ```

5. **Add required environment variables to `.env`:**
   ```
   COHERE_API_KEY=your_cohere_api_key_here
   QDRANT_URL=your_qdrant_cluster_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```

## Usage

1. **Create the main.py file** with the implementation as specified

2. **Run the ingestion pipeline:**
   ```bash
   python main.py
   ```

## Configuration

The pipeline supports configurable parameters for:
- Chunk size (default: 1000 characters)
- Chunk overlap (default: 100 characters)
- Source URL (default: https://arifabbas11.github.io/humanoid-robotics-book/)

## Verification

After running the pipeline:
1. Check that the "rag_embedding" collection exists in your Qdrant Cloud instance
2. Verify that embeddings have been stored with proper metadata
3. Run a test similarity search to confirm functionality
# Book RAG Content Ingestion Pipeline

This project implements a backend ingestion pipeline that crawls the deployed Docusaurus book website, extracts content, chunks it, generates Cohere embeddings, and stores them in Qdrant Cloud for RAG (Retrieval-Augmented Generation) applications.

## Features

- Crawls the Docusaurus book website to extract content
- Extracts clean text content while preserving semantic structure
- Chunks text with configurable size and overlap
- Generates semantic embeddings using Cohere
- Stores embeddings with metadata in Qdrant Cloud vector database
- Implements error handling and retry logic
- Supports idempotent operations to prevent duplicate entries

## Prerequisites

- Python 3.11+
- uv package manager
- Cohere API key
- Qdrant Cloud account and API key

## Installation

1. Clone the repository
2. Navigate to the backend directory: `cd backend`
3. Install dependencies using uv: `uv sync` or `uv pip install -r requirements.txt`

## Configuration

Create a `.env` file in the backend directory with the following variables:

```env
COHERE_API_KEY=your_cohere_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
BOOK_BASE_URL=https://arifabbas11.github.io/humanoid-robotics-book/
SITEMAP_URL=https://arifabbas11.github.io/humanoid-robotics-book/sitemap.xml
CHUNK_SIZE=512
CHUNK_OVERLAP=50
```

## Usage

The main pipeline functionality is implemented in `main.py` with the following key functions:

- `get_all_urls()`: Collects all URLs from the book website
- `extract_text_from_url()`: Extracts clean text content from a URL
- `chunk_text()`: Chunks text with configurable size and overlap
- `embed()`: Generates embeddings using Cohere
- `create_collection()`: Creates Qdrant collection for "rag_embedding"
- `save_chunk_to_qdrant()`: Saves chunked content to Qdrant
- `main()`: Orchestrates the complete pipeline

Run the pipeline with: `python main.py`

## Architecture

The implementation follows a single-file architecture in `main.py` as specified, containing all necessary functions for the ingestion pipeline. The pipeline processes content in the following order:

1. Content Extraction (US1)
2. Embedding Generation (US2)
3. Vector Storage (US3)

## Dependencies

- requests: For HTTP requests
- beautifulsoup4: For HTML parsing
- cohere: For embedding generation
- qdrant-client: For vector database operations
- python-dotenv: For environment variable management
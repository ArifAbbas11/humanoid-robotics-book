# Research: Book RAG Content Ingestion Pipeline

## Decision: Technology Stack
**Rationale**: Selected Python with specific libraries to meet all requirements:
- `requests` and `beautifulsoup4` for web crawling and content extraction
- `cohere` for semantic embeddings generation
- `qdrant-client` for vector database operations
- `python-dotenv` for environment management
- `uv` for fast dependency management (as requested)

**Alternatives considered**:
- Alternative embedding models (OpenAI, Hugging Face) - Cohere was specifically required
- Alternative vector databases (Pinecone, Weaviate) - Qdrant Cloud was specifically required
- Alternative web scraping tools (Selenium, Scrapy) - Requests + BeautifulSoup is lightweight and sufficient for static site

## Decision: Single File Architecture
**Rationale**: As requested in requirements, implemented as a single main.py file with specific functions:
- `get_all_urls`: Discover all book pages from the Docusaurus site
- `extract_text_from_url`: Extract clean text content from each page
- `chunk_text`: Split content into configurable chunks with overlap
- `embed`: Generate Cohere embeddings for each chunk
- `create_collection`: Set up Qdrant collection named "rag_embedding"
- `save_chunk_to_qdrant`: Store embeddings with metadata in Qdrant

## Decision: Content Extraction Strategy
**Rationale**: For Docusaurus sites, focus on main content areas while preserving semantic structure. Extract text from article/main tags while filtering out navigation, headers, and other non-content elements.

**Alternatives considered**:
- Using Docusaurus API directly - Not available for deployed GitHub Pages site
- Using headless browser - More complex than needed for static content

## Decision: Chunking Strategy
**Rationale**: Implement configurable chunk size with overlap to maintain context while allowing flexibility. Default to 512-1024 tokens with 10-20% overlap as best practice for semantic search.

## Decision: Qdrant Collection Design
**Rationale**: Create collection named "rag_embedding" as specifically requested. Store metadata including URL, section, chunk ID, and content for proper retrieval context.

## Decision: Error Handling and Resilience
**Rationale**: Implement retry logic for API calls, graceful handling of malformed pages, and idempotent operations to support re-runs as required by specification.
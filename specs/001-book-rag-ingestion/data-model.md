# Data Model: Book RAG Content Ingestion Pipeline

## Entities

### Book Content Chunk
**Purpose**: Represents a segment of book text with preserved semantic context
**Fields**:
- `id` (string): Unique identifier for the chunk
- `content` (string): The actual text content of the chunk
- `source_url` (string): Original URL where the content was found
- `section` (string): Section or page identifier from the book structure
- `chunk_index` (integer): Sequential position of this chunk within the source
- `created_at` (datetime): Timestamp when chunk was processed
- `metadata` (dict): Additional metadata including headings, context info

### Embedding Vector
**Purpose**: Represents the semantic vector representation of a content chunk
**Fields**:
- `chunk_id` (string): Reference to the source content chunk
- `vector` (list[float]): The embedding vector values from Cohere
- `vector_size` (integer): Dimension of the embedding vector
- `embedding_model` (string): Name/version of the model used
- `created_at` (datetime): Timestamp when embedding was generated

### Processing Metadata
**Purpose**: Contains information about pipeline execution
**Fields**:
- `run_id` (string): Unique identifier for this pipeline execution
- `start_time` (datetime): When the processing started
- `end_time` (datetime): When the processing completed
- `status` (string): Current status (running, completed, failed)
- `processed_urls` (list[string]): URLs that were successfully processed
- `failed_urls` (list[string]): URLs that failed during processing
- `total_chunks` (integer): Number of chunks created in this run

## Qdrant Collection Schema

### Collection: rag_embedding
**Configuration**:
- `name`: "rag_embedding"
- `vector_size`: 1024 (for Cohere embeddings)
- `distance`: Cosine similarity

**Payload Structure**:
- `chunk_id` (keyword): Unique identifier for the chunk
- `content` (text): Original text content
- `source_url` (keyword): Original URL of the content
- `section` (keyword): Section identifier
- `chunk_index` (integer): Position in source
- `created_at` (datetime): Timestamp
- `metadata` (object): Additional context information

## Relationships

- Each `Book Content Chunk` generates one `Embedding Vector`
- Multiple `Book Content Chunks` are created from each processed URL
- Each pipeline `Processing Metadata` record tracks multiple chunks
- `Embedding Vector` records are stored in Qdrant `rag_embedding` collection
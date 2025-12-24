"""
RAG Agent Service using OpenAI Agents SDK and FastAPI
Provides a question-answering API that uses retrieval-augmented generation
to answer questions based on the book content.
"""

import asyncio
import os
import uuid
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cohere
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
import openai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize clients
cohere_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=10
)

# For free inference, we can implement a simple rule-based approach
# or connect to a free API service if available

# Constants
QDRANT_COLLECTION_NAME = "rag_embedding"
VECTOR_SIZE = 1024  # Cohere embedding dimension
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5  # Number of chunks to retrieve
    filters: Optional[Dict[str, Any]] = None  # Optional metadata filters


class RetrievedChunk(BaseModel):
    content: str
    similarity_score: float
    source_url: str
    chunk_id: str


class AnswerResponse(BaseModel):
    answer: str
    confidence_level: str
    retrieved_chunks: List[RetrievedChunk]
    processing_time: float
    query_id: str


# Initialize FastAPI app
app = FastAPI(
    title="RAG Agent Service",
    description="Question-answering API using Retrieval-Augmented Generation",
    version="1.0.0"
)

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def retrieve_chunks(query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Retrieve relevant content chunks from Qdrant based on the query.

    Args:
        query: The query text to search for
        top_k: Number of chunks to retrieve
        filters: Optional metadata filters to apply

    Returns:
        List of retrieved chunks with metadata
    """
    try:
        # Check if collection exists
        try:
            qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION_NAME)
        except:
            logger.error(f"Collection {QDRANT_COLLECTION_NAME} does not exist or is not accessible")
            return []

        # Generate embedding for the query using Cohere
        response = cohere_client.embed(
            texts=[query],
            model='embed-english-v3.0',
            input_type='search_query'
        )

        query_embedding = response.embeddings[0]

        # Prepare filters for Qdrant
        qdrant_filters = None
        if filters:
            must_conditions = []
            for key, value in filters.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )

            if must_conditions:
                qdrant_filters = models.Filter(must=must_conditions)

        # Query in Qdrant
        search_response = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            query_filter=qdrant_filters,
            with_payload=True
        )

        # Format results - query_points returns a QueryResponse object with points
        retrieved_chunks = []
        for result in search_response.points:
            chunk_data = {
                'content': result.payload.get('text', '') if result.payload else '',
                'similarity_score': result.score,
                'source_url': result.payload.get('source_url', '') if result.payload else '',
                'chunk_id': result.id
            }
            retrieved_chunks.append(chunk_data)

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: {query[:50]}...")
        return retrieved_chunks

    except Exception as e:
        logger.error(f"Error retrieving chunks: {str(e)}")
        return []


async def generate_answer_with_free_agent(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Generate an answer using a simple approach with the provided context.

    Args:
        question: The question to answer
        context_chunks: Retrieved context chunks to ground the response

    Returns:
        Generated answer string
    """
    try:
        # Combine context chunks into a single context string
        context_text = "\n\n".join([f"Source: {chunk['source_url']}\nContent: {chunk['content']}"
                                   for chunk in context_chunks])

        # For a truly free solution, return a summary based on the most relevant context
        if context_chunks:
            # Return the most relevant chunk as the answer
            most_relevant = max(context_chunks, key=lambda x: x['similarity_score'])
            return f"Based on the book content: {most_relevant['content'][:500]}..."  # Limit length
        else:
            return "I couldn't find relevant information in the book to answer your question."

    except Exception as e:
        logger.error(f"Error generating answer with free agent: {str(e)}")
        return "I couldn't generate a response based on the provided context."


@app.post("/api/v1/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Endpoint to ask a question and get an answer based on the book content.

    Args:
        request: Question and parameters

    Returns:
        Answer with supporting information
    """
    start_time = time.time()

    try:
        logger.info(f"Processing question: {request.question}")

        # Retrieve relevant chunks from Qdrant
        retrieved_chunks = retrieve_chunks(
            query=request.question,
            top_k=request.top_k,
            filters=request.filters
        )

        if not retrieved_chunks:
            logger.warning(f"No relevant chunks found for question: {request.question}")
            return AnswerResponse(
                answer="I couldn't find relevant information in the book to answer your question.",
                confidence_level="none",
                retrieved_chunks=[],
                processing_time=time.time() - start_time,
                query_id=str(uuid.uuid4())  # Generate a proper UUID
            )

        # Generate answer using free agent with the retrieved context
        answer = await generate_answer_with_free_agent(request.question, retrieved_chunks)

        # Calculate confidence based on similarity scores
        avg_similarity = sum(chunk['similarity_score'] for chunk in retrieved_chunks) / len(retrieved_chunks)
        if avg_similarity > 0.7:
            confidence_level = "high"
        elif avg_similarity > 0.4:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        # Format retrieved chunks for response
        formatted_chunks = [
            RetrievedChunk(
                content=chunk['content'],
                similarity_score=chunk['similarity_score'],
                source_url=chunk['source_url'],
                chunk_id=chunk['chunk_id']
            )
            for chunk in retrieved_chunks
        ]

        processing_time = time.time() - start_time

        logger.info(f"Generated answer in {processing_time:.2f}s with {len(retrieved_chunks)} chunks")

        return AnswerResponse(
            answer=answer,
            confidence_level=confidence_level,
            retrieved_chunks=formatted_chunks,
            processing_time=processing_time,
            query_id=str(uuid.uuid4())  # Generate a proper UUID
        )

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Import necessary modules for content ingestion (add these at the top with other imports if not present)
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import hashlib
import random

def create_session_with_retries(
    retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: tuple = (500, 502, 504)
) -> requests.Session:
    """
    Create a requests session with retry configuration.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def safe_request(method: str, url: str, **kwargs) -> requests.Response:
    """
    Make a safe HTTP request with error handling.
    """
    try:
        session = create_session_with_retries()
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 30
        headers = kwargs.get('headers', {})
        headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        kwargs['headers'] = headers
        response = session.request(method, url, **kwargs)
        return response
    except Exception as e:
        logger.error(f"Request failed for {url}: {str(e)}")
        return None

def is_valid_url(url: str, base_domain: str = None) -> bool:
    """
    Check if a URL is valid and optionally belongs to the base domain.
    """
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ['http', 'https']:
            return False
        if not parsed_url.netloc:
            return False
        if base_domain and base_domain not in parsed_url.netloc:
            return False
        return True
    except Exception:
        return False

def extract_text_from_url(url: str) -> str:
    """
    Extract clean text content from a URL.
    """
    try:
        response = safe_request("GET", url)
        if not response:
            return ""

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside", "meta", "link"]):
            script.decompose()

        content_element = soup.find('body')
        if not content_element:
            return ""

        text_parts = []
        for element in content_element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'div', 'span', 'code', 'pre'], recursive=True):
            text = element.get_text(strip=True)
            if text:
                if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    text_parts.append(f"\n\n{element.name.upper()}: {text}\n")
                elif element.name == 'li':
                    text_parts.append(f"  - {text}\n")
                elif element.name in ['p', 'div']:
                    text_parts.append(f"{text}\n")
                elif element.name in ['code', 'pre']:
                    text_parts.append(f"```\n{text}\n```\n")
                else:
                    text_parts.append(f"{text} ")

        content = " ".join(text_parts)
        import re
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        content = content.strip()

        return content
    except Exception as e:
        logger.error(f"Error extracting text from URL {url}: {str(e)}")
        return ""

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into chunks with specified size and overlap.
    """
    if not text:
        return []
    if chunk_size <= 0:
        return []
    if overlap < 0 or overlap >= chunk_size:
        overlap = 0

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]

        if len(chunk) < chunk_size and start > 0:
            if len(chunk) < chunk_size // 2 and len(chunks) > 0:
                if len(chunks) > 0:
                    chunks[-1] = chunks[-1] + chunk
                break

        chunks.append(chunk)
        start = end - overlap

        if text_len - end < overlap and overlap > 0:
            break

    return chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using Cohere.
    """
    if not texts:
        return []

    # Validate input texts
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            return None
        if len(text) == 0:
            texts[i] = " "

    MAX_BATCH_SIZE = 96  # Cohere's typical batch limit
    all_embeddings = []

    try:
        # Process texts in batches if needed
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i:i + MAX_BATCH_SIZE]

            # Generate embeddings using Cohere
            response = cohere_client.embed(
                texts=batch,
                model='embed-english-v3.0',
                input_type='search_document'
            )

            if response and hasattr(response, 'embeddings') and response.embeddings:
                batch_embeddings = response.embeddings
                all_embeddings.extend(batch_embeddings)
            else:
                logger.error(f"Batch {i//MAX_BATCH_SIZE + 1}: Cohere API returned empty or invalid response")
                return None

        return all_embeddings

    except Exception as e:
        logger.error(f"Cohere API error during embedding: {str(e)}")
        return None

def save_chunk_to_qdrant(text_chunk: str, embedding: List[float], source_url: str = None) -> bool:
    """
    Save a text chunk with its embedding to Qdrant.
    """
    if not text_chunk or not embedding:
        return False

    # Validate embedding dimension
    if len(embedding) != 1024:  # Cohere embedding dimension
        return False

    # Generate a unique ID for this chunk based on its content
    chunk_id = hashlib.md5(f"{text_chunk}{str(embedding)[:50]}".encode()).hexdigest()

    try:
        # Prepare payload with the text chunk
        payload = {
            "text": text_chunk,
            "source_url": source_url or "",
            "timestamp": time.time(),
            "chunk_length": len(text_chunk)
        }

        # Upsert the point (this will update if it exists, or create if it doesn't)
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload=payload
                )
            ]
        )
        return True
    except Exception as e:
        logger.error(f"Error saving chunk to Qdrant: {str(e)}")
        return False

@app.post("/api/v1/ingest")
async def ingest_content():
    """
    Endpoint to trigger content ingestion from the book website.
    """
    try:
        logger.info("Starting content ingestion process...")

        # Get the book URLs - for now using the main book URL
        # In a real implementation, you would fetch the sitemap to get all URLs
        book_base_url = "https://arifabbas11.github.io/humanoid-robotics-book/"
        urls_to_process = [
            book_base_url,
            f"{book_base_url}intro",
            f"{book_base_url}ros-fundamentals/intro",
            f"{book_base_url}simulation/intro",
            f"{book_base_url}ai-navigation/intro",
            f"{book_base_url}vla-integration/intro"
        ]

        successful_count = 0
        failed_count = 0

        for url in urls_to_process[:5]:  # Process first 5 URLs to avoid timeout
            logger.info(f"Processing URL: {url}")

            content = extract_text_from_url(url)
            if not content:
                logger.warning(f"Failed to extract content from {url}")
                failed_count += 1
                continue

            logger.info(f"Extracted {len(content)} characters from {url}")

            # Chunk the content
            chunks = chunk_text(content, CHUNK_SIZE, CHUNK_OVERLAP)
            if not chunks:
                logger.warning(f"No chunks generated from content in {url}")
                failed_count += 1
                continue

            logger.info(f"Content chunked into {len(chunks)} pieces")

            # Generate embeddings for chunks
            embeddings = embed_texts(chunks)
            if not embeddings or len(embeddings) != len(chunks):
                logger.error(f"Failed to generate embeddings for {url}")
                failed_count += 1
                continue

            logger.info(f"Generated {len(embeddings)} embeddings successfully")

            # Save each chunk with its embedding to Qdrant
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                if save_chunk_to_qdrant(chunk, embedding, source_url=url):
                    logger.debug(f"Saved chunk {i+1}/{len(chunks)} for {url}")
                else:
                    logger.warning(f"Failed to save chunk {i+1}/{len(chunks)} for {url}")

            successful_count += 1
            logger.info(f"Successfully processed URL: {url}")

        result = {
            "status": "success",
            "message": f"Processed {successful_count} out of {len(urls_to_process[:5])} URLs successfully",
            "successful": successful_count,
            "failed": failed_count
        }

        logger.info(f"Content ingestion completed: {result}")
        return result

    except Exception as e:
        logger.error(f"Error during content ingestion: {str(e)}")
        return {"status": "error", "message": f"Content ingestion failed: {str(e)}"}

@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "healthy", "service": "RAG Agent Service"}


# For running the service directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
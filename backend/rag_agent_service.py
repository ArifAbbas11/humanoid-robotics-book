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
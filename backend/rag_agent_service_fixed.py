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
from pydantic import BaseModel
import cohere
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
import openai
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue

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

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

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

        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qdrant_filters,
            with_payload=True
        )

        # Format results
        retrieved_chunks = []
        for result in search_results:
            chunk_data = {
                'content': result.payload.get('text', ''),
                'similarity_score': result.score,
                'source_url': result.payload.get('source_url', ''),
                'chunk_id': result.id
            }
            retrieved_chunks.append(chunk_data)

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: {query[:50]}...")
        return retrieved_chunks

    except Exception as e:
        logger.error(f"Error retrieving chunks: {str(e)}")
        return []


async def generate_answer_with_openai_agent(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Generate an answer using OpenAI's agent with the provided context.

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

        # Create a system message to constrain the agent to use only the provided context
        system_message = f"""
        You are a helpful assistant for a book about humanoid robotics.
        Answer the user's question based ONLY on the provided context below.
        Do not use any prior knowledge or information not contained in the context.
        If the context doesn't contain enough information to answer the question,
        say so explicitly.

        CONTEXT:
        {context_text}
        """

        # Create the OpenAI client
        client = openai.OpenAI()

        # Create a thread with the user's question
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        # Create an assistant that uses the context
        assistant = client.beta.assistants.create(
            name="Book RAG Assistant",
            instructions=system_message,
            model="gpt-4-turbo-preview",  # Using a capable model
            tools=[]  # No tools needed since we're providing the context directly
        )

        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )

        # Wait for the run to complete
        while run.status in ["queued", "in_progress"]:
            await asyncio.sleep(0.5)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )

        # Get the messages from the thread
        messages = client.beta.threads.messages.list(thread_id=thread.id)

        # Extract the assistant's response
        for msg in messages.data:
            if msg.role == "assistant":
                # Extract text content from the message
                for content_block in msg.content:
                    if hasattr(content_block, 'text') and content_block.text:
                        return content_block.text.value

        # If no response found, return a default message
        return "I couldn't generate a response based on the provided context."

    except Exception as e:
        logger.error(f"Error generating answer with OpenAI agent: {str(e)}")
        return f"Sorry, I encountered an error processing your request: {str(e)}"


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

        # Generate answer using OpenAI agent with the retrieved context
        answer = await generate_answer_with_openai_agent(request.question, retrieved_chunks)

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
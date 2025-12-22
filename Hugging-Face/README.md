# RAG Agent Service for Humanoid Robotics Book

This repository contains a RAG (Retrieval-Augmented Generation) agent service that allows users to ask questions about the Humanoid Robotics book content.

## Overview

The RAG agent service provides a question-answering API that uses retrieval-augmented generation to answer questions based on the book content. It uses:
- Cohere for generating text embeddings
- Qdrant as a vector database for similarity search
- FastAPI for the web API framework
- Beautiful Soup for content extraction from web pages

## Deployment on Hugging Face

This service is designed for deployment on Hugging Face Spaces using Docker.

### Required Environment Variables

Set these secrets in your Hugging Face Space settings:

- `COHERE_API_KEY`: Your Cohere API key for generating embeddings
- `QDRANT_URL`: URL for your Qdrant vector database
- `QDRANT_API_KEY`: API key for your Qdrant database
- `BOOK_BASE_URL`: Base URL for the book content (default: https://arifabbas11.github.io/humanoid-robotics-book/)

### API Endpoints

- `POST /api/v1/ask`: Ask a question about the book content
- `GET /api/v1/health`: Health check endpoint

## Usage Example

### Asking a Question

```json
{
  "question": "What are the main components of a humanoid robot?",
  "top_k": 5
}
```

### Response Format

```json
{
  "answer": "The main components of a humanoid robot...",
  "confidence_level": "high",
  "retrieved_chunks": [...],
  "processing_time": 1.23,
  "query_id": "unique-id"
}
```

## Architecture

The system works in the following steps:
1. Extracts content from the book website using web scraping
2. Chunks the content into smaller pieces
3. Generates embeddings for each chunk using Cohere
4. Stores embeddings in Qdrant vector database
5. For incoming questions, retrieves relevant chunks based on semantic similarity
6. Generates answers based on the retrieved context

## License

This project is open source under the MIT license.
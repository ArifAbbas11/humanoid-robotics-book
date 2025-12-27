---
title: Humanoid Robotics Book Assistant
emoji: 🤖
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
license: mit
---

# Humanoid Robotics Book Assistant

This application provides a chatbot interface to ask questions about humanoid robotics based on the content from the "Physical AI & Humanoid Robotics Book".

## About

This RAG (Retrieval-Augmented Generation) system allows users to ask questions about humanoid robotics, ROS 2, simulation, AI navigation, and Vision-Language-Action integration. The system retrieves relevant content from the book and generates contextual answers.

## Features

- Ask questions about humanoid robotics concepts
- Get answers grounded in book content
- See sources for the information provided
- Confidence scoring for answers

## Environment Variables Required

To run this application, you need to set the following environment variables in the Space Secrets:

- `COHERE_API_KEY`: Your Cohere API key for generating embeddings
- `QDRANT_URL`: URL for your Qdrant vector database
- `QDRANT_API_KEY`: API key for your Qdrant database

## How to Use

1. After deployment, trigger content ingestion by calling the `/api/v1/ingest` endpoint (this populates the vector database with book content)
2. Once content is ingested, ask questions using the `/api/v1/ask` endpoint
3. The system will search through the book content and provide answers
4. View the sources used to generate the answer

## Technology Stack

- Gradio: Web interface
- FastAPI: Backend API
- Cohere: Text embeddings
- Qdrant: Vector database
- Python: Backend processing

## Update

Updated: Now processes all URLs from the sitemap for comprehensive book content.

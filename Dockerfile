# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend files into the container
COPY backend/ ./backend/

# Expose the port that the app runs on
EXPOSE 8000

# Set environment variables (these will be configured in Hugging Face Spaces)
ENV PYTHONPATH=/app

# Run the application
CMD ["uvicorn", "backend.rag_agent_service:app", "--host", "0.0.0.0", "--port", "8000"]
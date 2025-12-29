#!/bin/bash

# Start servers script for running both FastAPI and Streamlit

# Start FastAPI backend in background
echo "Starting FastAPI backend on port 8000..."
uvicorn fastapi_backend:app --host 0.0.0.0 --port ${FASTAPI_PORT:-8000} &
FASTAPI_PID=$!

# Wait a moment for FastAPI to start
sleep 3

# Start Streamlit frontend
echo "Starting Streamlit frontend on port ${PORT:-8501}..."
streamlit run app.py --server.port ${PORT:-8501} --server.enableCORS false --server.enableXsrfProtection false &
STREAMLIT_PID=$!

# Wait for both processes
wait $FASTAPI_PID $STREAMLIT_PID

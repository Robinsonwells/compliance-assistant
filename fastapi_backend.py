"""
FastAPI Backend for Cloudflare-safe background LLM processing.

This backend implements the Asynchronous Request-Reply pattern:
1. POST /api/chat - Submit query, return response_id immediately
2. GET /api/response/{response_id} - Poll for status and partial response
3. GET /api/response/{response_id}/final - Retrieve final response

Key principles:
- No HTTP request waits for full LLM response
- Background tasks run independently of HTTP lifecycle
- Cloudflare 100s timeout is never hit
"""
import os
import uuid
import asyncio
import json
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from background_response_manager import BackgroundResponseManager
from openai_background_processor import process_openai_background

load_dotenv()

app = FastAPI(title="Legal Assistant Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = BackgroundResponseManager()


class ChatRequest(BaseModel):
    query: str
    session_id: str
    access_code: str
    reasoning_effort: str = "medium"
    search_results: Optional[List[Dict[str, Any]]] = None


class ChatResponse(BaseModel):
    response_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    response_id: str
    status: str
    partial_response: str
    sequence_number: int
    time_to_first_token: Optional[float] = None
    error_message: Optional[str] = None


class FinalResponse(BaseModel):
    response_id: str
    status: str
    text: str
    token_usage: Dict[str, int]
    time_to_first_token: Optional[float] = None


@app.post("/api/chat", response_model=ChatResponse)
async def submit_chat(request: ChatRequest):
    """
    Step 1: Accept user query and immediately return a response_id.

    This endpoint MUST:
    - Create a pending_responses row
    - Start background processing
    - Return {response_id, status='queued'} quickly (< 1000ms)

    Constraints:
    - Must NOT block on OpenAI
    - Must NOT await the background task
    """
    response_id = str(uuid.uuid4())

    db_id = manager.create_pending_response(
        response_id=response_id,
        access_code=request.access_code,
        session_id=request.session_id,
        user_query=request.query,
        search_results=request.search_results,
        reasoning_effort=request.reasoning_effort,
    )

    if not db_id:
        raise HTTPException(
            status_code=500,
            detail="Failed to create pending response"
        )

    asyncio.create_task(
        process_openai_background(
            response_id=response_id,
            query=request.query,
            reasoning_effort=request.reasoning_effort,
            search_results=request.search_results or [],
            access_code=request.access_code,
            session_id=request.session_id,
        )
    )

    return ChatResponse(
        response_id=response_id,
        status="queued",
        message="Your query is being processed."
    )


@app.get("/api/response/{response_id}", response_model=StatusResponse)
async def get_response_progress(response_id: str):
    """
    Step 2: Frontend polls this every 0.5-2 seconds.

    Returns lightweight status and partial content.
    Must complete in tens of milliseconds (single DB read).
    """
    record = manager.get_pending_response(response_id)
    if not record:
        raise HTTPException(status_code=404, detail="Response not found")

    return StatusResponse(
        response_id=response_id,
        status=record.get("status", "unknown"),
        partial_response=record.get("partial_response", ""),
        sequence_number=record.get("sequence_number", 0),
        time_to_first_token=record.get("time_to_first_token"),
        error_message=record.get("error_message"),
    )


@app.get("/api/response/{response_id}/final", response_model=FinalResponse)
async def get_final_response(response_id: str):
    """
    Step 3: Retrieve final response (after status is 'completed').

    Client calls this when polling shows completed.
    """
    record = manager.get_pending_response(response_id)
    if not record:
        raise HTTPException(status_code=404, detail="Response not found")

    status = record.get("status")
    if status != "completed":
        raise HTTPException(
            status_code=202,
            detail=f"Not completed yet (status: {status})"
        )

    token_usage_raw = record.get("token_usage") or "{}"
    try:
        token_usage = json.loads(token_usage_raw) if isinstance(token_usage_raw, str) else token_usage_raw
    except Exception:
        token_usage = {}

    return FinalResponse(
        response_id=response_id,
        status="completed",
        text=record.get("final_response", ""),
        token_usage=token_usage,
        time_to_first_token=record.get("time_to_first_token"),
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("FASTAPI_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

"""
Background OpenAI Processor for true async LLM streaming.

This module handles long-running LLM calls completely decoupled from HTTP requests.
It streams from OpenAI, writes incremental progress to Supabase, and handles all errors.

Key invariants:
- Must not be tied to HTTP request lifecycle
- Updates Supabase via BackgroundResponseManager
- Maintains status transitions: queued → in_progress → streaming → completed/failed
"""
import os
import time
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI

from background_response_manager import BackgroundResponseManager
from system_prompts import LEGAL_COMPLIANCE_SYSTEM_PROMPT


openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=600.0
)

manager = BackgroundResponseManager()


async def process_openai_background(
    response_id: str,
    query: str,
    reasoning_effort: str,
    search_results: List[Dict[str, Any]],
    access_code: str,
    session_id: str,
):
    """
    Run in background. Stream from OpenAI and update Supabase via BackgroundResponseManager.

    Failure semantics:
    - On any exception, mark_failed(response_id, error_message)

    This function is allowed to run for several minutes without any HTTP timeout concerns.
    """
    try:
        manager.update_status(response_id, "in_progress")
        print(f"[BACKGROUND] Started processing response_id: {response_id}")

        context = ""
        for i, result in enumerate(search_results, 1):
            context += f"\n--- Source {i} ---\n"
            context += f"Citation: {result.get('citation', 'Unknown')}\n"
            context += f"Jurisdiction: {result.get('jurisdiction', 'Unknown')}\n"
            context += f"Content: {result.get('text', '')}\n"

        messages = [
            {"role": "system", "content": LEGAL_COMPLIANCE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}\n\nRelevant Legal Sources:\n{context}"}
        ]

        start_time = time.time()
        partial_text = ""
        seq = 0
        ttft_set = False

        stream = openai_client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            stream=True,
            reasoning={"effort": reasoning_effort},
            max_tokens=None,
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if not delta:
                continue

            if delta.content:
                text_chunk = delta.content
                partial_text += text_chunk
                seq += 1

                if not ttft_set:
                    ttft = time.time() - start_time
                    manager.set_time_to_first_token(response_id, ttft)
                    ttft_set = True
                    print(f"[BACKGROUND] First token received in {ttft:.2f}s")

                if seq % 50 == 0:
                    manager.update_partial_response(
                        response_id=response_id,
                        partial_text=partial_text[-5000:],
                        sequence_number=seq,
                    )

        total_time = time.time() - start_time
        print(f"[BACKGROUND] Streaming completed in {total_time:.1f}s, {len(partial_text)} chars")

        token_usage = {
            "output_chars": len(partial_text),
            "model": "gpt-5",
        }

        manager.mark_completed(
            response_id=response_id,
            final_response=partial_text,
            token_usage=token_usage,
        )

        print(f"[BACKGROUND] Response {response_id} marked as completed")

    except Exception as e:
        error_msg = f"Background processing error: {str(e)}"
        print(f"[BACKGROUND] ERROR: {error_msg}")
        manager.mark_failed(response_id, error_msg)
